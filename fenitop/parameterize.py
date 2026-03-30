import numpy as np
import ufl
from dolfinx import la
from dolfinx.fem import Function, form, Constant
from dolfinx.fem.petsc import create_matrix, assemble_matrix
from petsc4py import PETSc


class DensityFilter:
    def __init__(self, comm, rho, rho_tilde, R=1.0, petsc_options={}):
        """Construct a PDE filter from DG0 to CG1 space."""

        # * Initialization
        S0, S = rho.function_space, rho_tilde.function_space
        u0, u = ufl.TrialFunction(S0), ufl.TrialFunction(S)
        v, self.af = ufl.TestFunction(S), Function(S)

        self.rho, self.rho_tilde = rho, rho_tilde
        self.rho_tilde_wrap = la.create_petsc_vector_wrap(self.rho_tilde.x)
        self.af_wrap = la.create_petsc_vector_wrap(self.af.x)
        self.vec_s0, self.vec_s = rho.vector.copy(), rho_tilde.vector.copy()

        # * Construct Kf and T matrices based on the Helmholtz PDE
        dx = ufl.Measure("dx", metadata={"quadrature_degree": 2})
        r_pde = R / (2.0 * np.sqrt(3.0)) # Clausen et al. 2017 Eq (13)
        Kf_expr = (r_pde**2 * ufl.dot(ufl.grad(u), ufl.grad(v)) + u * v) * dx
        T_expr = u0 * v * dx
        Kf_form, T_form = form(Kf_expr), form(T_expr)
        Kf_mat, self.T_mat = create_matrix(Kf_form), create_matrix(T_form)

        # * Construct a filtering solver
        self.solver = PETSc.KSP().create(comm)
        self.solver.setOperators(Kf_mat)
        prefix = f"filter_solver_{id(self)}"
        self.solver.setOptionsPrefix(prefix)

        # * Apply PETSc options
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        Kf_mat.setOptionsPrefix(prefix)
        Kf_mat.setFromOptions()

        # * Assemble Kf and T matrices
        assemble_matrix(Kf_mat, Kf_form)
        Kf_mat.assemble()
        assemble_matrix(self.T_mat, T_form)
        self.T_mat.assemble()
        self.T_mat_transpose = self.T_mat.copy()
        self.T_mat_transpose.transpose()

    def forward(self):
        """Compute the filtered variables."""

        self.T_mat.mult(self.rho.vector, self.vec_s)
        self.solver.solve(self.vec_s, self.rho_tilde_wrap)
        self.rho_tilde.x.scatter_forward()
        return self.rho_tilde

    def backward(self, sf_vectors):
        
        """Recover the sensitivities."""
        values = []
        for sf in sf_vectors:
            if sf is not None:
                self.solver.solve(sf, self.af_wrap)
                self.af.x.scatter_forward()
                self.T_mat_transpose.mult(self.af.vector, self.vec_s0)
                values.append(self.vec_s0.array.copy())
            else:
                values.append(None)
        return values


class CG1Filter:
    """A PDE filter mapping from a CG1 space to a CG1 space.
    Used for secondary smoothing """

    def __init__(self, comm, V, u_out_func, R=1.0, petsc_options={}):
        self.V = V
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        self.u_in = Function(V)
        self.u_out = u_out_func
        self.u_out_wrap = la.create_petsc_vector_wrap(self.u_out.x)
        self.lambda_adj = Function(V)
        self.lambda_wrap = la.create_petsc_vector_wrap(self.lambda_adj.x)

        dx = ufl.Measure("dx", metadata={"quadrature_degree": 2})
        r_pde = R / (2.0 * np.sqrt(3.0))
        Kf_expr = (r_pde**2 * ufl.dot(ufl.grad(u), ufl.grad(v)) + u * v) * dx
        M_expr = u * v * dx  # Mass matrix for CG1 to CG1

        Kf_form, M_form = form(Kf_expr), form(M_expr)
        Kf_mat, self.M_mat = create_matrix(Kf_form), create_matrix(M_form)

        self.solver = PETSc.KSP().create(comm)
        self.solver.setOperators(Kf_mat)
        prefix = f"cg1filter_{id(self)}"
        self.solver.setOptionsPrefix(prefix)

        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        Kf_mat.setOptionsPrefix(prefix)
        Kf_mat.setFromOptions()

        assemble_matrix(Kf_mat, Kf_form)
        Kf_mat.assemble()
        assemble_matrix(self.M_mat, M_form)
        self.M_mat.assemble()

        self.vec_rhs = self.u_in.vector.copy()

    def forward(self, u_in_func):
        self.M_mat.mult(u_in_func.vector, self.vec_rhs)
        self.solver.solve(self.vec_rhs, self.u_out_wrap)
        self.u_out.x.scatter_forward()
        return self.u_out

    def backward(self, df_dout_vec):
        """Adjoint pass: solves Kf * lambda = df_dout_vec, then returns M * lambda."""
        self.solver.solve(df_dout_vec, self.lambda_wrap)
        self.lambda_adj.x.scatter_forward()
        df_din_vec = self.lambda_adj.vector.copy()
        self.M_mat.mult(self.lambda_adj.vector, df_din_vec)
        return df_din_vec


class Heaviside:
    def __init__(self, rho_phys):
        self.rho_phys = rho_phys

    def forward(self, beta, eta=0.5):
        denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
        self.drho = (
            beta * (1 - np.tanh(beta * (self.rho_phys.vector - eta)) ** 2) / denominator
        )
        self.rho_phys.vector.array = (
            np.tanh(beta * eta) + np.tanh(beta * (self.rho_phys.vector - eta))
        ) / denominator
        self.rho_phys.x.scatter_forward()

    def backward(self, vectors):
        for vector in vectors:
            if vector is not None:
                vector.array *= self.drho

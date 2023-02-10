from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import tqdm.autonotebook
from dolfinx.mesh import create_box
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    Constant,
    dirichletbc,
    form,
    Function,
    FunctionSpace,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)
from ufl import (
    cross,
    curl,
    div,
    dot,
    ds,
    dx,
    FacetNormal,
    FiniteElement,
    grad,
    Identity,
    inner,
    lhs,
    nabla_grad,
    rhs,
    sym,
    TestFunction,
    TrialFunction,
    VectorElement,
)


def mhd_sim(
    Ha_no=10,
    conductive=True,
    results_foldername="Results/",
    total_time=5e-01,
    dt=5e-01,
    Nx=20,
    Ny=30,
    Nz=30,
    export_mode=1,
):
    """
    runs a 3D navier stokes simulation with a lorentz force term modelling
    mhd effects on the fluid

    Args:
        Ha_no (int, float): the Hartmann number
        conductive (bool): The conductivity of the Hartmann walls
        results_foldername (str): The location in which results files will
            be saved
        total_time(int, float): total time of simulation
        dt (int, float): time step for the solver
        Nx (int, float): number of cells in mesh x plane
        Ny (int, float): number of cells in mesh y plane
        Nz (int, float): number of cells in mesh z plane
        export_mode (int, float): mode in which to export results, if 1 results
            from each time step will be exported, if 2 only final timestep
            is exported.
    """
    # Define boundaries
    eps = 1e-14

    def walls(x):
        walls_1 = np.logical_or(np.abs(x[1] - 0.0) < eps, np.abs(x[1] - 2.0) < eps)
        walls_2 = np.logical_or(np.abs(x[2] - 0.0) < eps, np.abs(x[2] - 2.0) < eps)
        return np.logical_or(walls_1, walls_2)

    def hartmann_walls(x):
        return np.logical_or(np.abs(x[1] - 0.0) < eps, np.abs(x[1] - 2.0) < eps)

    def inlet(x):
        return np.abs(x[0] - 0.0) < eps

    def outlet(x):
        return np.abs(x[0] - 20.0) < eps

    def inlet_velocity(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 10

        return values

    # define tensors for chorins projection
    def epsilon(u):
        """Strain-rate tensor"""
        return sym(nabla_grad(u))

    def sigma(u, p):
        """Stress tensor"""
        return 2 * mu * epsilon(u) - p * Identity(u.geometric_dimension())

    # define mesh
    mesh = create_box(
        MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([20, 2, 2])], [Nx, Ny, Nz]
    )

    # define temporal parameters
    t = 0
    T = total_time
    dt = dt  # Time step size
    num_steps = int(T / dt)
    k = Constant(mesh, PETSc.ScalarType(dt))

    # Define elements and function spaces
    velocity_ele = VectorElement("CG", mesh.ufl_cell(), 2)
    electric_potential_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
    pressure_ele = FiniteElement("CG", mesh.ufl_cell(), 1)

    V = FunctionSpace(mesh, velocity_ele)
    Q = FunctionSpace(mesh, electric_potential_ele)
    Q2 = FunctionSpace(mesh, pressure_ele)

    # Define boundary conditions
    bc_fully_conductive = dirichletbc(
        PETSc.ScalarType(0), locate_dofs_geometrical(Q, hartmann_walls), Q
    )

    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc_noslip = dirichletbc(u_noslip, locate_dofs_geometrical(V, walls), V)
    u_inlet = Function(V)
    u_inlet.interpolate(inlet_velocity)
    bc_inflow = dirichletbc(u_inlet, locate_dofs_geometrical(V, inlet))

    bc_outflow = dirichletbc(
        PETSc.ScalarType(0), locate_dofs_geometrical(Q2, outlet), Q2
    )

    if conductive is True:
        bcphi = [bc_fully_conductive]
    else:
        bcphi = []

    bcu = [bc_noslip, bc_inflow]
    bcp = [bc_outflow]

    # define functions and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)
    u_.name = "u"
    u_n = Function(V)
    n = FacetNormal(mesh)
    U = 0.5 * (u_n + u)

    phi = TrialFunction(Q)
    q = TestFunction(Q)
    phi_ = Function(Q)
    phi_.name = "phi"

    p = TrialFunction(Q2)
    q2 = TestFunction(Q2)
    p_ = Function(Q2)
    p_.name = "p"
    p_n = Function(Q2)

    # Define constant parameters
    Ha = Constant(mesh, (PETSc.ScalarType(Ha_no)))
    N = Ha**2
    mu = Constant(mesh, PETSc.ScalarType(1))  # Dynamic viscosity
    rho = Constant(mesh, PETSc.ScalarType(1))  # Density
    B = Constant(mesh, (PETSc.ScalarType(0), PETSc.ScalarType(-1), PETSc.ScalarType(0)))

    ######################################################################
    # Define variational formulation and solver paramaters for each step #
    ######################################################################

    # Step 1: Evaluate the electrical potential
    F1 = (
        inner(grad(phi), grad(q)) * dx
        - inner(dot(curl(u_n), B) + dot(u_n, curl(B)), q) * dx
    )
    a1 = form(lhs(F1))
    L1 = form(rhs(F1))
    if conductive is True:
        A1 = assemble_matrix(a1, bcs=bcphi)
    else:
        A1 = assemble_matrix(a1)
    A1.assemble()
    b1 = create_vector(L1)

    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A1)
    solver1.setTolerances(rtol=1e-12)
    solver1.setTolerances(atol=1e-08)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.HYPRE)
    pc1.setHYPREType("boomeramg")
    solver1.setConvergenceHistory()

    # Step 2: Tentative velocity step
    F2 = rho * dot((u - u_n) / k, v) * dx
    F2 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
    F2 += inner(sigma(U, p_n), epsilon(v)) * dx
    F2 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
    F2 += N * (
        inner(cross(B, grad(phi_)), v) * dx
        + inner(u_n * dot(B, B), v) * dx
        - inner(B * dot(B, u_n), v) * dx
    )
    a2 = form(lhs(F2))
    L2 = form(rhs(F2))
    A2 = assemble_matrix(a2, bcs=bcu)
    A2.assemble()
    b2 = create_vector(L2)

    solver2 = PETSc.KSP().create(mesh.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.BCGS)
    solver2.setTolerances(rtol=1e-08)
    solver2.setTolerances(atol=1e-08)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")
    solver2.setConvergenceHistory()

    # Step 3: Pressure corrrection step
    a3 = form(dot(nabla_grad(p), nabla_grad(q2)) * dx)
    L3 = form(dot(nabla_grad(p_n), nabla_grad(q2)) * dx - (rho / k) * div(u_) * q2 * dx)
    A3 = assemble_matrix(a3, bcs=bcp)
    A3.assemble()
    b3 = create_vector(L3)

    solver3 = PETSc.KSP().create(mesh.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.MINRES)
    solver3.setTolerances(rtol=1e-08)
    solver3.setTolerances(atol=1e-08)
    pc3 = solver3.getPC()
    pc3.setType(PETSc.PC.Type.HYPRE)
    pc3.setHYPREType("boomeramg")
    solver3.setConvergenceHistory()

    # Step 4: Velocity correction step
    a4 = form(rho * dot(u, v) * dx)
    L4 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
    A4 = assemble_matrix(a4)
    A4.assemble()
    b4 = create_vector(L4)

    solver4 = PETSc.KSP().create(mesh.comm)
    solver4.setOperators(A4)
    solver4.setType(PETSc.KSP.Type.CG)
    solver4.setTolerances(rtol=1e-08)
    solver4.setTolerances(atol=1e-08)
    pc4 = solver4.getPC()
    pc4.setType(PETSc.PC.Type.SOR)
    solver4.setConvergenceHistory()

    # Define results files and location
    if export_mode not in [1, 2]:
        raise ValueError("unexpected export_mode value")

    if export_mode == 1:
        u_xdmf = XDMFFile(mesh.comm, results_foldername + "u.xdmf", "w")
        u_xdmf.write_mesh(mesh)
        p_xdmf = XDMFFile(mesh.comm, results_foldername + "p.xdmf", "w")
        p_xdmf.write_mesh(mesh)
        phi_xdmf = XDMFFile(mesh.comm, results_foldername + "phi.xdmf", "w")
        phi_xdmf.write_mesh(mesh)
    
    # Initialise velocity field
    # u_n.x.array[:] = 10

    # Print convergence details option
    # opts = PETSc.Options()
    # opts["ksp_monitor"] = None
    # solver1.setFromOptions()
    # solver2.setFromOptions()
    # solver3.setFromOptions()
    # solver4.setFromOptions()

    progress = tqdm.autonotebook.tqdm(desc="Solving Navier-Stokes", total=num_steps)
    for i in range(num_steps):
        progress.update(1)
        # Update current time step
        t += dt

        # Step 1: Evaluate the electrical potential
        with b1.localForm() as loc_1:
            loc_1.set(0)
        assemble_vector(b1, L1)
        if conductive is True:
            apply_lifting(b1, [a1], [bcphi])
            b1.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            set_bc(b1, bcphi)
        else:
            b1.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
        solver1.solve(b1, phi_.vector)
        phi_.x.scatter_forward()

        # Step 2: Tentative velocity step
        with b2.localForm() as loc_2:
            loc_2.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcu])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcu)
        solver2.solve(b2, u_.vector)
        u_.x.scatter_forward()

        # Step 3: Pressure corrrection step
        with b3.localForm() as loc_3:
            loc_3.set(0)
        assemble_vector(b3, L3)
        apply_lifting(b3, [a3], [bcp])
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b3, bcp)
        solver3.solve(b3, p_.vector)
        p_.x.scatter_forward()

        # Step 4: Velocity correction step
        with b4.localForm() as loc_4:
            loc_4.set(0)
        assemble_vector(b4, L4)
        b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver4.solve(b4, u_.vector)
        u_.x.scatter_forward()

        # Update variable with solution form this time step
        u_n.x.array[:] = u_.x.array[:]
        p_n.x.array[:] = p_.x.array[:]

        # Write solutions to file
        if export_mode == 1:
            u_xdmf.write_function(u_, t)
            p_xdmf.write_function(p_, t)
            phi_xdmf.write_function(phi_, t)

    if export_mode == 2:
        u_xdmf.write_function(u_)
        p_xdmf.write_function(p_)
        phi_xdmf.write_function(phi_)

    u_xdmf.close()
    p_xdmf.close()
    phi_xdmf.close()

if __name__ == "__main__":

    mhd_sim(
        Ha_no=30,
        conductive=True,
        results_foldername="Results/",
        total_time=1e-01,
        dt=1e-03,
        Nx=20,
        Ny=30,
        Nz=30,
        export_mode=1
    )

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import tqdm.autonotebook

from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    dirichletbc,
    form,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
    create_matrix,
)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (
    create_box,
)
from ufl import (
    FiniteElement,
    TestFunction,
    TrialFunction,
    VectorElement,
    div,
    dot,
    dx,
    inner,
    lhs,
    nabla_grad,
    rhs,
    grad,
    cross,
    curl,
    ds,
    sym,
    Identity,
    FacetNormal,
)


# Define boundaries
def walls(x):
    walls_1 = np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 2.0))
    walls_2 = np.logical_or(np.isclose(x[2], 0.0), np.isclose(x[2], 2.0))
    return np.logical_or(walls_1, walls_2)


def hartmann_walls(x):
    return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 2.0))


def inlet(x):
    return np.isclose(x[0], 0.0)


def outlet(x):
    return np.isclose(x[0], 20.0)


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
    MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([20, 2, 2])], [20, 30, 30]
)

# define temporal parameters
t = 0
T = 3
dt = 1 / 800  # Time step size
num_steps = int(T / dt)
k = Constant(mesh, PETSc.ScalarType(dt))

# Define elements and function spaces
velocity_ele = VectorElement("CG", mesh.ufl_cell(), 2)
electric_potential_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
electric_current_density_ele = VectorElement("CG", mesh.ufl_cell(), 1)
lorentz_force_ele = VectorElement("CG", mesh.ufl_cell(), 1)
pressure_ele = FiniteElement("CG", mesh.ufl_cell(), 1)

V = FunctionSpace(mesh, velocity_ele)
Q = FunctionSpace(mesh, electric_potential_ele)
V2 = FunctionSpace(mesh, electric_current_density_ele)
V3 = FunctionSpace(mesh, lorentz_force_ele)
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
bc_outflow = dirichletbc(PETSc.ScalarType(0), locate_dofs_geometrical(Q2, outlet), Q2)

bcphi = [bc_fully_conductive]
bcu = [bc_noslip, bc_inflow]
bcp = [bc_outflow]

# define functions
u = TrialFunction(V)
v = TestFunction(V)
u_ = Function(V)
u_.name = "u"
u_n = Function(V)
u_phi = Function(V)
n = FacetNormal(mesh)
U = 0.5 * (u_n + u)

phi = TrialFunction(Q)
q = TestFunction(Q)
phi_ = Function(Q)
phi_test = Function(Q)
phi_.name = "phi"
phi_n = Function(Q)
phi_n2 = Function(Q)

J = TrialFunction(V2)
v2 = TestFunction(V2)
J_ = Function(V2)

F_lorentz = TrialFunction(V3)
v3 = TestFunction(V3)
F_lorentz_ = Function(V3)

p = TrialFunction(Q2)
q2 = TestFunction(Q2)
p_ = Function(Q2)
p_.name = "p"
p_n = Function(Q2)

# Define constant parameters
B = Constant(mesh, (PETSc.ScalarType(0), PETSc.ScalarType(-1), PETSc.ScalarType(0)))
Ha = Constant(mesh, (PETSc.ScalarType(10)))
N = Ha**2
mu = Constant(mesh, PETSc.ScalarType(1))  # Dynamic viscosity
rho = Constant(mesh, PETSc.ScalarType(1))  # Density

######################################################################
# Define variational formulation and solver paramaters for each step #
######################################################################

# Step 1: Evaluate the electrical potential
F1 = (
    inner(grad(phi), grad(q)) * dx
    - inner(dot(B, curl(u_phi)) + dot(u_phi, curl(B)), q) * dx
)
a1 = form(lhs(F1))
L1 = form(rhs(F1))
A1 = create_matrix(a1)
b1 = create_vector(L1)

solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Step 2: Evaluate the electric current density
F2 = inner(J, v2) * dx - N * (
    inner(grad(phi_), v2) * dx + inner(cross(u_phi, B), v2) * dx
)
a2 = form(lhs(F2))
L2 = form(rhs(F2))
A2 = assemble_matrix(a2)
A2.assemble()
b2 = create_vector(L2)

solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.CG)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.SOR)

# Step 3: Evaluate the Lorentz force
F3 = inner(F_lorentz, v3) * dx - inner(cross(J_, B), v3) * dx
a3 = form(lhs(F3))
L3 = form(rhs(F3))
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)

solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

# Step 4: Tentative velocity step
F4 = rho * dot((u - u_n) / k, v) * dx
F4 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
F4 += inner(sigma(U, p_n), epsilon(v)) * dx
F4 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
F4 -= dot(F_lorentz_, v) * dx
a4 = form(lhs(F4))
L4 = form(rhs(F4))
A4 = assemble_matrix(a4, bcs=bcu)
A4.assemble()
b4 = create_vector(L4)

solver4 = PETSc.KSP().create(mesh.comm)
solver4.setOperators(A4)
solver4.setType(PETSc.KSP.Type.BCGS)
pc4 = solver4.getPC()
pc4.setType(PETSc.PC.Type.HYPRE)
pc4.setHYPREType("boomeramg")

# Step 5: Pressure corrrection step
a5 = form(dot(nabla_grad(p), nabla_grad(q2)) * dx)
L5 = form(dot(nabla_grad(p_n), nabla_grad(q2)) * dx - (rho / k) * div(u_) * q2 * dx)
A5 = assemble_matrix(a5, bcs=bcp)
A5.assemble()
b5 = create_vector(L5)

solver5 = PETSc.KSP().create(mesh.comm)
solver5.setOperators(A5)
solver5.setType(PETSc.KSP.Type.BCGS)
pc5 = solver5.getPC()
pc5.setType(PETSc.PC.Type.HYPRE)
pc5.setHYPREType("boomeramg")

# Step 6: Velocity correction step
a6 = form(rho * dot(u, v) * dx)
L6 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
A6 = assemble_matrix(a6)
A6.assemble()
b6 = create_vector(L6)

solver6 = PETSc.KSP().create(mesh.comm)
solver6.setOperators(A6)
solver6.setType(PETSc.KSP.Type.CG)
pc6 = solver6.getPC()
pc6.setType(PETSc.PC.Type.SOR)

# Define results files and location
results_folder = "Results/"
u_xdmf = XDMFFile(mesh.comm, results_folder + "u.xdmf", "w")
u_xdmf.write_mesh(mesh)
p_xdmf = XDMFFile(mesh.comm, results_folder + "p.xdmf", "w")
p_xdmf.write_mesh(mesh)
phi_xdmf = XDMFFile(mesh.comm, results_folder + "phi.xdmf", "w")
phi_xdmf.write_mesh(mesh)
J_xdmf = XDMFFile(mesh.comm, results_folder + "J.xdmf", "w")
J_xdmf.write_mesh(mesh)

progress = tqdm.autonotebook.tqdm(desc="Solving Navier-Stokes", total=num_steps)
for i in range(num_steps):
    progress.update(1)
    # Update current time step
    t += dt

    # Step 1: Evaluate the electrical potential
    assemble_matrix(A1, a1, bcs=bcphi)
    A1.assemble()
    with b1.localForm() as loc:
        loc.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcphi])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcphi)
    solver1.solve(b1, phi_n.vector)
    phi_n.x.scatter_forward()

    phi_.vector.axpy(1, phi_n.vector)
    phi_.x.scatter_forward()

    # Step 2: Evaluate the electric current density
    with b2.localForm() as loc:
        loc.set(0)
    assemble_vector(b2, L2)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver2.solve(b2, J_.vector)
    J_.x.scatter_forward()

    # Step 3: Evaluate the Lorentz force
    with b3.localForm() as loc:
        loc.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, F_lorentz_.vector)
    F_lorentz_.x.scatter_forward()

    # Step 4: Tentative velocity step
    with b4.localForm() as loc_1:
        loc_1.set(0)
    assemble_vector(b4, L4)
    apply_lifting(b4, [a4], [bcu])
    b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b4, bcu)
    solver4.solve(b4, u_.vector)
    u_.x.scatter_forward()

    # Step 5: Pressure corrrection step
    with b5.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b5, L5)
    apply_lifting(b5, [a5], [bcp])
    b5.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b5, bcp)
    solver5.solve(b5, p_.vector)
    p_.x.scatter_forward()

    # Step 6: Velocity correction step
    with b6.localForm() as loc_3:
        loc_3.set(0)
    assemble_vector(b6, L6)
    b6.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver6.solve(b6, u_.vector)
    u_.x.scatter_forward()

    # Update variable with solution form this time step
    u_n.x.array[:] = u_.x.array[:]
    u_phi.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]

    # Write solutions to file
    u_xdmf.write_function(u_, t)
    p_xdmf.write_function(p_, t)
    phi_xdmf.write_function(phi_, t)
    J_xdmf.write_function(J_, t)


u_xdmf.close()
p_xdmf.close()
phi_xdmf.close()
J_xdmf.close()

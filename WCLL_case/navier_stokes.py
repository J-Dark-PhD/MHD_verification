from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import tqdm.autonotebook
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    Constant,
    dirichletbc,
    form,
    Function,
    FunctionSpace,
    locate_dofs_topological,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import (
    NonlinearProblem,
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)
from dolfinx.nls.petsc import NewtonSolver
from ufl import (
    cross,
    curl,
    div,
    dot,
    ds,
    FacetNormal,
    FiniteElement,
    grad,
    TestFunction,
    SpatialCoordinate,
    exp,
    conditional,
    lt,
    Identity,
    inner,
    lhs,
    nabla_grad,
    rhs,
    sym,
    TrialFunction,
    VectorElement,
)
from MeshFromXDMF import MeshXDMF
import properties

id_lipb = 6
id_W = 7
id_structure = 8
id_baffle = 9
id_pipe_1 = 10
id_pipe_2 = 11
id_pipe_3 = 12
id_pipe_4 = 13
id_pipe_5 = 14
id_pipe_6 = 15
id_pipe_7 = 16
id_pipe_8 = 17
id_pipe_9 = 18
id_channel_pipe_top = 19
id_channel_pipe_bot = 20
id_eurofers = [
    id_structure,
    id_baffle,
    id_pipe_1,
    id_pipe_2,
    id_pipe_3,
    id_pipe_4,
    id_pipe_5,
    id_pipe_6,
    id_pipe_7,
    id_pipe_8,
    id_pipe_9,
    id_channel_pipe_top,
    id_channel_pipe_bot,
]
id_fw_coolant_interface = 21
id_pipe_1_coolant_interface = 22
id_pipe_2_coolant_interface = 23
id_pipe_3_coolant_interface = 24
id_pipe_4_coolant_interface = 25
id_pipe_5_coolant_interface = 26
id_pipe_6_coolant_interface = 27
id_pipe_7_coolant_interface = 28
id_pipe_8_coolant_interface = 29
id_pipe_9_coolant_interface = 30
id_channel_pipe_top_coolant_interface = 31
id_channel_pipe_bot_coolant_interface = 32
ids_pipe_coolant_interface = [
    id_pipe_1_coolant_interface,
    id_pipe_2_coolant_interface,
    id_pipe_3_coolant_interface,
    id_pipe_4_coolant_interface,
    id_pipe_5_coolant_interface,
    id_pipe_6_coolant_interface,
    id_pipe_7_coolant_interface,
    id_pipe_8_coolant_interface,
    id_pipe_9_coolant_interface,
    id_channel_pipe_top_coolant_interface,
    id_channel_pipe_bot_coolant_interface,
]

id_plasma_facing_surface = 33
id_inlet = 34
id_outlet = 35

my_mesh = MeshXDMF(
    cell_file="meshes/mesh_domains_3D.xdmf",
    facet_file="meshes/mesh_boundaries_3D.xdmf",
    subdomains=[],
)

my_mesh.define_markers()
my_mesh.define_measures()

ft = my_mesh.surface_markers
ct = my_mesh.volume_markers

temperature_elements = FiniteElement("CG", my_mesh.mesh.ufl_cell(), 1)
V = FunctionSpace(my_mesh.mesh, temperature_elements)
T = Function(V)
v_T = TestFunction(V)

inlet_dofs = locate_dofs_topological(
    V, my_mesh.mesh.topology.dim - 1, ft.indices[ft.values == id_inlet]
)
bc_inlet_temperature = dirichletbc(PETSc.ScalarType(598.15), inlet_dofs, V)

bcs = [bc_inlet_temperature]


class convenctive_flux_bc:
    def __init__(self, h_coeff, T_ext, surfaces) -> None:
        self.h_coeff = h_coeff
        self.T_ext = T_ext
        self.surfaces = surfaces


class flux_bc:
    def __init__(self, value, surfaces) -> None:
        self.value = value
        self.surfaces = surfaces


plasma_heat_flux = flux_bc(value=0.5e06, surfaces=id_plasma_facing_surface)
convective_flux_bz = convenctive_flux_bc(
    h_coeff=5.025e03, T_ext=584.65, surfaces=ids_pipe_coolant_interface
)
convective_flux_fw = convenctive_flux_bc(
    h_coeff=8.876e03 * 5, T_ext=584.65, surfaces=id_fw_coolant_interface
)

convective_flux_bcs = [convective_flux_bz, convective_flux_fw]

thermal_cond = Constant(my_mesh.mesh, PETSc.ScalarType(1))
thermal_cond_W = Constant(my_mesh.mesh, PETSc.ScalarType(2))
thermal_cond_eurofer = Constant(my_mesh.mesh, PETSc.ScalarType(3))
thermal_cond_lipb = Constant(my_mesh.mesh, PETSc.ScalarType(4))


class source:
    def __init__(self, value, volume) -> None:
        self.volume = volume
        if isinstance(value, (float, int)):
            self.value = Constant(my_mesh.mesh, PETSc.ScalarType(value))
        else:
            self.value = value


x = SpatialCoordinate(my_mesh.mesh)
sources = [
    source(value=23.2e06 * exp(-71.74 * x[0]), volume=id_W),
    source(
        value=conditional(
            lt(x[0], 0.15),
            9.6209e06 * exp(-12.02 * x[0]),
            4.7109e06 * exp(-7.773 * x[0]),
        ),
        volume=id_eurofers,
    ),
    source(
        value=conditional(
            lt(x[0], 0.15),
            6.3034e05 * x[0] ** (-0.789),
            3.4588e06 * exp(-3.993 * x[0]),
        ),
        volume=id_lipb,
    ),
]

F = dot(properties.thermal_cond_W(T) * grad(T), grad(v_T)) * my_mesh.dx(id_W)
F += dot(properties.thermal_cond_lipb(T) * grad(T), grad(v_T)) * my_mesh.dx(id_lipb)

for id in id_eurofers:
    F += dot(properties.thermal_cond_eurofer(T) * grad(T), grad(v_T)) * my_mesh.dx(id)

for source_term in sources:
    if type(source_term.volume) is list:
        volumes = source_term.volume
    else:
        volumes = [source_term.volume]
    for volume in volumes:
        F += -source_term.value * v_T * my_mesh.dx(volume)

for bc in convective_flux_bcs:
    if type(bc.surfaces) is list:
        surfaces = bc.surfaces
    else:
        surfaces = [bc.surfaces]
    for surf in surfaces:
        form = -bc.h_coeff * (T - bc.T_ext)
        F += -form * v_T * my_mesh.ds(surf)

F += -plasma_heat_flux.value * v_T * my_mesh.ds(id_plasma_facing_surface)

problem = NonlinearProblem(F, T, bcs=bcs)

solver = NewtonSolver(MPI.COMM_WORLD, problem)

print("Solving temperature field")

solver.solve(T)

T_xdmf = XDMFFile(my_mesh.mesh.comm, "T.xdmf", "w")
T_xdmf.write_mesh(my_mesh.mesh)
T_xdmf.write_function(T)
T_xdmf.close()

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
    values = np.zeros((mesh.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 10

    return values


# define tensors for chorins projection
def epsilon(u):
    """Strain-rate tensor"""
    return sym(nabla_grad(u))


def sigma(u, p):
    """Stress tensor"""
    return 2 * 1 * epsilon(u) - p * Identity(len(u))


# define mesh
mesh = MeshXDMF(
    cell_file="meshes/mesh_domains_3D.xdmf",
    facet_file="meshes/mesh_boundaries_3D.xdmf",
    subdomains=[],
)

mesh.define_markers()
mesh.define_measures()

# define temporal parameters

conductive = True
Ha_no = 10
results_foldername = "Results/"
total_time = 5e-01
dt = 5e-01
export_mode = 1

t = 0
dt = dt  # Time step size
num_steps = int(total_time / dt)
k = Constant(mesh.mesh, PETSc.ScalarType(dt))

# Define elements and function spaces
velocity_ele = VectorElement("CG", mesh.mesh.ufl_cell(), 2)
electric_potential_ele = FiniteElement("CG", mesh.mesh.ufl_cell(), 1)
pressure_ele = FiniteElement("CG", mesh.mesh.ufl_cell(), 1)

V = FunctionSpace(mesh.mesh, velocity_ele)
Q = FunctionSpace(mesh.mesh, electric_potential_ele)
Q2 = FunctionSpace(mesh.mesh, pressure_ele)

# Define boundary conditions
bc_fully_conductive = dirichletbc(
    PETSc.ScalarType(0), locate_dofs_geometrical(Q, hartmann_walls), Q
)

u_noslip = np.array((0,) * mesh.mesh.geometry.dim, dtype=PETSc.ScalarType)
bc_noslip = dirichletbc(u_noslip, locate_dofs_geometrical(V, walls), V)
u_inlet = Function(V)
u_inlet.interpolate(inlet_velocity)
bc_inflow = dirichletbc(u_inlet, locate_dofs_geometrical(V, inlet))

bc_outflow = dirichletbc(PETSc.ScalarType(0), locate_dofs_geometrical(Q2, outlet), Q2)

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
n = FacetNormal(mesh.mesh)
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
Ha = Constant(mesh.mesh, (PETSc.ScalarType(Ha_no)))
N = Ha**2
mu = properties.visc_lipb(T)  # Dynamic viscosity
rho = properties.rho_lipb(T)  # Density
B = Constant(
    mesh.mesh, (PETSc.ScalarType(0), PETSc.ScalarType(-1), PETSc.ScalarType(0))
)

######################################################################
# Define variational formulation and solver paramaters for each step #
######################################################################

conductive = True

# Step 1: Evaluate the electrical potential
F1 = (
    inner(grad(phi), grad(q)) * mesh.dx
    - inner(dot(curl(u_n), B) + dot(u_n, curl(B)), q) * mesh.dx
)
a1 = form(lhs(F1))
L1 = form(rhs(F1))
if conductive is True:
    A1 = assemble_matrix(a1, bcs=bcphi)
else:
    A1 = assemble_matrix(a1)
A1.assemble()
b1 = create_vector(L1)

solver1 = PETSc.KSP().create(mesh.mesh.comm)
solver1.setOperators(A1)
solver1.setTolerances(rtol=1e-12)
solver1.setTolerances(atol=1e-08)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")
solver1.setConvergenceHistory()

# Step 2: Tentative velocity step
F2 = properties.rho_lipb(T) * dot((u - u_n) / k, v) * mesh.dx(id_lipb)
F2 += properties.rho_lipb(T) * dot(dot(u_n, nabla_grad(u_n)), v) * mesh.dx(id_lipb)
F2 += inner(sigma(U, p_n), epsilon(v)) * mesh.dx(id_lipb)
F2 += dot(p_n * n, v) * ds(id_lipb) - dot(
    properties.visc_lipb(T) * nabla_grad(U) * n, v
) * ds(id_lipb)
F2 += N * (
    inner(cross(B, grad(phi_)), v) * mesh.dx
    + inner(u_n * dot(B, B), v) * mesh.dx
    - inner(B * dot(B, u_n), v) * mesh.dx
)
a2 = form(lhs(F2))
L2 = form(rhs(F2))
A2 = assemble_matrix(a2, bcs=bcu)
A2.assemble()
b2 = create_vector(L2)

solver2 = PETSc.KSP().create(mesh.mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.BCGS)
solver2.setTolerances(rtol=1e-08)
solver2.setTolerances(atol=1e-08)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")
solver2.setConvergenceHistory()

# Step 3: Pressure corrrection step
a3 = form(dot(nabla_grad(p), nabla_grad(q2)) * mesh.dx)
L3 = form(
    dot(nabla_grad(p_n), nabla_grad(q2)) * mesh.dx
    - (properties.rho_lipb(T) / k) * div(u_) * q2 * mesh.dx
)
A3 = assemble_matrix(a3, bcs=bcp)
A3.assemble()
b3 = create_vector(L3)

solver3 = PETSc.KSP().create(mesh.mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.MINRES)
solver3.setTolerances(rtol=1e-08)
solver3.setTolerances(atol=1e-08)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.HYPRE)
pc3.setHYPREType("boomeramg")
solver3.setConvergenceHistory()

# Step 4: Velocity correction step
a4 = form(properties.rho_lipb(T) * dot(u, v) * mesh.dx)
L4 = form(
    properties.rho_lipb(T) * dot(u_, v) * mesh.dx
    - k * dot(nabla_grad(p_ - p_n), v) * mesh.dx
)
A4 = assemble_matrix(a4)
A4.assemble()
b4 = create_vector(L4)

solver4 = PETSc.KSP().create(mesh.mesh.comm)
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

u_xdmf = XDMFFile(mesh.mesh.comm, results_foldername + "u.xdmf", "w")
u_xdmf.write_mesh(mesh.mesh)
p_xdmf = XDMFFile(mesh.mesh.comm, results_foldername + "p.xdmf", "w")
p_xdmf.write_mesh(mesh.mesh)
phi_xdmf = XDMFFile(mesh.mesh.comm, results_foldername + "phi.xdmf", "w")
phi_xdmf.write_mesh(mesh.mesh)

# Initialise velocity field
# u_n.x.array[:] = 10

# Print convergence details option
# opts = PETSc.Options()
# opts["ksp_monitor"] = None
# solver1.setFromOptions()
# solver2.setFromOptions()
# solver3.setFromOptions()
# solver4.setFromOptions()

progress = tqdm.autonotebook.tqdm(desc="Solving MHD", total=num_steps)
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
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcphi)
    else:
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
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
    u_xdmf.write_function(u_, t)
    p_xdmf.write_function(p_, t)
    phi_xdmf.write_function(phi_, t)

u_xdmf.close()
p_xdmf.close()
phi_xdmf.close()

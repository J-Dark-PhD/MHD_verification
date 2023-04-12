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
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
    NonlinearProblem,
)
from dolfinx.nls.petsc import NewtonSolver
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
T_n = TestFunction(V)

eps = 1e-14


def front_boundary(x):
    return np.abs(x[0] - 0) < eps


def rear_boundary(x):
    return np.abs(x[0] - 0.567) < eps


bc_front = dirichletbc(
    PETSc.ScalarType(1), locate_dofs_geometrical(V, front_boundary), V
)
bc_rear = dirichletbc(PETSc.ScalarType(0), locate_dofs_geometrical(V, rear_boundary), V)

bcs = [bc_front, bc_rear]

f = Constant(my_mesh.mesh, PETSc.ScalarType(0))

F = dot(grad(T), grad(T_n)) * my_mesh.dx
F += -f * T_n * my_mesh.dx

problem = NonlinearProblem(F, T, bcs=bcs)

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.solve(T)

T_xdmf = XDMFFile(my_mesh.mesh.comm, "T.xdmf", "w")
T_xdmf.write_mesh(my_mesh.mesh)
T_xdmf.write_function(T)
T_xdmf.close()

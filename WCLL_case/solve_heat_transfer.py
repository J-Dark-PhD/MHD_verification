from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    dirichletbc,
    Function,
    FunctionSpace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    NonlinearProblem,
)
from dolfinx.nls.petsc import NewtonSolver
from ufl import (
    dot,
    FiniteElement,
    grad,
    TestFunction,
    SpatialCoordinate,
    exp,
    conditional,
    lt,
)
from my_classes import MeshXDMF, FluxBC, ConvenctiveFluxBC, Source
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


def Heat_Transfer_Solver():
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

    plasma_heat_flux = FluxBC(value=0.5e06, surfaces=id_plasma_facing_surface)
    convective_flux_bz = ConvenctiveFluxBC(
        h_coeff=5.025e03, T_ext=584.65, surfaces=ids_pipe_coolant_interface
    )
    convective_flux_fw = ConvenctiveFluxBC(
        h_coeff=8.876e03 * 5, T_ext=584.65, surfaces=id_fw_coolant_interface
    )

    bcs = [bc_inlet_temperature]
    flux_bcs = [plasma_heat_flux, convective_flux_bz, convective_flux_fw]

    x = SpatialCoordinate(my_mesh.mesh)
    sources = [
        Source(value=23.2e06 * exp(-71.74 * x[0]), volumes=id_W),
        Source(
            value=conditional(
                lt(x[0], 0.15),
                9.6209e06 * exp(-12.02 * x[0]),
                4.7109e06 * exp(-7.773 * x[0]),
            ),
            volumes=id_eurofers,
        ),
        Source(
            value=conditional(
                lt(x[0], 0.15),
                6.3034e05 * x[0] ** (-0.789),
                3.4588e06 * exp(-3.993 * x[0]),
            ),
            volumes=id_lipb,
        ),
    ]

    F = dot(properties.thermal_cond_W(T) * grad(T), grad(v_T)) * my_mesh.dx(id_W)
    F += dot(properties.thermal_cond_lipb(T) * grad(T), grad(v_T)) * my_mesh.dx(id_lipb)

    for id in id_eurofers:
        F += dot(properties.thermal_cond_eurofer(T) * grad(T), grad(v_T)) * my_mesh.dx(
            id
        )

    for source_term in sources:
        for volume in source_term.volumes:
            F += -source_term.value * v_T * my_mesh.dx(volume)

    for bc in flux_bcs:
        bc.create_form(T)
        for surface in bc.surfaces:
            F += -bc.form * v_T * my_mesh.ds(surface)

    problem = NonlinearProblem(F, T, bcs=bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    print("Solving temperature field")

    solver.solve(T)

    T_xdmf = XDMFFile(my_mesh.mesh.comm, "Results/T.xdmf", "w")
    T_xdmf.write_mesh(my_mesh.mesh)
    T_xdmf.write_function(T)
    T_xdmf.close()

    return T


Heat_Transfer_Solver()

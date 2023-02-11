from mhd import mhd_sim

Ha_test_values = [0, 10, 30, 60]
mesh_test_values = [20, 30, 40, 50]

for Ha_value in Ha_test_values:
    for mesh_value in mesh_test_values:

        N_cells = mesh_value

        if Ha_value == 0:
            total_time = 5e-01
            dt = 5e-03
        elif Ha_value == 10:
            total_time = 5e-01
            dt = 5e-03
        elif Ha_value == 30:
            total_time = 1e-01
            dt = 1e-03
        elif Ha_value == 60:
            total_time = 4e-02
            dt = 2e-04

        print("Running case Hartmann_no = {}, conductive".format(Ha_value))
        print("Mesh size = {} x {}".format(mesh_value, mesh_value))
        conductive_results_folder = (
            "Results/mesh_testing/conductive/Ha={}/mesh={}x{}/".format(
                Ha_value, mesh_value, mesh_value
            )
        )
        mhd_sim(
            Ha_no=Ha_value,
            conductive=True,
            results_foldername=conductive_results_folder,
            total_time=total_time,
            dt=dt,
            Nx=20,
            Ny=N_cells,
            Nz=N_cells,
            export_mode=2,
        )

        print("Running case Hartmann_no = {}, insulated".format(Ha_value))
        print("Mesh size = {} x {}".format(mesh_value, mesh_value))
        insulated_results_folder = (
            "Results/mesh_testing/insulated/Ha={}/mesh={}x{}/".format(
                Ha_value, mesh_value, mesh_value
            )
        )
        mhd_sim(
            Ha_no=Ha_value,
            conductive=False,
            results_foldername=insulated_results_folder,
            total_time=total_time,
            dt=dt,
            Nx=20,
            Ny=N_cells,
            Nz=N_cells,
            export_mode=2,
        )

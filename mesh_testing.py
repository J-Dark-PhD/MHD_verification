from mhd import mhd_sim

mesh_test_values = [20, 30, 40, 50]

Ha_test_values = [0, 10, 30, 60, 100]
conductive_total_times = [5e-01, 5e-01, 1e-01, 2e-02, 1e-02]
conductive_dts = [5e-03, 5e-03, 1e-03, 2e-04, 1e-04]

for Ha_value, total_time, dt in zip(
    Ha_test_values, conductive_total_times, conductive_dts
):
    for mesh_value in mesh_test_values:
    
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
            Ny=mesh_value,
            Nz=mesh_value,
            export_mode=2,
        )

insulated_total_times = [5e-01, 5e-01, 1e-01, 4e-02, 3e-02]
insulated_dts = [5e-03, 5e-03, 1e-03, 2e-04, 1e-04]

for Ha_value, total_time, dt in zip(
    Ha_test_values, insulated_total_times, insulated_dts
):
    for mesh_value in mesh_test_values:

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
            Ny=mesh_value,
            Nz=mesh_value,
            export_mode=2,
        )


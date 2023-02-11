from mhd import mhd_sim

Ha_test_values = [0, 10, 30, 60, 100]
conductive_total_times = [5e-01, 5e-01, 1e-01, 2e-02, 1e-02]
conductive_dts = [5e-03, 5e-03, 1e-03, 2e-04, 1e-04]
conductive_mesh_densities = [30, 30, 40, 100, 200]

for Ha_value, total_time, dt, mesh_density in zip(
    Ha_test_values, conductive_total_times, conductive_dts, conductive_mesh_densities
):

    print("Running case Hartmann_no = {}, conductive".format(Ha_value))
    conductive_results_folder = "Results/fully_conductive/Ha={}/".format(Ha_value)
    mhd_sim(
        Ha_no=Ha_value,
        conductive=True,
        results_foldername=conductive_results_folder,
        total_time=total_time,
        dt=dt,
        Nx=20,
        Ny=mesh_density,
        Nz=mesh_density,
    )

insulated_total_times = [5e-01, 5e-01, 1e-01, 4e-02, 3e-02]
insulated_dts = [5e-03, 5e-03, 1e-03, 2e-04, 1e-04]
insulated_mesh_densities = [30, 40, 100, 100, 100]

for Ha_value, total_time, dt, mesh_density in zip(
    Ha_test_values, insulated_total_times, insulated_dts, insulated_mesh_densities
):

    print("Running case Hartmann_no = {}, insulated".format(Ha_value))
    insulated_results_folder = "Results/fully_insulated/Ha={}/".format(Ha_value)
    mhd_sim(
        Ha_no=Ha_value,
        conductive=False,
        results_foldername=insulated_results_folder,
        total_time=total_time,
        dt=dt,
        Nx=20,
        Ny=mesh_density,
        Nz=mesh_density,
    )

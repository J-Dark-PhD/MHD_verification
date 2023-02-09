from mhd import mhd_sim

Ha_test_values = [0, 10, 30, 60, 100]

for Ha_value in Ha_test_values:

    mesh_density = 100

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
        total_time = 2e-02
        dt = 2e-04
    elif Ha_value == 100:
        total_time = 1e-02
        dt = 1e-04
        mesh_density = 200

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

for Ha_value in Ha_test_values:

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
    elif Ha_value == 100:
        total_time = 3e-02
        dt = 1e-04

    print("Running case Hartmann_no = {}, insulated".format(Ha_value))
    insulated_results_folder = "Results/fully_insulated/Ha={}/".format(Ha_value)
    mhd_sim(
        Ha_no=Ha_value,
        conductive=False,
        results_foldername=insulated_results_folder,
        total_time=total_time,
        dt=dt,
        Nx=20,
        Ny=100,
        Nz=100,
    )

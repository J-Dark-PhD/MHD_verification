from mhd import mhd_sim

test_values = [0, 10, 30, 60, 100]

for Ha_value in test_values:

    if Ha_value == 0:
        total_time = 0.5
        dt = 1 / 200
    elif Ha_value == 10:
        total_time = 0.1
        dt = 1 / 1000
    elif Ha_value == 30:
        total_time = 0.05
        dt = 1 / 2000
    elif Ha_value == 60:
        total_time = 0.02
        dt = 1 / 5000
    elif Ha_value == 100:
        total_time = 0.01
        dt = 1 / 10000

    print("Running case Hartmann_no = {}, conductive".format(Ha_value))
    conductive_results_folder = "Results/fully_conductive/Ha={}/".format(Ha_value)
    mhd_sim(
        Ha_no=Ha_value,
        conductive=True,
        results_foldername=conductive_results_folder,
        total_time=total_time,
        dt=dt,
    )

    print("Running case Hartmann_no = {}, insulated".format(Ha_value))
    insulated_results_folder = "Results/fully_insulated/Ha={}/".format(Ha_value)
    mhd_sim(
        Ha_no=Ha_value,
        conductive=False,
        results_foldername=insulated_results_folder,
        total_time=total_time,
        dt=dt,
    )

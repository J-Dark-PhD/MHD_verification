import matplotlib.pyplot as plt
import numpy as np

Ha_values = [0, 10, 30, 60]
mesh_values = [20, 30, 40, 50]
conductive_results_folder = "../mesh_testing/conductive/"
insulated_results_folder = "../mesh_testing/insulated/"
data = np.genfromtxt(
    conductive_results_folder + "Ha=60/mesh=40x40/profile.csv",
    delimiter=",",
    names=True,
)
x_values = data["arc_length"]


def conductive_case():
    conductive_velocities = []

    for Ha in Ha_values:
        conductive_velocities_per_ha = []
        for mesh_value in mesh_values:
            conductive_data = np.genfromtxt(
                conductive_results_folder
                + "Ha={}/mesh={}x{}/profile.csv".format(Ha, mesh_value, mesh_value),
                delimiter=",",
                names=True,
            )
            vel = conductive_data["u0"]
            conductive_velocities_per_ha.append(vel)
        conductive_velocities.append(conductive_velocities_per_ha)

    normalsed_conductive_velocities = []
    for ha_case in conductive_velocities:
        velocities_per_ha = []
        ref_values = ha_case[-1]
        for velocity_set in ha_case:
            values = velocity_set / ref_values
            velocities_per_ha.append(values - 1)
        normalsed_conductive_velocities.append(velocities_per_ha)

    # colours = ["cyan", "black", "red", "lawngreen", "blue"]
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=12)

    for Ha_value, velocities in zip(Ha_values, normalsed_conductive_velocities):
        plt.figure()
        for vel, mesh_size in zip(velocities, mesh_values):
            plt.plot(x_values, vel, label="Mesh = {}x{}".format(mesh_size, mesh_size))

        plt.title("Conductive case: Ha = {}".format(Ha_value))
        plt.xlabel("Z/L")
        plt.ylabel("$u/u_{0}$")
        plt.ylim(-2, 2)
        plt.xlim(0, 2)
        plt.legend()
        alpha = 0.25
        plt.axhspan(
            2,
            -2,
            color="tab:blue",
            alpha=alpha,
            label="2.0\%",
        )
        plt.axhspan(
            1,
            -1,
            color="tab:blue",
            alpha=alpha,
            label="1.0\%",
        )
        plt.axhspan(
            0.5,
            -0.5,
            color="tab:blue",
            alpha=alpha,
            label="0.5\%",
        )
        plt.axhspan(
            0.1,
            -0.1,
            color="tab:blue",
            alpha=alpha,
            label="0.1\%",
        )
        plt.tight_layout()


def insulated_case():
    insulated_velocities = []

    for Ha in Ha_values:
        insulated_velocities_per_ha = []
        for mesh_value in mesh_values:
            insulated_data = np.genfromtxt(
                insulated_results_folder
                + "Ha={}/mesh={}x{}/profile.csv".format(Ha, mesh_value, mesh_value),
                delimiter=",",
                names=True,
            )
            vel = insulated_data["u0"]
            insulated_velocities_per_ha.append(vel)
        insulated_velocities.append(insulated_velocities_per_ha)

    normalsed_insulated_velocities = []
    for ha_case in insulated_velocities:
        velocities_per_ha = []
        ref_values = ha_case[-1]
        for velocity_set in ha_case:
            values = velocity_set / ref_values
            velocities_per_ha.append(values - 1)
        normalsed_insulated_velocities.append(velocities_per_ha)

    # colours = ["cyan", "black", "red", "lawngreen", "blue"]
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=12)

    for Ha_value, velocities in zip(Ha_values, normalsed_insulated_velocities):
        plt.figure()
        for vel, mesh_size in zip(velocities, mesh_values):
            plt.plot(x_values, vel, label="Mesh = {}x{}".format(mesh_size, mesh_size))

        plt.title("Insulated case: Ha = {}".format(Ha_value))
        plt.xlabel("Z/L")
        plt.ylabel("$u/u_{0}$")
        plt.ylim(-2, 2)
        plt.xlim(0, 2)
        plt.legend()
        alpha = 0.25
        plt.axhspan(
            2,
            -2,
            color="tab:blue",
            alpha=alpha,
            label="2.0\%",
        )
        plt.axhspan(
            1,
            -1,
            color="tab:blue",
            alpha=alpha,
            label="1.0\%",
        )
        plt.axhspan(
            0.5,
            -0.5,
            color="tab:blue",
            alpha=alpha,
            label="0.5\%",
        )
        plt.axhspan(
            0.1,
            -0.1,
            color="tab:blue",
            alpha=alpha,
            label="0.1\%",
        )
        plt.tight_layout()


conductive_case()
insulated_case()

plt.show()

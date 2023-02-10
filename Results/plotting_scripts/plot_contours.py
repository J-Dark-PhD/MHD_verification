import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def contour_plot_ha_100():
    results_folder = "contour_data/ha=100/"
    y_values = np.linspace(0, 2, num=11)

    data = np.genfromtxt(results_folder + "0.2.csv", delimiter=",", names=True)
    x_values = list(data["arc_length"])

    X = np.array(x_values)
    Y = y_values
    Z = []
    for case in y_values:
        data = np.genfromtxt(results_folder + "{:.1f}.csv".format(case), delimiter=",", names=True)
        Z.append(list(data["u0"]))

    XX, YY = np.meshgrid(X, Y)
    ZZ = np.array(Z)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    XX, YY = np.meshgrid(X, Y)
    ZZ = np.array(Z)

    # Plot the surface.
    surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.jet,
                        linewidth=0, antialiased=False, vmin=0, vmax=60)

    # Customize the z axis.
    ax.set_zticks([])
    ax.set_xlabel("z")
    ax.set_xlim(0, 2)
    ax.set_zlim(0, 60)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 1.5, 2.0])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 1.5, 2.0])
    ax.set_ylim(0, 2)
    ax.set_ylabel("y")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.75, aspect=8, label="Velocity")


def contour_plot_ha_0():
    results_folder = "contour_data/ha=0/"
    y_values = np.linspace(0, 2, num=11)

    data = np.genfromtxt(results_folder + "0.2.csv", delimiter=",", names=True)
    x_values = list(data["arc_length"])

    X = np.array(x_values)
    Y = y_values
    Z = []
    for case in y_values:
        data = np.genfromtxt(results_folder + "{:.1f}.csv".format(case), delimiter=",", names=True)
        Z.append(list(data["u0"]))

    XX, YY = np.meshgrid(X, Y)
    ZZ = np.array(Z)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    XX, YY = np.meshgrid(X, Y)
    ZZ = np.array(Z)

    # Plot the surface.
    surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.jet,
                        linewidth=0, antialiased=False, vmin=0, vmax=60)

    # Customize the z axis.
    ax.set_zticks([])
    ax.set_xlabel("z")
    ax.set_xlim(0, 2)
    ax.set_zlim(0, 60)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 1.5, 2.0])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 1.5, 2.0])
    ax.set_ylim(0, 2)
    ax.set_ylabel("y")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.75, aspect=8, label="Velocity")

contour_plot_ha_100()
contour_plot_ha_0()

plt.show()
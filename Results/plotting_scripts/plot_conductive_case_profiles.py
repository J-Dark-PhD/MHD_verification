import matplotlib.pyplot as plt
import numpy as np

results_folder = "../fully_conductive/profiles/"
Ha_values = [0, 10, 30, 60, 100]
x_values = []
velocities = []

for Ha in Ha_values:
    data = np.genfromtxt(results_folder + "case_ha={}.csv".format(Ha), delimiter=",", names=True)
    x = data["arc_length"]
    vel = data["u0"]

    x_values.append(x)
    velocities.append(vel)

normalised_velocities = []
for case in velocities:
    normalised_vels = case/case[500]
    normalised_velocities.append(normalised_vels)

graph_x_values = []
for case in x_values:
    new_values = case - 1
    graph_x_values.append(new_values)

colours = ["cyan", "black", "red", "lawngreen", "blue"]
plt.figure()
for x, vel, Ha, colour in zip(graph_x_values, normalised_velocities, Ha_values, colours):
    plt.plot(x, vel, label="Ha = {}".format(Ha), color=colour)


plt.xlabel("Z/L")
plt.xlim(-1, 1)
plt.ylim(0, 25)
plt.ylabel("$u/u_{0}$")
plt.legend()
plt.tight_layout()

plt.show()
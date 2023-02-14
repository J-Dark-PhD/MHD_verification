import matplotlib.pyplot as plt
import numpy as np

results_folder = "../fully_insulated/profiles/"
Ha_values = [100, 60, 30, 10, 0]
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

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

colours = ["blue", "lawngreen", "red", "black", "cyan"]
plt.figure()
for x, vel, Ha, colour in zip(graph_x_values, normalised_velocities, Ha_values, colours):
    plt.plot(x, vel, label="Ha = {}".format(Ha), color=colour)

plt.legend()
plt.xlabel("y/L")
plt.xlim(-1, 1)
plt.ylim(bottom=0)
plt.ylabel("$u/u_{0}$")
plt.tight_layout()

plt.show()
import matplotlib.pyplot as plt
import numpy as np

k_B = 8.6e-5

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=12)

# ##### Tungsten ######
#
# Taken from (P.Tolias, 2017)


def Cp_W(T):  # units in J/(kg*K)
    return (
        21.868372
        + 8.068661e-03 * T
        - 1e-06 * T**2
        + 1.075862e-09 * T**3
        + 1.406637e04 / T**2
    )


def rho_W(T):  # units in kg/m**3
    """
    (adjusted by factor 1000 as orginial equation in g/cm**3)
    """
    T_W_0 = 293.15
    return (
        19250
        - 2.66207e-01 * (T - T_W_0)
        - 3.0595e-06 * (T - T_W_0) ** 2
        - 9.5185e-09 * (T - T_W_0) ** 3
    )


def thermal_cond_W(T):  # units in W/(m*K)
    return (
        149.441
        - 45.466e-03 * T
        + 13.193e-06 * T**2
        - 1.484e-09 * T**3
        + 3.866e06 / (T + 1) ** 2
    )
    # (T+1) is there to avoid dividing by 0


# taken from (Frauenfelder, R. 1969)
D_0_W = 4.1e-7  # Diffusivity pre-exponential factor (m^(2).s^(-1))
E_D_W = 0.39  # Diffusivity activation energy (eV)
S_0_W = 1.87e24  # Solubility pre-exponential factor (m^(-3).Pa^(-0.5))
E_S_W = 1.04  # Solutbiility activation energy (eV)

# alt value
D_0_W_alt = (1.9e-07) / (3**0.5)
E_D_W_alt = 0.2

D_0_W_alt_2 = (2.06e-07) / (3**0.5)
E_D_W_alt_2 = 0.28

atom_density_W = 6.3222e28


def D_W(T):
    return D_0_W * np.exp(-E_D_W / k_B / T)


def D_W_alt(T):
    return D_0_W_alt * np.exp(-E_D_W_alt / k_B / T)


def D_W_alt_2(T):
    return D_0_W_alt_2 * np.exp(-E_D_W_alt_2 / k_B / T)


def S_W(T):
    return S_0_W * np.exp(-E_S_W / k_B / T)


# ##### EUROfer ######
#
#   Values taken from Materials properties handbook


def Cp_eurofer(T):  # units in J/(kg*K)
    return -139.66 + 3.4777 * T - 0.0063847 * T**2 + 4.0984e-06 * T**3


# def Cp_eurofer_2(T):  # units in J/(kg*K)(Mergia)
#     return 2.696*T - 0.00496*T**2 + 3.335e-06*T**3


def rho_eurofer(T):  # units in kg/m**3
    return 7852.102143 - 0.331026405 * T


def thermal_cond_eurofer(T):  # units in W/(m*K)
    return 5.4308 + 0.13565 * T - 2.3862e-04 * T**2 + 1.3393e-07 * T**3


# taken from (Chen, 2021)
D_0_eurofer = 3.15e-08  # Diffusivity pre-exponential factor (m2/s)
E_D_eurofer = 0.0622  # Diffusivity activation energy (eV)
S_0_eurofer = 2.4088e23  # Solubility pre-exponential factor (atom/m3 Pa^-0.5)
E_S_eurofer = 0.3026  # Solutbiility activation energy (eV)

atom_density_eurofer = 8.409e28  # (m-3)
trap_density_eurofer = 4.5e23  # (m-3)
trap_energy_eurofer = 0.7804  # (eV)

# recombination coefficient from Liu Journal of Nuclear Materials (2021)
Kr_0_eurofer = 1.4143446334700682e-26
E_Kr_eurofer = -0.25727457261201786

# recombination coefficient from Braun (1980)
# Kr_0_eurofer = 5.9680e-17
# E_Kr_eurofer = 0.888

# recombination coefficient from Esteban (2000)
# Kr_0_eurofer = 4.7127e-31
# E_Kr_eurofer = 2.471

# recombination coefficient instantaneus

# Kr_0_eurofer = 0
# E_Kr_eurofer = 1


def D_eurofer(T):
    return D_0_eurofer * np.exp(-E_D_eurofer / k_B / T)


def S_eurofer(T):
    return S_0_eurofer * np.exp(-E_S_eurofer / k_B / T)


# ##### LiPi ######
#
#  Values taken from (D.Martelli et al, 2019)


def Cp_lipb(T):  # units in J/(kg*K)
    """
    adjusted by factor 1000 as orginial equation in J/(g*K)
    for values of temperature 508K < T < 800K,
    """
    return 195 - 9.116e-03 * T


def rho_lipb(T):  # units in kg/(m**3)
    return 10520.35 - 1.19051 * T


def thermal_cond_lipb(T):  # units in W/(m*K)
    """
    adjusted by factor 100 as original equation in W/(cm*K))
    """
    return 9.14779235 + 0.019631 * T


def visc_lipb(T):  # units (Pa s)
    return (
        0.01555147189
        - 4.827051855e-05 * T
        + 5.641475215e-08 * T**2
        - 2.2887e-11 * T**3
    )


def beta_lipb(T):  # units in K-1
    return 1.1221e-04 + 1.531e-08 * T


rho_0_lipb = 9808.2464435  # value T = 300K, units in kg/(m**3)

# taken from (Reiter, 1990)
D_0_lipb = 4.03e-08  # Diffusivity coefficient pre-exponential factor
E_D_lipb = 0.2021  # Diffusivity coefficient activation energy (eV)

# taken from (Aiello, 2008)
S_0_lipb = 1.427214e23  # Solubility coefficient pre-exponential factor
E_S_lipb = 0.133  # Solutbiility coefficient activation energy (eV)


def D_lipb(T):
    return D_0_lipb * np.exp(-E_D_lipb / k_B / T)


def S_lipb(T):
    return S_0_lipb * np.exp(-E_S_lipb / k_B / T)


# ##### Al2O3 - Permeation barrier ##### #

# taken from ()
D_0_al2o3 = 7.35e-07  # Diffusivity coefficient pre-exponential factor
E_D_al2o3 = 1.899  # Diffusivity coefficient activation energy (eV)
S_0_al2o3 = 2.519e25  # Solubility coefficient pre-exponential factor
E_S_al2o3 = 0.902  # Solutbiility coefficient activation energy (eV)


def D_al2o3(T):
    return D_0_al2o3 * np.exp(-E_D_al2o3 / k_B / T)


def S_al2o3(T):
    return S_0_al2o3 * np.exp(-E_S_al2o3 / k_B / T)


# Al taken from (Hydrogen Permeation Measurements on Alumina - E.Serra 2005)
D_0_al = 9.7e-08  # Diffusivity coefficient pre-exponential factor
E_D_al = 0.829  # Diffusivity coefficient activation energy (eV)
S_0_al = 9.133e19  # Solubility coefficient pre-exponential factor
E_S_al = 0.234  # Solutbiility coefficient activation energy (eV)


# ##### Plotting Data ##### #

if __name__ == "__main__":

    def tick_function(X):
        V = 1000 / (X)
        return ["%.0f" % z for z in V]

    T = np.arange(500, 900, step=1)
    red_W = (171 / 255, 15 / 255, 26 / 255)
    grey_eurofer = (130 / 255, 130 / 255, 130 / 255)

    darkening_factor_green = 0.75
    green_lipb = (
        darkening_factor_green * 146 / 255,
        darkening_factor_green * 196 / 255,
        darkening_factor_green * 125 / 255,
    )

    # ##### Thermophyical properties ######################################## #

    # ##### Density ##### #

    plt.figure()
    plt.plot(T, rho_W(T), label="W", color=red_W)
    plt.plot(T, rho_eurofer(T), label="Eurofer", color=grey_eurofer)
    plt.plot(T, rho_lipb(T), label="LiPb", color=green_lipb)
    plt.xlabel(r"T (K)")
    plt.ylabel(r"$\rho$ (Kg m$^{-3}$)")

    # ##### Specific heat capacity ##### #

    plt.figure()
    plt.plot(T, Cp_W(T), label="W", color=red_W)
    plt.plot(T, Cp_eurofer(T), label="Eurofer", color=grey_eurofer)
    plt.plot(T, Cp_lipb(T), label="LiPb", color=green_lipb)
    plt.xlabel(r"T (K)")
    plt.ylabel(r"$c_p$ (J kg$^{-1}$ K$^{-1}$)")

    # #### Thermal conductivity ##### #

    plt.figure()
    plt.plot(T, thermal_cond_W(T), label="W", color=red_W)
    plt.plot(T, thermal_cond_eurofer(T), label="Eurofer", color=grey_eurofer)
    plt.plot(T, thermal_cond_lipb(T), label="LiPb", color=green_lipb)
    plt.xlabel(r"T (K)")
    plt.ylabel(r"$\lambda$ (W m$^{-1}$ K$^{-1}$)")

    # #### Viscosity ##### #

    plt.figure()
    plt.plot(T, visc_lipb(T), label="LiPb", color=green_lipb)
    plt.xlabel(r"T (K)")
    plt.ylabel(r"Viscosity (Pa s)")

    # #### Thermal expansion coefficient ##### #

    plt.figure()
    plt.plot(T, beta_lipb(T), label="LiPb", color=green_lipb)
    plt.xlabel(r"T (K)")
    plt.ylabel(r"Thermal expansion coefficient")
    plt.ylim(bottom=0, top=0.0002)
    # #### Hydrogen transport properties ################################### #

    # #### Diffusivity ##### #

    # testing alternate diffsivity
    plt.figure()
    plt.plot(1000 / T, D_W(T), label="standard", color="black")
    plt.plot(1000 / T, D_W_alt(T), label="DFT", color="red")
    plt.plot(1000 / T, D_W_alt_2(T), label="Holzner", color="blue")
    plt.annotate("standard", (1.6, 3.5e-10), color="black")
    plt.annotate("DFT", (1.6, 3.5e-9), color="red")
    plt.annotate("Holzner", (1.6, 8e-10), color="blue")
    plt.xlabel(r"1000/T (K)")
    plt.ylabel(r"Diffusivity (m$^{2}$ s$^{-1}$)")
    plt.yscale("log")
    plt.minorticks_on()
    plt.grid(which="minor", alpha=0.3)
    plt.grid(which="major", alpha=0.7)

    # plt.figure(figsize=(6.4, 4.8/2))
    fig, axs = plt.subplots(2, 1, sharex=True)

    plt.sca(axs[0])
    plt.plot(1000 / T, D_eurofer(T), label="Eurofer", color=grey_eurofer)
    plt.plot(1000 / T, D_lipb(T), label="PbLi", color=green_lipb)
    plt.plot(1000 / T, D_W(T), label="W", color=red_W)
    plt.annotate("EUROFER", (0.02 + 1000 / T[0], D_eurofer(T[0])), color=grey_eurofer)
    plt.annotate("LiPb", (0.02 + 1000 / T[0], D_lipb(T[0])), color=green_lipb)
    plt.annotate("W", (0.02 + 1000 / T[0], D_W(T[0])), color=red_W)
    # plt.xlabel(r"1000/T (K)")
    plt.ylabel(r"Diffusivity (m$^{2}$ s$^{-1}$)")
    plt.yscale("log")
    plt.minorticks_on()
    plt.grid(which="minor", alpha=0.3)
    plt.grid(which="major", alpha=0.7)

    # #### Solubility ##### #
    plt.sca(axs[1])
    # plt.figure(figsize=(6.4, 4.8/2))
    # plt.plot(1000/T, S_W(T), label="W", color=red_W)
    plt.plot(1000 / T, S_al2o3(T), label="Barrier", color="black")
    plt.plot(1000 / T, S_eurofer(T), label="Eurofer", color=grey_eurofer)
    plt.plot(1000 / T, S_lipb(T), label="LiPb", color=green_lipb)
    plt.xlabel(r"1000/T (K)")
    plt.ylabel(r"Solubility (m$^{-3}$Pa$^{-0.5}$)")
    plt.yscale("log")
    plt.annotate("EUROFER", (0.02 + 1000 / T[0], S_eurofer(T[0])), color=grey_eurofer)
    plt.annotate("LiPb", (0.02 + 1000 / T[0], S_lipb(T[0])), color=green_lipb)
    plt.annotate("W", (0.02 + 1000 / T[0], S_W(T[0])), color=red_W)
    plt.xlim(right=2.2)

    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        if plt.gca().get_yscale() == "linear":
            plt.ylim(bottom=0)
        # plt.yscale("log")
        # plt.legend()
        plt.minorticks_on()
        plt.grid(which="minor", alpha=0.3)
        plt.grid(which="major", alpha=0.5)
        plt.tight_layout()

    plt.show()

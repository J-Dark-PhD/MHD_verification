# MHD_verification

A Computational Fluid Dynamics (CFD) solver is extended to model MHD effects designed in fenicsx for coupling with festim in order to have an advection field projected onto a hydrogen transport simulation domain.

One case so far

## Install Docker container
This model uses fenicsx, a newer version of the legacy FEniCS project.
The FEniCS project provides a [Docker image](https://hub.docker.com/r/fenicsproject/stable/) with FEniCS and its dependencies (python3, UFL, DOLFIN, numpy, sympy...)  already installed. See their ["FEniCS in Docker" manual](https://fenics.readthedocs.io/projects/containers/en/latest/).

Get Docker [here](https://www.docker.com/community-edition).

Pull the Docker image and run the container, sharing a folder between the host and container:

For Windows users:
```python
docker run -ti -v ${PWD}:/home/shared --name mhd_verification dolfinx/dolfinx:stable
```

## Laminar duct flow with conjugate MHD


 - A. Blishchik, M. van der Lans, S. Kenjeres. _An extensive numerical benchmark of the various magnetohydrodynamic flows_. International Journal of Heat and Fluid Flow, 90, 2021, DOI: [10.1016/j.ijheatfluidflow.2021.108800](https://doi.org/10.1016/j.ijheatfluidflow.2021.108800)


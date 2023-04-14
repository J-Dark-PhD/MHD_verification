from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    Constant,
)
from ufl import (
    Measure,
)


class Mesh:
    """
    Mesh class
    Attributes:
        mesh (dolfinx.Mesh): the mesh
        volume_markers (dolfinx.MeshTags): markers of the mesh cells
        surface_markers (dolfinx.MeshTags): markers of the mesh facets
        dx (dolfinx.Measure):
        ds (dolfinx.Measure):
    """

    def __init__(
        self, mesh=None, volume_markers=None, surface_markers=None, subdomains=[]
    ) -> None:
        """Inits Mesh
        Args:
            mesh (dolfinx.Mesh, optional): the mesh. Defaults to None.
            volume_markers (dolfinx.MeshTags, optional): markers of the mesh cells. Defaults to None.
            surface_markers (dolfinx.MeshTags, optional): markers of the mesh facets. Defaults to None.
            subdomains (list, optional): list of festimx.Subdomain objects
        """
        self.mesh = mesh
        self.volume_markers = volume_markers
        self.surface_markers = surface_markers
        self.subdomains = subdomains

        self.dx = None
        self.ds = None

        if self.mesh is not None:
            # create cell to facet connectivity
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim, self.mesh.topology.dim - 1
            )

            # create facet to cell connectivity
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim - 1, self.mesh.topology.dim
            )

    def define_markers(self):
        self.volume_markers = self.define_volume_markers()

        self.surface_markers = self.define_surface_markers()

    def define_measures(self):
        """Creates the ufl.Measure objects for self.dx and self.ds"""

        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.surface_markers)
        self.dx = Measure("dx", domain=self.mesh, subdomain_data=self.volume_markers)


class MeshXDMF(Mesh):
    def __init__(self, cell_file, facet_file, subdomains) -> None:
        """_summary_
        Args:
            cell_file (str): _description_
            facet_file (str): _description_
            subdomains (list, optional): list of festimx.Subdomain objects
        """

        self.cell_file = cell_file
        self.facet_file = facet_file

        volume_file = XDMFFile(MPI.COMM_WORLD, self.cell_file, "r")
        mesh = volume_file.read_mesh(name="Grid")

        super().__init__(mesh=mesh, subdomains=subdomains)

    def define_surface_markers(self):
        """Creates the surface markers
        Returns:
            dolfinx.MeshTags: the tags containing the surface
                markers
        """
        bondary_file = XDMFFile(MPI.COMM_WORLD, self.facet_file, "r")
        mesh_tags_facets = bondary_file.read_meshtags(self.mesh, name="Grid")

        return mesh_tags_facets

    def define_volume_markers(self):
        """Creates the volume markers
        Returns:
            dolfinx.MeshTags: the tags containing the volume
                markers
        """
        volume_file = XDMFFile(MPI.COMM_WORLD, self.cell_file, "r")
        mesh_tags_cells = volume_file.read_meshtags(self.mesh, name="Grid")

        return mesh_tags_cells


class boundary_condition:
    """Base boundary condition class

    Args:
        surfaces (list, int): the surfaces of the BC
    """

    def __init__(self, surfaces) -> None:

        if not isinstance(surfaces, list):
            surfaces = [surfaces]

        self.surfaces = surfaces


class flux_bc(boundary_condition):
    """Boundary condition applying flux

    Args:
        surfaces (list, int): surfaces of the BC
        value (float, int, ufl.expression): value of the flux. Deaults to
        None
    """

    def __init__(self, surfaces, value=None, **kwargs) -> None:
        super().__init__(surfaces=surfaces, **kwargs)
        self.value = value

    def create_form(self, T):
        self.form = self.value


class convenctive_flux_bc(flux_bc):
    """fluxbc subclass for convective heat flux
    -lambda * grad(T) * n = h_coeff * (T - T_ext)

    Args:
        h_coeff (float, ufl.expression): heat transfer coefficient (W/ms/K)
        T_ext (float, ufl.expression): fluid bulk temperature (K)
        surfaces (list, int): surfaces of the BC
    """

    def __init__(self, h_coeff, T_ext, surfaces) -> None:
        super().__init__(surfaces=surfaces)
        self.h_coeff = h_coeff
        self.T_ext = T_ext

    def create_form(self, T):
        self.form = -self.h_coeff * (T - self.T_ext)


class source:
    """Volumetric source term

    Args:
        value: (float, int, ufl.expression): the value of the
            volumteric source term
        volume (int, list): the volume in which the source is
            applied
        mesh (fenics.mesh, optional): the domain in which the
            constant value is applied
    """

    def __init__(self, value, volumes, mesh=None) -> None:

        if not isinstance(volumes, list):
            volumes = [volumes]
        self.volumes = volumes

        if isinstance(value, (float, int)):
            if mesh is None:
                raise ValueError("Mesh domain needs to be defined")
            self.value = Constant(mesh, PETSc.ScalarType(value))
        else:
            self.value = value

from dolfinx.io import XDMFFile
from mpi4py import MPI
from Mesh import Mesh


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

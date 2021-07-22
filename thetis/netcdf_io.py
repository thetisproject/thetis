import netCDF4
import firedrake as fd
from .utility import get_functionspace


def write_nc_field(function, filename, fieldname):
    """
    Write Function to disk in netCDF format

    :arg function: Firedrake function to export
    :arg string filename: output filename
    :arg string fieldname: canonical field name
    """
    fs = function.function_space()
    mesh = fs.mesh()

    tri_mesh = mesh.ufl_cell() == fd.triangle
    dg_family = 'DG' if tri_mesh else 'DQ'
    cg_family = 'CG' if tri_mesh else 'Q'
    fs_p0 = get_functionspace(mesh, dg_family, 0)
    fs_p1 = get_functionspace(mesh, cg_family, 1)
    fs_p1dg = get_functionspace(mesh, dg_family, 1)

    elem = fs.ufl_element()
    family = elem.family()
    degree = elem.degree()
    supported = [
        ('Discontinuous Lagrange', 0),
        ('Discontinuous Lagrange', 1),
        ('DQ', 0),
        ('DQ', 1),
        ('Lagrange', 1),
        ('Q', 1),
    ]
    assert (family, degree) in supported, \
        f'Unsupported function space: "{family}" degree={degree}'
    data_at_cells = degree == 0
    is_dg = family in ['Discontinuous Lagrange', 'DQ']
    fs_vertex = fs_p1dg if (is_dg and degree == 1) else fs_p1
    is_vector = fs.shape != ()
    if is_vector:
        vector_dim = fs.shape[0]
    face_nodes = elem.cell().num_vertices()

    gdim = mesh.geometric_dimension()
    tdim = mesh.topological_dimension()
    if tdim != 2:
        raise NotImplementedError(f'Fields of dimension {gdim} are not supported')
    if gdim != 2:
        raise NotImplementedError(f'Mesh coordinates of dimension {gdim} are not supported')

    # define mesh connectivity
    global_nb_cells = fs_p0.dim()
    global_nb_vertices = fs_vertex.dim()

    def get_lg_index(fs):
        local_nb_dofs = fs.dof_dset.size * fs.dof_dset.cdim
        local2global_map = fs.dof_dset.lgmap
        local2global_dof_ix = local2global_map.getIndices()[:local_nb_dofs]
        return local2global_dof_ix

    local2global_cell_ix = get_lg_index(fs_p0)
    local2global_vertex_ix = get_lg_index(fs_vertex)
    local2global_map = fs_vertex.dof_dset.lgmap
    local_nb_cells = fs_p0.dof_dset.size * fs_p0.dof_dset.cdim
    local_cell_nodes = fs_vertex.cell_node_list[:local_nb_cells, :]
    global_cell_nodes = local2global_map.apply(local_cell_nodes).reshape(local_cell_nodes.shape)

    # make coordinate field
    elem = fs_vertex.ufl_element()
    family = elem.family()
    degree = elem.degree()
    fs_coords = get_functionspace(mesh, family, degree, vector=True)
    f_coords = fd.Function(fs_coords)
    x, y = fd.SpatialCoordinate(mesh)
    f_coords.interpolate(fd.as_vector((x, y)))

    with netCDF4.Dataset(filename, 'w', parallel=True) as ncfile:
        ncfile.createDimension('face', global_nb_cells)
        ncfile.createDimension('face_nb_nodes', face_nodes)
        ncfile.createDimension('vertex', global_nb_vertices)

        mesh_prefix = 'Mesh'

        var = ncfile.createVariable(
            f'{mesh_prefix}_face_nodes', 'u8', ('face', 'face_nb_nodes'))
        var.start_index = 0
        var[local2global_cell_ix, :] = global_cell_nodes

        for i, label in zip(range(gdim), ('x', 'y', 'z')):
            var = ncfile.createVariable(f'{mesh_prefix}_node_{label}', 'f', ('vertex', ))
            var[local2global_vertex_ix] = f_coords.dat.data_ro[:, i]

        if data_at_cells:
            l2g_ix = local2global_cell_ix
            shape = ('face', )
        else:
            l2g_ix = local2global_vertex_ix
            shape = ('vertex', )
        if is_vector:
            for i, label in zip(range(vector_dim), ('x', 'y', 'z')):
                var = ncfile.createVariable(f'{fieldname}_{label}', 'f', shape)
                var[l2g_ix] = function.dat.data_ro[:, i]
        else:
            var = ncfile.createVariable(fieldname, 'f', shape)
            var[l2g_ix] = function.dat.data_ro

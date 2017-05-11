from thetis import *
from mpi4py import MPI

n_layers = 6
lx = 100e3
ly = 3000.
nx = 80
ny = 3
mesh2d = RectangleMesh(nx, ny, lx, ly)

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
P0_2d = FunctionSpace(mesh2d, 'DG', 0)
bathymetry_2d = Function(P1_2d, name='Bathymetry')

#z_levels = np.arange(0, 80, 20)
z_levels = np.arange(0, 80, 2)
#z_levels = np.array([0, 5, 7, 10, 13, 18, 21, 25])

depth_max = 20.0
depth_min = 7.0
xy = SpatialCoordinate(mesh2d)
bathymetry_2d.interpolate(depth_max - (depth_max-depth_min)*xy[0]/lx)

bathymetry_p0_2d = Function(P0_2d, name='Bathymetry')
bathymetry_p0_2d.interpolate(depth_max - (depth_max-depth_min)*xy[0]/lx)

# compute correct number of levels for each element column
P0v_2d = VectorFunctionSpace(mesh2d, 'DG', 0)
levels_2d = Function(P0v_2d, name='levels', dtype=np.int32).assign(0.0)

n_nodes = 3
z_lev_op2 = op2.Global(len(z_levels), z_levels, dtype=np.float, name='z_levels')
code = """
    void my_kernel(int **levels, double **bathymetry, double* z_levels)
    {
        //double mean_depth = 0;
        //for (int i = 0; i < %(nnodes)d; i++) {
        //    mean_depth += bathymetry[i][0];
        //}
        //mean_depth /= %(nnodes)d;
        double max_depth = 0;
        for (int i = 0; i < %(nnodes)d; i++) {
            max_depth = fmax(max_depth, bathymetry[i][0]);
        }
        int i = 0;
        for (; i < %(n_z_levels)d; i++) {
            if (max_depth < z_levels[i] + %(TOL)f) {
                break;
            }
        }
        levels[0][0] = 0;
        levels[0][1] = i;
    }"""
kernel = op2.Kernel(code % {'nnodes': n_nodes, 'n_z_levels': len(z_levels), 'TOL': 1e-4}, 'my_kernel')
op2.par_loop(
    kernel,
    mesh2d.cell_set,
    levels_2d.dat(op2.WRITE, levels_2d.function_space().cell_node_map()),
    bathymetry_2d.dat(op2.READ, bathymetry_2d.function_space().cell_node_map()),
    z_lev_op2(op2.READ),
    iterate=op2.ALL
)
max_levels = mesh2d.comm.allreduce(levels_2d.dat.data.max(), op=MPI.MAX)
print 'max nlev', max_levels
levels_2d.dat.data[:, 0] = max_levels - levels_2d.dat.data[:, 1]
#print levels_2d.dat.data

mesh = ExtrudedMesh(mesh2d, layers=levels_2d.dat.data, layer_height=1.0)
#print mesh.coordinates.dat.data

sigma_coords = mesh.coordinates.dat.data[:, 2]  # 0 at bottom, 1 at surf
mesh.coordinates.dat.data[:, 2] = (- sigma_coords + max_levels)

#import matplotlib.pyplot as plt
#plt.plot(mesh.coordinates.dat.data[:, 0], mesh.coordinates.dat.data[:, 2], 'k.')
#plt.show()

coordinates = mesh.coordinates
fs_3d = coordinates.function_space()
fs_2d = bathymetry_2d.function_space()
new_coordinates = Function(fs_3d)

# number of nodes in vertical direction
n_vert_nodes = fs_3d.finat_element.space_dimension() / fs_2d.finat_element.space_dimension()

nodes = get_facet_mask(fs_3d, 'geometric', 'bottom')
idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='node_idx')
kernel = op2.Kernel("""
    void my_kernel(double **new_coords, double **old_coords, double **bath2d, int *idx, double* z_levels) {
        for ( int d = 0; d < %(nodes)d; d++ ) {
            for ( int e = 0; e < %(v_nodes)d; e++ ) {
                new_coords[idx[d]+e][0] = old_coords[idx[d]+e][0];
                new_coords[idx[d]+e][1] = old_coords[idx[d]+e][1];
                int iz = (int)old_coords[idx[d]+e][2];
                new_coords[idx[d]+e][2] = -z_levels[iz];
            }
        }
    }""" % {'nodes': fs_2d.finat_element.space_dimension(),
            'v_nodes': n_vert_nodes},
    'my_kernel')

op2.par_loop(kernel, mesh.cell_set,
                new_coordinates.dat(op2.WRITE, fs_3d.cell_node_map()),
                coordinates.dat(op2.READ, fs_3d.cell_node_map()),
                bathymetry_2d.dat(op2.READ, fs_2d.cell_node_map()),
                idx(op2.READ),
                z_lev_op2(op2.READ),
                iterate=op2.ALL)

kernel = op2.Kernel("""
    void my_kernel(double **new_coords, double **old_coords, double **bath2d, int *idx, double* z_levels) {
        for ( int d = 0; d < %(nodes)d; d++ ) {
                int iz = (int)old_coords[idx[d]+0][2];
                double z_above = -z_levels[iz-1];
                double z_below = -z_levels[iz];
                double height = z_above + bath2d[d][0];
                new_coords[idx[d]+0][2] = -bath2d[d][0];
        }
    }""" % {'nodes': fs_2d.finat_element.space_dimension(),
            'v_nodes': n_vert_nodes},
    'my_kernel')

op2.par_loop(kernel, mesh.cell_set,
                new_coordinates.dat(op2.WRITE, fs_3d.cell_node_map()),
                coordinates.dat(op2.READ, fs_3d.cell_node_map()),
                bathymetry_2d.dat(op2.READ, fs_2d.cell_node_map()),
                idx(op2.READ),
                z_lev_op2(op2.READ),
                iterate=op2.ON_BOTTOM)

mesh.coordinates.assign(new_coordinates)

#import matplotlib.pyplot as plt
#plt.plot(mesh.coordinates.dat.data[:, 0], mesh.coordinates.dat.data[:, 2], 'k.')
#plt.show()

P1DG = FunctionSpace(mesh, 'DG', 1)

func = Function(P1DG, name='tracer')

func.interpolate(CellVolume(mesh))
print 'vol', func.dat.data.min(), func.dat.data.max()

x, y, z = SpatialCoordinate(mesh)
func.project(z)

out = File('tracer.pvd')
out.write(func)
out.write(func)

u_max = 4.5
w_max = 5e-3

outputdir = 'outputs_closed_zlayers'
t_end = 6 * 3600
t_export = 900.0

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, mesh=mesh)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'ssprk22'
options.solve_salt = True
options.solve_temp = False
options.solve_vert_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = True
options.use_limiter_for_tracers = True
options.uv_lax_friedrichs = None
options.tracer_lax_friedrichs = None
options.t_export = t_export
options.t_end = t_end
options.outputdir = outputdir
options.u_advection = Constant(u_max)
options.w_advection = Constant(w_max)
options.check_vol_conservation_2d = True
options.check_vol_conservation_3d = True
options.check_salt_conservation = True
options.check_salt_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d', 'uv_bottom_2d']

solver_obj.create_equations()
solver_obj.w_solver.solve()

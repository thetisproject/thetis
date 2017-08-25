"""
Define DOME inflow temperature/density and velocity fields.

Inflow field is stratified and in geostrophic balance.
Equations are based on Legg et al. (2006)

"""
import numpy

# constants
bay_width = 100e3
bay_depth = h_e = 600.
bay_length = 50e3
bay_x_lim = [800e3, 900e3]
basin_lx = 1100e3
basin_ly = 600e3
basin_extend = 120e3
basin_depth = 3600.

# define bathymetry slope
y_slope = [300e3, 600e3]
depth_lim = [3600., 600.]

h_0 = 300.0       # inflow height [m]
delta_rho = 2.0   # density anomaly [kg/m3]
rho_0 = 1000.0    # reference density [kg/m3]
g = 9.81          # gravitational acceleration [m/s2]
b_0 = 0.0         # buoyancy at surface [m/s2]
f_0 = 1e-4        # Coriolis parameter [1/s]
Ri_m = 1.0/3.0    # minimum gradient Richardson number

# temperature limits for a linear equation of state
temp_lim = [10.0, 20.0]
alpha = delta_rho/(temp_lim[1] - temp_lim[0])  # thermal expansion coeff
beta = 0.0  # haline contraction coeff
t_ref = temp_lim[1]
salt_const = 0.0

# buoyancy anomaly
# b = -g/rho_0 (rho - rho_0)  [m/s2]
db_0 = -g/rho_0 * -delta_rho

# inflow velocity amplitude [m/s]
U_0 = numpy.sqrt(db_0*h_0)

# buoyancy frequency of the background stratification
# N2 = - g/rho_0 d(rho)/dz = d(b)/dz
N2 = db_0/basin_depth

# length scale of decay
L_rho = U_0/f_0

# inflow depth as a function of x_w = distance from western wall of the bay
h_func = lambda x_w: h_0 * numpy.exp(-x_w/L_rho)

# modified z coordinate
z_star_func = lambda x_w, z: (z - h_func(x_w) + h_e + 1e-3)/h_func(x_w)

# function F(z_star)
F_func = lambda z_star: numpy.maximum(numpy.minimum(1./Ri_m * (z_star)/(z_star + 1.) + 0.5, 1.0), 0.0)

# solution for buoyancy, b = min(b_a, b_b)
b_a_func = lambda x_w, z: b_0 - db_0*(1. - F_func(z_star_func(x_w, z)))
b_b_func = lambda x_w, z: b_0 + N2*z  # background stratification
b_func = lambda x_w, z: numpy.minimum(b_a_func(x_w, z), b_b_func(x_w, z))

v_func = lambda x_w, z: -U_0*numpy.exp(-x_w/L_rho)*(1 - F_func(z_star_func(x_w, z)))

# compute density solution
rho_func = lambda x_w, z: -rho_0/g*b_func(x_w, z) + rho_0

# compute temperature solution from linear equation of state
# rho = rho_0 - alpha*(t - t_ref)
temp_func = lambda x_w, z: -(rho_func(x_w, z) - rho_0)/alpha + t_ref


def plot_fields():
    """plots inflow temperature and velocity fields"""

    import matplotlib.pyplot as plt
    x_plot = numpy.linspace(0, bay_width, 500)
    z_plot = numpy.linspace(-bay_depth, 0, 400)

    X, Z = numpy.meshgrid(x_plot, z_plot)

    fig, ax_list = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    b_plot = b_func(X, Z)
    v_plot = v_func(X, Z)
    rho_plot = rho_func(X, Z)
    temp_plot = temp_func(X, Z)

    print('buoyancy anomaly= {:}'.format(db_0))
    print('U_0= {:}'.format(U_0))
    print('N2= {:}, N= {:}'.format(N2, numpy.sqrt(N2)))
    print('buoyancy= {:} .. {:}'.format(b_plot.min(), b_plot.max()))
    print('density= {:} .. {:}'.format(rho_plot.min(), rho_plot.max()))
    print('temperature= {:} .. {:}'.format(temp_plot.min(), temp_plot.max()))
    print('v velocity= {:} .. {:}'.format(v_plot.min(), v_plot.max()))

    cmap = plt.get_cmap('inferno_r')

    ax = ax_list[0]
    levs = numpy.linspace(temp_plot.min(), temp_plot.max(), 32)
    cax = ax.contourf(x_plot, z_plot, temp_plot, levels=levs, cmap=cmap)
    fig.colorbar(cax, ax=ax, ticks=levs, orientation='vertical')
    ax.set_title('Temperature [C]')

    ax = ax_list[1]
    levs = numpy.linspace(v_plot.min(), v_plot.max(), 32)
    cax = ax.contourf(x_plot, z_plot, v_plot, 32, cmap=cmap)
    fig.colorbar(cax, ax=ax, ticks=levs, orientation='vertical')
    ax.set_title('Y-Velocity [m/s]')

    imgfile = 'inflow_fields.png'
    print('saving image: {:}'.format(imgfile))
    fig.savefig(imgfile, bbox_inches='tight')


if __name__ == '__main__':
    plot_fields()

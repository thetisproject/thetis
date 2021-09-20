"""
Tests convergence of div(uv) in 2D
"""
from firedrake import *
from thetis.utility import get_functionspace
from scipy import stats
import numpy
import os

op2.init(log_level=WARNING)


def compute(refinement=1, order=1, do_export=False):
    print('--- soving refinement {:}'.format(refinement))
    n = 5*refinement
    mesh = UnitSquareMesh(n, n)

    family = 'DG'
    p0dg = get_functionspace(mesh, family, order-1)
    p1dg = get_functionspace(mesh, family, order)
    p1dg_v = get_functionspace(mesh, family, order, vector=True)
    p1dg_ho = get_functionspace(mesh, family, order + 2)
    p1dg_v_ho = get_functionspace(mesh, family, order + 2, vector=True)

    lx = 1.0
    x, y = SpatialCoordinate(mesh)
    uv_expr = as_vector((sin(0.2*pi*(3.0*x + 1.0*y)/lx),
                         0.2*sin(0.2*pi*(1.0*x + 3.0*y)/lx)))
    div_expr = 0.12*pi*cos(0.2*pi*(1.0*x + 3.0*y)/lx)/lx + 0.6*pi*cos(0.2*pi*(3.0*x + 1.0*y)/lx)/lx

    div_uv = Function(p1dg, name='div')
    div_uv.project(div_expr)
    div_uv_source = Function(p0dg, name='div')
    div_uv_source.project(div_expr)
    uv = Function(p1dg_v, name='uv')
    uv.project(uv_expr)

    div_ana = Function(p1dg_ho, name='div_ho')
    div_ana.project(div_expr)
    uv_ana = Function(p1dg_v_ho, name='uv+ho')
    uv_ana.project(uv_expr)

    if do_export:
        print('analytical div {:} {:}'.format(div_uv.dat.data.min(), div_uv.dat.data.max()))
        # export analytical solutions
        out_uv = File('uv.pvd')
        out_div = File('div.pvd')
        out_uv.write(uv)
        out_div.write(div_uv)

    div_uv.assign(0)

    test = TestFunction(p1dg)
    tri = TrialFunction(p1dg)
    normal = FacetNormal(mesh)

    # solve div_uv = div(uv)
    a = inner(test, tri)*dx

    # div(uv) point-wise
    # l = inner(div(uv), test)*dx

    # div(uv) integrated by parts
    l = -inner(grad(test), uv)*dx + dot(avg(uv), jump(test, normal))*dS + test*dot(uv, normal)*ds

    # analytical source
    # l = inner(test, div_uv_source)*dx

    solve(a == l, div_uv)

    if do_export:
        print('numerical div {:} {:}'.format(div_uv.dat.data.min(), div_uv.dat.data.max()))
        # export numerical solutions
        out_uv.write(uv)
        out_div.write(div_uv)

    l2err_uv = errornorm(uv_ana, uv)
    l2err_div = errornorm(div_ana, div_uv)
    print('L2 norm uv {:}'.format(l2err_uv))
    print('L2 norm div {:}'.format(l2err_div))
    return l2err_uv, l2err_div


def convergence_test(ref_list, order=1, export=False, saveplot=False):
    """Run convergence test for the given mesh refiments"""
    err = []
    for r in ref_list:
        err.append(compute(r, order=order, do_export=export))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(err))
    y_log_div = y_log[:, 1]
    y_log_uv = y_log[:, 0]

    def check_convergence(x_log, y_log, expected_slope, test_name, field_str, saveplot):
        slope_rtol = 0.1
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        if saveplot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            # plot points
            ax.plot(x_log, y_log, 'k.')
            x_min = x_log.min()
            x_max = x_log.max()
            offset = 0.05*(x_max - x_min)
            npoints = 50
            xx = numpy.linspace(x_min - offset, x_max + offset, npoints)
            yy = intercept + slope*xx
            # plot line
            ax.plot(xx, yy, linestyle='--', linewidth=0.5, color='k')
            ax.text(xx[2*int(npoints/3)], yy[2*int(npoints/3)], '{:4.2f}'.format(slope),
                    verticalalignment='top',
                    horizontalalignment='left')
            ax.set_xlabel('log10(dx)')
            ax.set_ylabel('log10(L2 error)')
            ax.set_title(field_str)
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
            order_str = 'o{:}'.format(order)
            imgfile = '_'.join(['convergence', test_name, field_str, ref_str, order_str])
            imgfile += '.png'
            imgdir = '.'
            imgfile = os.path.join(imgdir, imgfile)
            print('saving figure {:}'.format(imgfile))
            plt.savefig(imgfile, dpi=200, bbox_inches='tight')
        if expected_slope is not None:
            err_msg = '{:}: Wrong convergence rate {:.4f}, expected {:.4f}'.format(test_name, slope, expected_slope)
            assert abs(slope - expected_slope)/expected_slope < slope_rtol, err_msg
            print('{:}: {:} convergence rate {:.4f} PASSED'.format(test_name, field_str, slope))
        else:
            print('{:}: {:} convergence rate {:.4f}'.format(test_name, field_str, slope))
        return slope
    check_convergence(x_log, y_log_div, order, 'divergence_2d', 'div', saveplot)
    check_convergence(x_log, y_log_uv, order + 1, 'divergence_2d', 'uv', saveplot)


def test_divergence_2d(**kwargs):
    convergence_test([1, 2, 4, 8], **kwargs)


if __name__ == '__main__':
    test_divergence_2d(saveplot=True)

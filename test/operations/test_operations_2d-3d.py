import unittest
import numpy as np


class mesh2d3dOperationsBase(object):

    def createSpaces(self, fs_name, fs_order, vfs_name, vfs_order):
        from firedrake import *
        mesh2d = UnitSquareMesh(5, 5)
        mesh = ExtrudedMesh(mesh2d, layers=10, layer_height=-0.1)

        P1_2d = FunctionSpace(mesh2d, fs_name, fs_order)
        U_2d = VectorFunctionSpace(mesh2d, fs_name, fs_order)

        P1 = FunctionSpace(mesh, fs_name, fs_order,
                           vfamily=vfs_name, vdegree=vfs_order)
        U = VectorFunctionSpace(mesh, fs_name, fs_order,
                                vfamily=vfs_name, vdegree=vfs_order)

        self.uv_3d = Function(U, name='Velocity')
        self.uv_3dint = Function(U, name='Velocity')
        self.uv_2d = Function(U_2d, name='Velocity')

        self.uv_3d_x = Function(U, name='Velocity')
        self.uv_2d_x = Function(U_2d, name='Velocity')

        self.c3d = Function(P1, name='Tracer')
        self.c2d = Function(P1_2d, name='Tracer')

        self.c3d_x = Function(P1, name='Tracer')
        self.c2d_x = Function(P1_2d, name='Tracer')

        self.const3d = Function(P1)
        self.const3d.interpolate(Expression(('3.0')))

        self.c3d.interpolate(Expression(('x[2] + 2.0')))
        self.c2d.interpolate(Expression(('4.0')))
        self.c3d_x.interpolate(Expression(('x[0] + 2.0')))
        self.c2d_x.interpolate(Expression(('2*x[0]')))
        self.uv_3d.interpolate(Expression(('x[2] + 1.0', '2.0*x[2] + 4.0', '3.0*x[2] + 6.0')))
        self.uv_2d.interpolate(Expression(('4.0', '8.0')))
        self.uv_3d_x.interpolate(Expression(('x[0] + 1.0', '2.0*x[1] + 4.0', '3.0*x[0]*x[2] + 6.0')))
        self.uv_2d_x.interpolate(Expression(('4.0*x[0]', '8.0*x[1]')))

    def test_copy3dFieldTo2d(self):
        import cofs.utility as utility
        utility.copy3dFieldTo2d(self.c3d, self.c2d, useBottomValue=True)
        self.assertTrue(np.allclose(self.c2d.dat.data[:], 1.0))

        utility.copy3dFieldTo2d(self.c3d, self.c2d, useBottomValue=False)
        self.assertTrue(np.allclose(self.c2d.dat.data[:], 2.0))

    def test_copy3dFieldTo2d_vec(self):
        import cofs.utility as utility
        utility.copy3dFieldTo2d(self.uv_3d, self.uv_2d, useBottomValue=True)
        self.assertTrue(np.allclose(self.uv_2d.dat.data[:, 0], 0.0))
        self.assertTrue(np.allclose(self.uv_2d.dat.data[:, 1], 2.0))

        utility.copy3dFieldTo2d(self.uv_3d, self.uv_2d, useBottomValue=False)
        self.assertTrue(np.allclose(self.uv_2d.dat.data[:, 0], 1.0))
        self.assertTrue(np.allclose(self.uv_2d.dat.data[:, 1], 4.0))

    def test_copy3dFieldTo2d_x(self):
        import cofs.utility as utility
        utility.copy3dFieldTo2d(self.c3d_x, self.c2d_x, useBottomValue=True)
        self.assertTrue(np.allclose(self.c2d_x.dat.data.min(), 2.0))
        self.assertTrue(np.allclose(self.c2d_x.dat.data.max(), 3.0))

        utility.copy3dFieldTo2d(self.c3d_x, self.c2d_x, useBottomValue=False)
        self.assertTrue(np.allclose(self.c2d_x.dat.data.min(), 2.0))
        self.assertTrue(np.allclose(self.c2d_x.dat.data.max(), 3.0))

    def test_copy3dFieldTo2d_x_vec(self):
        import cofs.utility as utility
        utility.copy3dFieldTo2d(self.uv_3d_x, self.uv_2d_x, useBottomValue=True)
        self.assertTrue(np.allclose(self.uv_2d_x.dat.data[:, 0].min(), 1.0))
        self.assertTrue(np.allclose(self.uv_2d_x.dat.data[:, 0].max(), 2.0))
        self.assertTrue(np.allclose(self.uv_2d_x.dat.data[:, 1].min(), 4.0))
        self.assertTrue(np.allclose(self.uv_2d_x.dat.data[:, 1].max(), 6.0))

        utility.copy3dFieldTo2d(self.uv_3d_x, self.uv_2d_x, useBottomValue=False)
        self.assertTrue(np.allclose(self.uv_2d_x.dat.data[:, 0].min(), 1.0))
        self.assertTrue(np.allclose(self.uv_2d_x.dat.data[:, 0].max(), 2.0))
        self.assertTrue(np.allclose(self.uv_2d_x.dat.data[:, 1].min(), 4.0))
        self.assertTrue(np.allclose(self.uv_2d_x.dat.data[:, 1].max(), 6.0))

    def test_copy2dFieldTo3d(self):
        import cofs.utility as utility
        utility.copy2dFieldTo3d(self.c2d, self.c3d)
        self.assertTrue(np.allclose(self.c3d.dat.data[:], 4.0))

    def test_copy2dFieldTo3d_x(self):
        import cofs.utility as utility
        utility.copy2dFieldTo3d(self.c2d_x, self.c3d_x)
        self.assertTrue(np.allclose(self.c3d_x.dat.data.min(), 0.0))
        self.assertTrue(np.allclose(self.c3d_x.dat.data.max(), 2.0))

    def test_copy2dFieldTo3d_x_vec(self):
        import cofs.utility as utility
        utility.copy2dFieldTo3d(self.uv_2d_x, self.uv_3d_x)
        self.assertTrue(np.allclose(self.uv_3d_x.dat.data[:, 0].min(), 0.0))
        self.assertTrue(np.allclose(self.uv_3d_x.dat.data[:, 0].max(), 4.0))
        self.assertTrue(np.allclose(self.uv_3d_x.dat.data[:, 1].min(), 0.0))
        self.assertTrue(np.allclose(self.uv_3d_x.dat.data[:, 1].max(), 8.0))

    def test_copy2dFieldTo3d_vec(self):
        import cofs.utility as utility
        utility.copy2dFieldTo3d(self.uv_2d, self.uv_3d)
        self.assertTrue(np.allclose(self.uv_3d.dat.data[:, 0], 4.0))
        self.assertTrue(np.allclose(self.uv_3d.dat.data[:, 1], 8.0))


class test_mesh2d3dOperationsP1(mesh2d3dOperationsBase, unittest.TestCase):

    def setUp(self):
        mesh2d3dOperationsBase.createSpaces(self, 'CG', 1, 'CG', 1)


class test_mesh2d3dOperationsP2(mesh2d3dOperationsBase, unittest.TestCase):

    def setUp(self):
        mesh2d3dOperationsBase.createSpaces(self, 'CG', 2, 'CG', 2)


class test_mesh2d3dOperationsP1DG(mesh2d3dOperationsBase, unittest.TestCase):
    # FIXME somehow DG tests fail if ran in combination with CG tests?
    def setUp(self):
        mesh2d3dOperationsBase.createSpaces(self, 'DG', 1, 'DG', 1)


class test_mesh2d3dOperationsP2DG(mesh2d3dOperationsBase, unittest.TestCase):
    # FIXME somehow DG tests fail if ran in combination with CG tests?
    def setUp(self):
        mesh2d3dOperationsBase.createSpaces(self, 'DG', 2, 'DG', 2)


if __name__ == '__main__':
    """Run all tests"""
    unittest.main()

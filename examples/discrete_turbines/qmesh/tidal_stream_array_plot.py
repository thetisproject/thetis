def mesh():
    '''Todo: add docstring '''
    # Reading in the shapefile describing the domain boundaries, and creating a gmsh file.
    boundaries = qmesh.vector.Shapes()
    boundaries.fromFile('outline.shp')
    loopShapes = qmesh.vector.identifyLoops(boundaries, isGlobal=False, defaultPhysID=1000, fixOpenLoops=True)
    polygonShapes = qmesh.vector.identifyPolygons(loopShapes, smallestNotMeshedArea=100, meshedAreaPhysID=1)

    array_plot = qmesh.vector.Shapes()
    array_plot.fromFile('array_plot.shp')
    array_shapes = qmesh.vector.identifyLoops(array_plot, isGlobal=False, defaultPhysID=1000, fixOpenLoops=True)
    array_polygon = qmesh.vector.identifyPolygons(array_shapes, smallestNotMeshedArea=100, meshedAreaPhysID=2)

    gradation = []
    gradation.append(qmesh.raster.meshMetricTools.gradationToShapes())
    gradation[-1].setShapes(array_polygon)
    gradation[-1].setRasterBounds(-100, 800., -100, 400.0)
    gradation[-1].setRasterResolution(450, 150)
    gradation[-1].setGradationParameters(3.0, 50.0, 500.0, 10)
    gradation[-1].calculateLinearGradation()
    gradation[-1].writeNetCDF('gradation_to_turbine.nc')

    # Calculate overall mesh-metric raster
    if len(gradation) == 1:
        meshMetricRaster = gradation[0]
    else:
        meshMetricRaster = qmesh.raster.meshMetricTools.minimumRaster(gradation)

    meshMetricRaster.writeNetCDF('meshMetric.nc')

    # Create domain object and write gmsh files.
    domain = qmesh.mesh.Domain()
    domainLines, domainPolygons = qmesh.vector.insertRegions(loopShapes, polygonShapes, array_shapes, array_polygon)
    domain.setGeometry(domainLines, domainPolygons)
    domain.setMeshMetricField(meshMetricRaster)
    domain.setTargetCoordRefSystem('EPSG:32630', fldFillValue=1000.0)

    # Meshing
    domain.gmsh(geoFilename='tidal_0.geo',
                fldFilename='tidal_0.fld',
                mshFilename='tidal_mesh_plot.msh')


def convertMesh(meshname):
    '''Todo: add docstring '''
    mesh = qmesh.mesh.Mesh()
    mesh.readGmsh(meshname+'.msh', 'EPSG:32630')
    mesh.writeShapefile(meshname)


if __name__ == '__main__':
    import qmesh
    # Initialising qgis API
    qmesh.initialise()
    mesh()
    convertMesh('tidal_mesh_plot')

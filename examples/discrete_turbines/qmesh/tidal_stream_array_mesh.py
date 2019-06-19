def mesh():
    '''Todo: add docstring '''
    # Reading in the shapefile describing the domain boundaries, and creating a gmsh file.
    boundaries = qmesh.vector.Shapes()
    boundaries.fromFile('outline.shp')
    loopShapes = qmesh.vector.identifyLoops(boundaries, isGlobal=False, defaultPhysID=1000, fixOpenLoops=True)
    polygonShapes = qmesh.vector.identifyPolygons(loopShapes, smallestNotMeshedArea=100, meshedAreaPhysID=1)

    inner_plot_lines = qmesh.vector.Shapes()
    inner_plot_lines.fromFile('turbine_circles.shp')
    inner_plot_loops = qmesh.vector.identifyLoops(inner_plot_lines, fixOpenLoops=True, extraPointsPerVertex=10)
    inner_plot_polygons = qmesh.vector.identifyPolygons(inner_plot_loops, meshedAreaPhysID=1)

    gradation = []
    gradation.append(qmesh.raster.meshMetricTools.gradationToShapes())
    gradation[-1].setShapes(inner_plot_polygons)
    gradation[-1].setRasterBounds(-100, 800., -100, 400.0)
    gradation[-1].setRasterResolution(450, 150)
    gradation[-1].setGradationParameters(4.0, 50.0, 500.0, 10)
    gradation[-1].calculateLinearGradation()
    gradation[-1].writeNetCDF('gradation_to_turbine.nc')
    # gradation[-1].fromFile('gradation_to_turbineplot.nc')

    # Calculate overall mesh-metric raster
    if len(gradation) == 1:
        meshMetricRaster = gradation[0]
    else:
        meshMetricRaster = qmesh.raster.meshMetricTools.minimumRaster(gradation)

    meshMetricRaster.writeNetCDF('meshMetric.nc')

    # Create domain object and write gmsh files.
    domain = qmesh.mesh.Domain()
    domainLines, domainPolygons = qmesh.vector.insertRegions(loopShapes, polygonShapes, inner_plot_loops, inner_plot_polygons)
    domain.setGeometry(domainLines, domainPolygons)
    domain.setMeshMetricField(meshMetricRaster)
    domain.setTargetCoordRefSystem('EPSG:32630', fldFillValue=1000.0)
    # Meshing
    domain.gmsh(geoFilename='tidal_mesh.geo',
                fldFilename='tidal_mesh.fld',
                mshFilename='tidal_mesh.msh')


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
    # convertMesh('tidal_mesh')

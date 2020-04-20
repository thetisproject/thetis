import shapely.geometry
import fiona
import fiona.crs
import pyproj

UTM_ZONE30 = pyproj.Proj(proj='utm', zone=30, datum='WGS84', units='m', errcheck=True)
schema = {'geometry': 'LineString', 'properties': {'PhysID': 'int'}}
crs = fiona.crs.from_string(UTM_ZONE30.srs)

# Outline location
x0, y0, x1, y1 = 0, 0, 640, 320
features = \
    [shapely.geometry.LineString([(x0, y0), (x1, y0)]),
     shapely.geometry.LineString([(x1, y0), (x1, y1)]),
     shapely.geometry.LineString([(x1, y1), (x0, y1)]),
     shapely.geometry.LineString([(x0, y1), (x0, y0)])]
with fiona.collection("outline.shp", "w", "ESRI Shapefile", schema, crs=crs) as output:
    for i in range(len(features)):
        output.write({'geometry': shapely.geometry.mapping(features[i]), 'properties': {'PhysID': i}})

# Creating array shapefiles to introduce it as subdomain
x_a0, y_a0, x_a1, y_a1 = x1/4.0, y1/10., 2 * x1/4., y1-y1/10.
features2 = \
    [shapely.geometry.LineString([(x_a0, y_a0), (x_a1, y_a0)]),
     shapely.geometry.LineString([(x_a1, y_a0), (x_a1, y_a1)]),
     shapely.geometry.LineString([(x_a1, y_a1), (x_a0, y_a1)]),
     shapely.geometry.LineString([(x_a0, y_a1), (x_a0, y_a0)])]

with fiona.collection("array_plot.shp", "w", "ESRI Shapefile", schema, crs=crs) as output:
    for i in range(len(features2)):
        output.write({'geometry': shapely.geometry.mapping(features2[i]), 'properties': {'PhysID': 100}})

# Creating array_polygon to have a consistent gradation
schema = {'geometry': 'Polygon', 'properties': {'PhysID': 'int'}}
with fiona.collection("array_polygon.shp", "w", "ESRI Shapefile", schema, crs=crs) as output:
    output.write({'geometry': shapely.geometry.mapping(shapely.geometry.Polygon([(x_a0, y_a0), (x_a1, y_a0),
                                                                                 (x_a1, y_a1), (x_a0, y_a1)])),
                  'properties': {'PhysID': 100}})

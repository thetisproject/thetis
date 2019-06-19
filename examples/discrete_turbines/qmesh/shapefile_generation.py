import shapely.geometry
import numpy as np
import fiona.crs
import pyproj
from shapely.geometry.point import Point

UTM_ZONE30 = pyproj.Proj(
    proj='utm',
    zone=30,
    datum='WGS84',
    units='m',
    errcheck=True)
schema = {'geometry': 'LineString', 'properties': {'PhysID': 'int'}}
crs = fiona.crs.from_string(UTM_ZONE30.srs)

x0, y0, x1, y1 = 0, 0, 640, 320
features = \
    [shapely.geometry.LineString([(x0, y0), (x1, y0)]),
     shapely.geometry.LineString([(x1, y0), (x1, y1)]),
     shapely.geometry.LineString([(x1, y1), (x0, y1)]),
     shapely.geometry.LineString([(x0, y1), (x0, y0)])]
with fiona.collection("outline_2.shp", "w", "ESRI Shapefile", schema, crs=crs) as output:
        for i in range(len(features)):
            output.write({'geometry': shapely.geometry.mapping(features[i]), 'properties': {'PhysID': i}})

# Array coordinates
array_list = np.zeros((7, 2))
array_1 = np.arange(64, 320, 64)
array_2 = np.arange(64 + 32, 320-64, 64)
array_list[0:4, 0] = 640 / 3
array_list[4:, 0] = 640 / 3 + 64
array_list[0:4, 1] = array_1
array_list[4:, 1] = array_2
np.save("Turbine_coords.npy", array_list)

features2 = []
for x, y in array_list:
    p = Point(x, y)
    circle = shapely.geometry.LineString(list(p.buffer(10).exterior.coords))
    features2.append(circle)

with fiona.collection("turbine_circles.shp", "w", "ESRI Shapefile", schema, crs=crs) as output:

    for i in range(len(features2)):
        output.write({'geometry': shapely.geometry.mapping(features2[i]), 'properties': {'PhysID': 100}})

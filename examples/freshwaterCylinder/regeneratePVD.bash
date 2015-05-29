#!/bin/bash

end=288
OUT=$1
python generatePVD.py -d $OUT -e $end -n Elevation2d
python generatePVD.py -d $OUT -e $end -n Velocity2d
python generatePVD.py -d $OUT -e $end -n Velocity3d
python generatePVD.py -d $OUT -e $end -n VertVelo3d
python generatePVD.py -d $OUT -e $end -n Salinity3d
python generatePVD.py -d $OUT -e $end -n Barohead2d
python generatePVD.py -d $OUT -e $end -n Barohead3d
python generatePVD.py -d $OUT -e $end -n GJVParamH
python generatePVD.py -d $OUT -e $end -n GJVParamV
python generatePVD.py -d $OUT -e $end -n MeshVelo3d
python generatePVD.py -d $OUT -e $end -n DAVelocity2d
python generatePVD.py -d $OUT -e $end -n DAVelocity3d

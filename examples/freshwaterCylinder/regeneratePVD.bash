#!/bin/bash

end=380
OUT=$1
par="-p"
python generatePVD.py -d $OUT -e $end $par -n Elevation2d
python generatePVD.py -d $OUT -e $end $par -n Velocity2d
python generatePVD.py -d $OUT -e $end $par -n Velocity3d
python generatePVD.py -d $OUT -e $end $par -n VertVelo3d
python generatePVD.py -d $OUT -e $end $par -n Salinity3d
python generatePVD.py -d $OUT -e $end $par -n Barohead2d
python generatePVD.py -d $OUT -e $end $par -n Barohead3d
python generatePVD.py -d $OUT -e $end $par -n GJVParamH
python generatePVD.py -d $OUT -e $end $par -n GJVParamV
python generatePVD.py -d $OUT -e $end $par -n MeshVelo3d
python generatePVD.py -d $OUT -e $end $par -n DAVelocity2d
python generatePVD.py -d $OUT -e $end $par -n DAVelocity3d

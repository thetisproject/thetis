#!/bin/bash

end=540
OUT=$1
python generatePVD.py -d $OUT -e $end -n Velocity3d
python generatePVD.py -d $OUT -e $end -n VertVelo3d
python generatePVD.py -d $OUT -e $end -n Salinity3d


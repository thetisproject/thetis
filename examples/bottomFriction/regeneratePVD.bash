#!/bin/bash

end=100
OUT=$1
par="-p"
python generatePVD.py -d $OUT -e $end $par -n Elevation2d
python generatePVD.py -d $OUT -e $end $par -n Elevation3d
python generatePVD.py -d $OUT -e $end $par -n Velocity2d
python generatePVD.py -d $OUT -e $end $par -n Velocity3d
python generatePVD.py -d $OUT -e $end $par -n VertVelo3d
python generatePVD.py -d $OUT -e $end $par -n ParabVisc3d
python generatePVD.py -d $OUT -e $end $par -n ShearFreq3d
python generatePVD.py -d $OUT -e $end $par -n TurbPsi3d
python generatePVD.py -d $OUT -e $end $par -n TurbKEnergy3d
python generatePVD.py -d $OUT -e $end $par -n EddyVisc3d
python generatePVD.py -d $OUT -e $end $par -n TurbEps3d
python generatePVD.py -d $OUT -e $end $par -n TurbLen3d

#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Missing arguments: output_dir pvstate.pvsm"
    exit -1
fi

OUTDIR=$1
VISUSTATE=$2

# generate pdv file
./regeneratePVD.bash $OUTDIR

# change input dir in paraview visualization state
sed "s/outputs/$OUTDIR/g" $VISUSTATE > tmp.pvsm

# launch paraview
paraview --state=tmp.pvsm &> pvoutput.txt &


#declare grid_file=pointwise/caps.GeomToMesh.lb8.ugrid # with pointwise
declare grid_file=aflr3/aflr3_0.lb8.ugrid # with AFLR3
export HSCT_DIR=~/git/f2f-cases/hsct
export HSCT_GEOM=$HSCT_DIR/geometry
export HSCT_GEOM2=$HSCT_GEOM/_hsct_chamfer3
cp $HSCT_GEOM2/capsFluid/Scratch/$grid_file $HSCT_DIR/cfd/cruise_turb/Flow/hsct-turb.lb8.ugrid
cp $HSCT_GEOM2/capsFluid/Scratch/$grid_file $HSCT_DIR/cfd/climb_turb/Flow/hsct-turb.lb8.ugrid
cp $HSCT_GEOM2/capsFluid/Scratch/$grid_file $HSCT_DIR/cfd/climb_laminar/Flow/hsct-turb.lb8.ugrid

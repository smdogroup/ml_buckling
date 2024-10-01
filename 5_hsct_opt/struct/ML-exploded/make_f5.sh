# run this yourself or here (this first call build_exploded_mesh takes awhile)
# mpirun -n 3 python build_exploded_mesh.py

cp capsExploded_0/Scratch/tacs/tacs.* upperOML/
cd upperOML/
python ../run_struct.py --exploded 1
$TACS_DIR/extern/f5tovtk/f5tovtk_element *.f5
cd ../
cp capsExploded_1/Scratch/tacs/tacs.* int-struct/
cd int-struct/
python ../run_struct.py --exploded 2
$TACS_DIR/extern/f5tovtk/f5tovtk_element *.f5
cd ../
cp capsExploded_2/Scratch/tacs/tacs.* lowerOML/
cd lowerOML/
python ../run_struct.py --exploded 3
$TACS_DIR/extern/f5tovtk/f5tovtk_element *.f5
cd ../
# once this is done run paraview and make the visualization

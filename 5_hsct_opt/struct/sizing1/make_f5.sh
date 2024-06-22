cp capsExploded_0/Scratch/tacs/tacs.* upperOML/
cd upperOML/
python run_tacs.py
$TACS_DIR/extern/f5tovtk/f5tovtk *.f5
cd ../
cp capsExploded_1/Scratch/tacs/tacs.* int-struct/
cd int-struct/
python run_tacs.py
$TACS_DIR/extern/f5tovtk/f5tovtk *.f5
cd ../
cp capsExploded_2/Scratch/tacs/tacs.* lowerOML/
cd lowerOML/
python run_tacs.py
$TACS_DIR/extern/f5tovtk/f5tovtk *.f5
cd ../
# once this is done run paraview and make the visualization
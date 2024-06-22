cp _fuselageTail_0/Scratch/tacs/tacs.* fuselage/
cd fuselage/
python run_tacs.py
$TACS_DIR/extern/f5tovtk/f5tovtk_element *.f5
cd ../

cp _fuselageTail_1/Scratch/tacs/tacs.* htail/
cd htail/
python run_tacs.py
$TACS_DIR/extern/f5tovtk/f5tovtk_element *.f5
cd ../
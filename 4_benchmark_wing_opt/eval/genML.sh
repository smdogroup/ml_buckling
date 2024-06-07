rm -rf capsStruct*
python make_exploded.py --useML True --exploded 1
mv capsStruct1_0/Scratch/tacs/tacs_output_file000.f5 ML-view/uwing.f5
cd ML-view/
~/git/tacs/extern/f52vtk/f52vtk_element uwing.f5
cd ../
python make_exploded.py --useML True --exploded 2
mv capsStruct1_0/Scratch/tacs/tacs_output_file000.f5 ML-view/intstruct.f5
cd ML-view/
~/git/tacs/extern/f52vtk/f52vtk_element 52vtk_element intstruct.f5
cd ../
python make_exploded.py --useML True --exploded 3
mv capsStruct1_0/Scratch/tacs/tacs_output_file000.f5 ML-view/lwing.f5
cd ML-view/
~/git/tacs/extern/f52vtk/f52vtk_element 52vtk_element lwing.f5
cd ../


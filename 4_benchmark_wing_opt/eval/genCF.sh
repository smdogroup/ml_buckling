rm -rf capsStruct*
python make_exploded.py --exploded 1
mv capsStruct1_0/Scratch/tacs/tacs_output_file000.f5 CF-view/uwing.f5
cd CF-view/
~/git/tacs/extern/f5tovtk/f5tovtk_element uwing.f5
cd ../
python make_exploded.py --exploded 2
mv capsStruct1_0/Scratch/tacs/tacs_output_file000.f5 CF-view/intstruct.f5
cd CF-view/
~/git/tacs/extern/f5tovtk/f5tovtk_element intstruct.f5
cd ../
python make_exploded.py --exploded 3
mv capsStruct1_0/Scratch/tacs/tacs_output_file000.f5 CF-view/lwing.f5
cd CF-view/
~/git/tacs/extern/f5tovtk/f5tovtk_element lwing.f5
cd ../


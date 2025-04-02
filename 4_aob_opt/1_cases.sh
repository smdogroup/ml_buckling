
mpirun -n 4 python 1_oneway_composite_sizing.py --no-useML
mpirun -n 4 python 1_oneway_composite_sizing.py --useML

# mpirun -n 4 python 1_oneway_metal_sizing.py --no-useML
# mv SNOPT_summary.out design/CF-metal-SNOPT_summary.out
# mpirun -n 4 python 1_oneway_metal_sizing.py --useML
# mv SNOPT_summary.out design/ML-metal-SNOPT_summary.out
3026624.6158990553
Mode type predicted as global
        eig_CF=24.54451594725003
        eig_FEA=None
Traceback (most recent call last):
  File "/home/seanfireball1/git/_archive/ml_buckling/2_stiffened_panels/1_gen_data.py", line 119, in <module>
    get_buckling_load(
  File "/home/seanfireball1/git/_archive/ml_buckling/2_stiffened_panels/src/buckling_analysis.py", line 268, in get_buckling_load
    "phi" : stiff_analysis.min_global_eigmode,
  File "/home/seanfireball1/git/_archive/ml_buckling/ml_buckling/stiffened_plate_analysis.py", line 1748, in min_global_eigmode
    def min_global_eigmode(self) -> np.ndarray:
  File "/home/seanfireball1/git/_archive/ml_buckling/ml_buckling/stiffened_plate_analysis.py", line 1744, in min_global_mode_index
gamma=1.1052715147470042 rho0=0.24836986385685916 num_stiff=9 ply_angle=30.259956990100775 composite_material.__name__='torayBT250E' plate_slenderness=38.659330389079834
gamma=1.1052715147470042 rho0=0.24836986385685916 num_stiff=9 ply_angle=30.259956990100775 composite_material.__name__='torayBT250E' plate_slenderness=38.659330389079834
gamma=1.1052715147470042 rho0=0.24836986385685916 num_stiff=9 ply_angle=30.259956990100775 composite_material.__name__='torayBT250E' plate_slenderness=38.659330389079834
    def min_global_mode_index(self) -> int:
AttributeError: 'StiffenedPlateAnalysis' object has no attribute '_min_global_imode'
guess_x0=(-0.052631578947368585, -0.10526315789473695) resid_norm=2.0392570489572828
guess_x0=(-0.052631578947368585, -0.10526315789473695) resid_norm=2.0392570489572828
guess_x0=(-0.052631578947368585, -0.10526315789473695) resid_norm=2.0392570489572828
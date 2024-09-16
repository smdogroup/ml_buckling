import numpy as np
import ml_buckling as mlb
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD

df = pd.read_csv("raw_data/stiffener_crippling.csv")

# for each data point in the raw data add a new parameter
# zeta = A66/A11 * (b/h)^2
xi = df["xi"].to_numpy()
gen_eps = df["gen_eps"].to_numpy()
rho_0 = df["a0/b0"].to_numpy()
SR = df["b/h"].to_numpy()
materials = df["material"].to_numpy()
zeta = np.zeros((materials.shape[0],))
ply_angles = df["ply_angle"].to_numpy()
kmin = df["kmin"].to_numpy()

for i in range(materials.shape[0]):
    material_name = materials[i]
    ply_angle = ply_angles[i]
    h = 1.0
    b = h * SR[i]
    AR = 1.0  # doesn't affect zeta..
    a = b * AR
    material = mlb.UnstiffenedPlateAnalysis.get_material_from_str(material_name)
    new_plate: mlb.UnstiffenedPlateAnalysis = material(
        comm,
        bdf_file="plate.bdf",
        a=a,
        b=b,
        h=h,
        ply_angle=ply_angle,
    )

    zeta[i] = new_plate.zeta

log_xi = np.log(1.0 + xi)
log_rho0 = np.log(rho_0)
log_eps = np.log(1.0 + gen_eps)
log_zeta = np.log(1.0 + 1000.0 * zeta)
log_kmin = np.log(kmin)

data_dict = {
    "x0": list(log_xi),
    "x1": list(log_rho0),
    "x2": list(log_eps),
    "x3": list(log_zeta),
    "y": list(log_kmin),
}

new_df = pd.DataFrame(data_dict)
new_df.to_csv("data/stiffener_crippling.csv")

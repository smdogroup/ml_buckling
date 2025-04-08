import numpy as np

# output from 5 deg AOA pullup, -3 deg AOA pushdown
# -------------------------------------------------
# func.full_name='pullup-cl' func.value.real=143.9502133471505
# func.full_name='pullup-cd' func.value.real=10.401864856191292
# func.full_name='pushdown-cl' func.value.real=-37.84674949615323
# func.full_name='pushdown-cd' func.value.real=13.71194800479088

# compute the full wing area (both sides)
c1 = 48.0 # m
c2 = 40.4 
c3 = 20.7
c4 = 4.15
ybar1 = 0.11
ybar2 = 0.30
ybar3 = 1.0 - ybar1 - ybar2

aspect = 3.2
temp = (c1 + c2) * ybar1 + (c2 + c3) * ybar2 + (c3 + c4) * ybar3
area = temp**2 * aspect / 16
span = (area * aspect)**0.5
print(f'{area=}')
print(f"{span=}")

# from above
pullup_CL = 143.9502 # ND
pushdown_CL = -37.8467 # ND
pullup_CL /= area
pushdown_CL /= area

# ran at wrong qinf, Tinf by accident
# qinf_factor = 2.2657e4 /3.1682e4
# Tinf_factor = 300 / 216
# corr_factor = qinf_factor * Tinf_factor
# print(f'{corr_factor=}') # equals almost exactly 1.0, so ignore it
# # because density propto 1/T

# above is full wing area

# compute the AOA and AOA increase factors to achieve 2.5g pullup and -1g pushdown
TOGM = 3.4e5 # kg, from HSCT study
TOGW = TOGM * 9.8 # to N
qinf = 2.2657e4 # at mach 0.4 sea level
qinf /= 2.0 # correction to 1/2 * rhoinf * vinf^2

pullup_lift = TOGW * 2.5 # 2.5g pullup
pushdown_lift = TOGW * -1.0 # -1g pushdown

# NACA 64A012 airfoil with CL  =0.2 and pressure recovery at 80% the chord
zero_lift_AOA = -1.5

# half area?
area /= 2.0

pullup_des_CL = pullup_lift / qinf / area
pushdown_des_CL = pushdown_lift / qinf / area
print(f"{pullup_des_CL=} {pushdown_des_CL=}")
print(f"{pullup_CL=} {pushdown_CL=}")

aoa_adjust_factors = np.array([pullup_des_CL, pushdown_des_CL]) / \
    np.array([pullup_CL, pushdown_CL])
orig_AOA = np.array([5.0, -3.0])
new_AOA = (orig_AOA - zero_lift_AOA) * aoa_adjust_factors + zero_lift_AOA
print(f"{aoa_adjust_factors=}")
print(f"{orig_AOA=}")
print(f"{new_AOA=}")

print(f"{aoa_adjust_factors=}")
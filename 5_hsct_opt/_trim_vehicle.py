import numpy as np

# output from 5 deg AOA pullup, -3 deg AOA pushdown
# -------------------------------------------------

# func.full_name='pullup-cl' func.value.real=143.9502133471505
# func.full_name='pullup-cd' func.value.real=10.401864856191292
# func.full_name='pushdown-cl' func.value.real=-37.84674949615323
# func.full_name='pushdown-cd' func.value.real=13.71194800479088

# calculate wing area
# -------------------

# at each station
chords = np.array([48.0, 40.4, 20.7, 4.15])
half_span = 22.44
sec_spans = half_span * np.array([0.11, 0.30, 1.0 - 0.41])
print(f"{sec_spans=}")
half_area = sum([0.5 * (chords[i] + chords[i+1]) * sec_spans[i] for i in range(3)])
print(f"{half_area=}")
area = half_area * 2.0
print(f"{area=}")


# uses half area here since half wing and no ref area in FUN3D
pullup_CL = 143.9502 # ND
pushdown_CL = -37.8467 # ND
pullup_CL /= half_area
pushdown_CL /= half_area

print(f"{pullup_CL=:.5e} {pushdown_CL=:.5e}")

# compute the AOA and AOA increase factors to achieve 2.5g pullup and -1g pushdown
TOGM = 3.4e5 # kg, from HSCT study
TOGW = TOGM * 9.8 # kg to N
rho = 1.225 # kg/m^3
ainf = 331 # m/s
M = 0.4
Vinf = ainf * M
qinf = 0.5 * rho * Vinf**2
print(f'{qinf=}')

pullup_lift = TOGW * 2.5 # 2.5g pullup
pushdown_lift = TOGW * -1.0 # -1g pushdown

# NACA 64A012 airfoil with CL  =0.2 and pressure recovery at 80% the chord
zero_lift_AOA = -1.5
# zero_lift_AOA = 0.0

# half area?
#area /= 2.0

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


# wing structural mas to be about 51-61000 kg
# taken as 15-18% realistic range for large delta wings
"""
# concorde: 185,000 kg TOGM, wing area 358 m^2, 26000 kg wing mass, 14% TOGM
# B-70 Valkryie : 250,000 kg TOGM, wing area 600 m^2, 35-40,000 kg wing mass, 14-16% TOGM
My aircraft is bigger, larger delta wing, may have wingbox mass about 16% to 18% TOGM
"""

est_wingbox_mass = np.array([0.16, 0.18]) * TOGM
est_half_wingbox_mass = est_wingbox_mass / 2.0
print(f"{est_half_wingbox_mass}")

# very thin wings 3-4% t/c may be significantly heavier
# thicker wings t/c > 5% will probably weight around 41-47,000 kg

"""
some things that may make my wing heavier:
- I have confidence in the wing area and CFD loads now, estimated 16 deg AOA pullup, -7.6 deg AOA pushdown
- titanium denser than aluminum => often heavier (good at high temp though)
- 2.5g pullup and 1g pushdown is limiting case (fairly high)
- maybe need thicker t/c wing to reduce mass :
    t/c is 0.025 at root, t/c 0.095 at tip
    so t/c is within reasonable ranges and thinner at root is normal
- using 1.5 safety factor instead of 1.2-1.3 for buckling typically (some effect)
- check FE mesh => if mesh is underresolved, may result in significantly heavier wing..

Conclusion : most of setup is reasonable => I suspect the coarse FE mesh with initial stress concentration
    is leading to a heavier wing. 
    - initial mesh was 45k elements, 23k nodes and had huge ks=149 stress concentration at root in initial design
    - Generate a larger # element CFD mesh, and just lower optimality and feasibility requirements maybe for faster runtime
"""
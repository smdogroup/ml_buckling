__all__  = ["compute_stiffened_ABD", "gen_mesh", "rotated_ply_effective_props"]

import numpy as np
import ml_buckling as mlb

def compute_stiffened_ABD(
    E1, E2, G12, nu12,
    ply_angles_deg,
    ply_fractions,
    plate_thickness,
    h_w,     # stiffener height
    t_w,      # stiffener thickness (width)
    nstiff=1,
):

    if nstiff == 0:
        h_w *= 0.0
        t_w *= 0.0

    def get_Q(E1, E2, G12, nu12):
        nu21 = (E2 / E1) * nu12
        denom = 1 - nu12 * nu21
        Q11 = E1 / denom
        Q22 = E2 / denom
        Q12 = nu12 * E2 / denom
        Q66 = G12
        return np.array([
            [Q11, Q12, 0],
            [Q12, Q22, 0],
            [0,   0,   Q66]
        ])

    def transform_Q(Q, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        c2 = c * c
        s2 = s * s
        cs = c * s

        T = np.array([
            [c2, s2, 2 * cs],
            [s2, c2, -2 * cs],
            [-cs, cs, c2 - s2]
        ])
        T_inv = np.linalg.inv(T)
        Qbar = T_inv @ Q @ T_inv.T
        return Qbar

    def integrate_ABD(z_bot, z_top, Qbar):
        dz = z_top - z_bot
        A = Qbar * dz
        B = 0.5 * Qbar * (z_top**2 - z_bot**2)
        D = (1/3) * Qbar * (z_top**3 - z_bot**3)
        return A, B, D

    # Normalize ply fractions to ensure they sum to 1
    ply_fractions = np.array(ply_fractions, dtype=float)
    ply_fractions /= np.sum(ply_fractions)

    # Compute per-ply thickness from total plate thickness
    ply_thicknesses = ply_fractions * plate_thickness

    # Q = get_Q(E1, E2, G12, nu12)
    # Qbars = [transform_Q(Q, theta) for theta in ply_angles]

    # actually need to use the Qbar from rotated properties of each ply
    Qbars = []
    for ply_angle_deg in ply_angles_deg:
        util = mlb.CompositeMaterialUtility(E1, E2, nu12, G12).rotate_ply(ply_angle_deg)
        E1_rot, E2_rot, G12_rot, nu12_rot = util.E11, util.E22, util.G12, util.nu12
        Qbars += [get_Q(E1_rot, E2_rot, G12_rot, nu12_rot)]

    # Compute centroid z from bottom of panel
    A_plate = 1.0 * plate_thickness
    A_stiff = t_w * h_w
    A_total = A_plate + A_stiff
    z_plate = 0.0
    z_stiff = plate_thickness / 2 + h_w / 2
    z_centroid = (A_plate * z_plate + A_stiff * z_stiff) / A_total

    # print(f"{z_centroid=}")

    # Initialize A, B, D matrices
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    # First do A11 and D11 with shifted z (centroid shift applied)
    A11_shifted = 0.0
    D11_shifted = 0.0

    z0_shifted = -plate_thickness/2.0 + z_centroid
    for t, Qbar in zip(ply_thicknesses, Qbars):
        z1_shifted = z0_shifted + t
        Ai, Bi, Di = integrate_ABD(z0_shifted, z1_shifted, Qbar)
        A11_shifted += Ai[0, 0]
        D11_shifted += Di[0, 0]
        z0_shifted = z1_shifted

    # Then integrate the rest of the ABD without shifting (normal coordinates)
    z0 = -plate_thickness/2.0
    for t, Qbar in zip(ply_thicknesses, Qbars):
        z1 = z0 + t
        Ai, Bi, Di = integrate_ABD(z0, z1, Qbar)

        # print(f"{t=} {Di=}")

        A += Ai
        B += Bi
        D += Di
        z0 = z1

    # Overwrite shifted A11 and D11 into plate ABD
    A[0, 0] = A11_shifted
    D[0, 0] = D11_shifted

    # Assemble ABD matrix (B shifted to 0 if symmetric)
    ABD = np.block([
        [A, np.zeros((3, 3))],
        [np.zeros((3, 3)), D]
    ])

    # also compute the overall E1s and Is from Qbar
    Qbar_ovr = np.zeros((3,3))
    for frac, Qbar in zip(ply_fractions, Qbars):
        Qbar_ovr += Qbar * frac
    E1s = Qbar_ovr[0,0] - Qbar[0,1]**2 / Qbar[-1,-1]
    Is_0 = t_w * h_w**3 / 12.0
    z_stiff = plate_thickness / 2.0 + h_w / 2.0
    Is = Is_0 + A_stiff * (z_centroid - z_stiff)**2

    return ABD, z_centroid, E1s, Is

def gen_mesh(comm, a, b, h, h_w, t_w, E1, E2, nu12, G, nstiff=1, nelems=2000, theta=0.0, strain_mult:float=1.0, 
             strain_vec=np.array([1,0,0]), can_print:bool=False):
    # ------------
    AR = a / b
    c_material = mlb.CompositeMaterial(
        E11=E1, E22=E2, G12=G, nu12=nu12,
        ply_angles=[theta],
        ply_fractions=[1.0],
        ref_axis=[1,0,0],
    )
    # --------------

    if nstiff == 0:
        h_w = 1e-8
        t_w = 1e-8

    geometry = mlb.StiffenedPlateGeometry(
        a=a, b=b, h=h, num_stiff=nstiff, h_w=h_w, t_w=t_w
    )
    stiff_analysis = mlb.StiffenedPlateAnalysis(
        comm=comm, geometry=geometry,
        stiffener_material=c_material,
        plate_material=c_material,
    )

    _nelems = nelems
    MIN_Y = 5
    MIN_Z = 5  # 5
    N = geometry.num_local
    AR_s = geometry.a / geometry.h_w
    nx = np.ceil(np.sqrt(_nelems / (1.0 / AR + (N - 1) / AR_s)))
    den = 1.0 / AR + (N - 1) * 1.0 / AR_s
    ny = max([np.ceil(nx / AR / N), MIN_Y])
    nz = 3

    # strain_vec = stiff_analysis.affine_exx * strain_mult * strain_vec
    # # print(f"{strain_vec=}")
    # # exit()

    stiff_analysis.pre_analysis(
        nx_plate=int(nx),
        ny_plate=int(ny),
        nz_stiff=int(nz),
        nx_stiff_mult=2,
        exx=stiff_analysis.affine_exx,
        exy=0.0,
        eyy=0.0,
        clamped=False,
        _make_rbe=False,
        _explicit_poisson_exp=True,
        # all_explicit_strains=True, # new setting for multi ply case (we specify all strains..)
    )
    comm.Barrier()

    if can_print: print(F"{stiff_analysis}")
    # exit()

    return stiff_analysis

def rotated_ply_effective_props(E1, E2, G12, nu12, theta_rad):
    """
    Compute effective in-plane properties for a rotated ply, with axial-shear coupling terms removed.
    
    Parameters:
    E1, E2: in-plane Young's moduli in ply coords
    G12: in-plane shear modulus
    nu12: major Poisson's ratio
    theta_rad: ply angle in radians

    Returns:
    E1_eff, E2_eff, G12_eff, nu12_eff: effective orthotropic properties with no axial-shear coupling
    """
    # print(f"{theta_rad=}")

    C = np.cos(theta_rad)
    S = np.sin(theta_rad)
    C2 = np.cos(2 * theta_rad)
    S2 = np.sin(2 * theta_rad)
    E1_rot = (
        C ** 4 / E1 + S ** 4 / E2 + 0.25 * (1.0 / G12 - 2 * nu12 / E1) * S2 ** 2
    ) ** (-1)
    E2_rot = (
        S ** 4 / E1 + C ** 4 / E2 + 0.25 * (1.0 / G12 - 2 * nu12 / E1) * S2 ** 2
    ) ** (-1)
    _temp1 = 1.0 / E1 + 2.0 * nu12 / E1 + 1.0 / E2
    G12_rot = (_temp1 - (_temp1 - 1.0 / G12) * C2 ** 2) ** (-1)
    nu12_rot = E1 * (nu12 / E1 - 0.25 * (_temp1 - 1.0 / G12) * S2 ** 2)

    return E1_rot, E2_rot, G12_rot, nu12_rot


if __name__ == "__main__":
    E1 = 117.9e9
    E2 = 9.7e9
    G12 = 4.8e9
    nu12 = 0.35

    ply_angles = np.deg2rad([0.0, 90.0, 90.0, 0.0])
    ply_fractions = [0.25, 0.25, 0.25, 0.25]  # must sum to 1
    plate_thickness = 0.01  # m
    h_w = 0.02
    t_w = h_w / 20.0

    ABD, z_c, E1s, Is = compute_stiffened_ABD(
        E1, E2, G12, nu12,
        ply_angles,
        ply_fractions,
        plate_thickness,
        h_w, t_w
    )

    # import matplotlib.pyplot as plt
    # plt.imshow(np.log(ABD))
    # plt.show()

    # compare this to an online ABD matrix calculator..
    # results here:
    # A11 = 6.451e8, A12 = 3.432e7
    # A22 = 6.451e8, A66 = 4.804e7
    # D11, D12 = 8.859e3, 2.895e2
    # D22, D66 = 2.022e3, 4.052e2
    # and all 16,26 entries zero and sym
    # also B = 0 due to centroid shift zc = 0.00503
    # this does match the ABD matrix of the panel on online ABD matrix calculator, but doesn't seem to shift for z_c?

    print("Centroid z from bottom of panel:", z_c)
    print("ABD matrix (shifted so B = 0):\n", ABD)

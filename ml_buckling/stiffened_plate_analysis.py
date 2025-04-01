__all__ = ["StiffenedPlateAnalysis"]

import numpy as np
from tacs import pyTACS, constitutive, elements, utilities, caps2tacs, TACS
import os
from pprint import pprint
from .stiffened_plate_geometry import StiffenedPlateGeometry
from .composite_material import CompositeMaterial
from .composite_material_utility import CompositeMaterialUtility

# from typing_extensions import Self
from scipy.optimize import fsolve

dtype = utilities.BaseUI.dtype


def exp_kernel1(xp, xq, sigma_f, L):
    # xp, xq are Nx1, Nx1 vectors
    return sigma_f ** 2 * np.exp(-0.5 * (xp - xq).T @ (xp - xq) / L ** 2)


class StiffenedPlateAnalysis:
    def __init__(
        self,
        comm,
        geometry: StiffenedPlateGeometry,
        plate_material: CompositeMaterial,
        stiffener_material: CompositeMaterial,
        name=None,  # use the plate name to differentiate plate folder names
        _compress_stiff=False,  # whether to compress stiffeners to in axial case (TODO : figure this out => need to study static analysis)
    ):
        self.comm = comm
        self.geometry = geometry
        self.plate_material = plate_material
        self.stiffener_material = stiffener_material

        self._compress_stiff_override = _compress_stiff

        # geometry properties
        self._name = name
        self._tacs_aim = None
        self._index = 0
        self._saved_alphas = False

        # save the input strains
        self._exx = None
        self._exy = None
        self._eyy = None

        self._MAC_msg = "MAC not performed.."

    @classmethod
    def copy(cls, analysis, name=None):
        return cls(
            comm=analysis.comm,
            geometry=analysis.geometry,
            plate_material=analysis.plate_material,
            stiffener_material=analysis.stiffener_material,
            name=analysis._name if name is None else name,
            _compress_stiff=analysis._compress_stiff_override,
        )

    @property
    def buckling_folder_name(self) -> str:
        if self._name:
            return "_buckling-" + self._name
        else:
            return "_buckling"

    @property
    def static_folder_name(self) -> str:
        if self._name:
            return "_static-" + self._name
        else:
            return "_static"

    @property
    def csm_file(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir_path, "_stiffened_panel.csm")

    @property
    def caps_lock(self):
        if self._tacs_aim is not None:
            tacs_dir = self._tacs_aim.root_analysis_dir
            scratch_dir = os.path.dirname(tacs_dir)
            # caps_dir = os.path.dirname(scratch_dir)
            return os.path.join(scratch_dir, "capsLock")

    @property
    def dat_file(self):
        if self._use_caps:
            return self._tacs_aim.root_dat_file
        else:
            return self.bdf_file

    @property
    def bdf_file(self):
        if self._use_caps:
            tacs_dir = self._tacs_aim.root_analysis_dir
            return os.path.join(tacs_dir, "tacs.bdf")
        else:
            cwd = os.getcwd()
            return os.path.join(cwd, "_stiffened_panel.bdf")

    @property
    def Darray_stiff(self) -> float:
        """array [D11,D12,D22,D66] for the stiffener"""
        zL = -self.geometry.t_w / 2.0  # symmetric about 0
        _Darray = np.zeros((4,))
        ply_thicknesses = self.stiffener_material.get_ply_thicknesses(self.geometry.t_w)
        for iply, ply_angle in enumerate(self.stiffener_material.ply_angles):
            ply_thick = ply_thicknesses[iply]
            zU = zL + ply_thick
            util = CompositeMaterialUtility(
                E11=self.stiffener_material.E11,
                E22=self.stiffener_material.E22,
                nu12=self.stiffener_material.nu12,
                G12=self.stiffener_material.G12,
            ).rotate_ply(ply_angle)

            nu_denom = 1 - util.nu12 * util.nu21
            Q11 = util.E11 / nu_denom
            Q22 = util.E22 / nu_denom
            Q12 = util.nu12 * Q22
            Q66 = util.G12

            _Darray[0] += 1.0 / 3 * Q11 * (zU ** 3 - zL ** 3)
            _Darray[1] += 1.0 / 3 * Q12 * (zU ** 3 - zL ** 3)
            _Darray[2] += 1.0 / 3 * Q22 * (zU ** 3 - zL ** 3)
            _Darray[3] += 1.0 / 3 * Q66 * (zU ** 3 - zL ** 3)

            zL = zU * 1.0
        return _Darray

    @property
    def xi_stiff(self):
        _Darray = self.Darray_stiff
        D11 = _Darray[0]
        D12 = _Darray[1]
        D22 = _Darray[2]
        D66 = _Darray[3]
        return (D12 + 2 * D66) / np.sqrt(D11 * D22)

    @property
    def gen_poisson_stiff(self):
        _Darray = self.Darray_stiff
        D11 = _Darray[0]
        D12 = _Darray[1]
        D22 = _Darray[2]
        return 1 / self.xi_stiff * D12 / np.sqrt(D11 * D22)

    @property
    def old_Darray_plate(self) -> float:
        """array [D11,D12,D22,D66] for the stiffener"""
        zL = -self.geometry.h / 2.0
        _Darray = np.zeros((4,))
        ply_thicknesses = self.plate_material.get_ply_thicknesses(self.geometry.h)
        for iply, ply_angle in enumerate(self.plate_material.ply_angles):
            ply_thick = ply_thicknesses[iply]
            zU = zL + ply_thick
            util = CompositeMaterialUtility(
                E11=self.plate_material.E11,
                E22=self.plate_material.E22,
                nu12=self.plate_material.nu12,
                G12=self.plate_material.G12,
            ).rotate_ply(ply_angle)

            nu_denom = 1 - util.nu12 * util.nu21
            Q11 = util.E11 / nu_denom
            Q22 = util.E22 / nu_denom
            Q12 = util.nu12 * Q22
            Q66 = util.G12

            _Darray[0] += 1.0 / 3 * Q11 * (zU ** 3 - zL ** 3)
            _Darray[1] += 1.0 / 3 * Q12 * (zU ** 3 - zL ** 3)
            _Darray[2] += 1.0 / 3 * Q22 * (zU ** 3 - zL ** 3)
            _Darray[3] += 1.0 / 3 * Q66 * (zU ** 3 - zL ** 3)

            zL = zU * 1.0
        return _Darray

    @property
    def Darray_plate(self) -> float:
        """array [D11,D12,D22,D66] for the stiffener"""
        # first compute D22,D12,D66 with centroid at center of skin
        zL = -self.geometry.h / 2.0
        # zL = -self.geometry.h / 2.0 - self.centroid # symmetric about 0
        _Darray = np.zeros((4,))
        ply_thicknesses = self.plate_material.get_ply_thicknesses(self.geometry.h)
        for iply, ply_angle in enumerate(self.plate_material.ply_angles):
            ply_thick = ply_thicknesses[iply]
            zU = zL + ply_thick
            util = CompositeMaterialUtility(
                E11=self.plate_material.E11,
                E22=self.plate_material.E22,
                nu12=self.plate_material.nu12,
                G12=self.plate_material.G12,
            ).rotate_ply(ply_angle)

            nu_denom = 1 - util.nu12 * util.nu21
            # Q11 = util.E11 / nu_denom
            Q22 = util.E22 / nu_denom
            Q12 = util.nu12 * Q22
            Q66 = util.G12

            _Darray[1] += 1.0 / 3 * Q12 * (zU ** 3 - zL ** 3)
            _Darray[2] += 1.0 / 3 * Q22 * (zU ** 3 - zL ** 3)
            _Darray[3] += 1.0 / 3 * Q66 * (zU ** 3 - zL ** 3)

            zL = zU * 1.0

        # then compute D11 with overall centroid
        # zL = -self.geometry.h / 2.0
        zL = -self.geometry.h / 2.0 - self.centroid  # symmetric about 0
        for iply, ply_angle in enumerate(self.plate_material.ply_angles):
            ply_thick = ply_thicknesses[iply]
            zU = zL + ply_thick
            util = CompositeMaterialUtility(
                E11=self.plate_material.E11,
                E22=self.plate_material.E22,
                nu12=self.plate_material.nu12,
                G12=self.plate_material.G12,
            ).rotate_ply(ply_angle)

            nu_denom = 1 - util.nu12 * util.nu21
            Q11 = util.E11 / nu_denom
            _Darray[0] += 1.0 / 3 * Q11 * (zU ** 3 - zL ** 3)

            # # do D22 with overall centroid also? prob not
            # Q22 = util.E22 / nu_denom
            # _Darray[1] += 1.0 / 3 * Q12 * (zU ** 3 - zL ** 3)

            zL = zU * 1.0
        return _Darray

    @property
    def Aarray_plate(self) -> float:
        """array [A11,A12,A22,A66] for the plate"""
        zL = -self.geometry.h / 2.0  # symmetric about 0
        _Aarray = np.zeros((4,))
        ply_thicknesses = self.plate_material.get_ply_thicknesses(self.geometry.h)
        for iply, ply_angle in enumerate(self.plate_material.ply_angles):
            ply_thick = ply_thicknesses[iply]
            zU = zL + ply_thick
            util = CompositeMaterialUtility(
                E11=self.plate_material.E11,
                E22=self.plate_material.E22,
                nu12=self.plate_material.nu12,
                G12=self.plate_material.G12,
            ).rotate_ply(ply_angle)

            nu_denom = 1 - util.nu12 * util.nu21
            Q11 = util.E11 / nu_denom
            Q22 = util.E22 / nu_denom
            Q12 = util.nu12 * Q22
            Q66 = util.G12

            _Aarray[0] += Q11 * (zU - zL)
            _Aarray[1] += Q12 * (zU - zL)
            _Aarray[2] += Q22 * (zU - zL)
            _Aarray[3] += Q66 * (zU - zL)

            zL = zU * 1.0
        return _Aarray

    @property
    def Aarray_stiff(self) -> float:
        """array [A11,A12,A22,A66] for the stiffener"""
        zL = -self.geometry.t_w / 2.0  # symmetric about 0
        _Aarray = np.zeros((4,))
        ply_thicknesses = self.stiffener_material.get_ply_thicknesses(self.geometry.t_w)
        for iply, ply_angle in enumerate(self.stiffener_material.ply_angles):
            ply_thick = ply_thicknesses[iply]
            zU = zL + ply_thick
            util = CompositeMaterialUtility(
                E11=self.stiffener_material.E11,
                E22=self.stiffener_material.E22,
                nu12=self.stiffener_material.nu12,
                G12=self.stiffener_material.G12,
            ).rotate_ply(ply_angle)

            nu_denom = 1 - util.nu12 * util.nu21
            Q11 = util.E11 / nu_denom
            Q22 = util.E22 / nu_denom
            Q12 = util.nu12 * Q22
            Q66 = util.G12

            _Aarray[0] += Q11 * (zU - zL)
            _Aarray[1] += Q12 * (zU - zL)
            _Aarray[2] += Q22 * (zU - zL)
            _Aarray[3] += Q66 * (zU - zL)

            zL = zU * 1.0
        return _Aarray

    @property
    def A11_eff(self) -> float:
        Aarray = self.Aarray_plate
        A11 = Aarray[0]
        A12 = Aarray[1]
        A22 = Aarray[2]
        # A11prime entry in compliance matrix where A16, A26 are zero and B matrix = 0 so A,D decoupled
        return A11 - A12 ** 2 / A22

    @property
    def A12_eff(self) -> float:
        Aarray = self.Aarray_plate
        A11 = Aarray[0]
        A12 = Aarray[1]
        A22 = Aarray[2]
        # A11prime entry in compliance matrix where A16, A26 are zero and B matrix = 0 so A,D decoupled
        return A12 - A11 * A22 / A12

    @property
    def old_xi_plate(self):
        _Darray = self.Darray_plate
        D11 = _Darray[0]
        D12 = _Darray[1]
        D22 = _Darray[2]
        D66 = _Darray[3]
        return (D12 + 2 * D66) / np.sqrt(D11 * D22)

    @property
    def xi_plate(self):
        _Darray = self.Darray_plate
        D11 = _Darray[0]
        D12 = _Darray[1]
        D22 = _Darray[2]
        D66 = _Darray[3]
        return (D12 + 2 * D66) / np.sqrt(D11 * D22)

    @property
    def old_affine_aspect_ratio(self):
        _Darray = self.old_Darray_plate
        D11 = _Darray[0]
        D22 = _Darray[2]
        return self.geometry.a / self.geometry.b * (D22 / D11) ** 0.25

    @property
    def affine_aspect_ratio(self):
        _Darray = self.Darray_plate
        D11 = _Darray[0]
        D22 = _Darray[2]
        return self.geometry.a / self.geometry.b * (D22 / D11) ** 0.25

    @property
    def delta(self) -> float:
        """area ratio parameter extended to N stiffener case"""
        if self.geometry.num_stiff == 0:
            return 0.0
        return (
            self.stiffener_material.E_eff  # E_eff for composite laminate case (multi-ply)
            * self.geometry.area_S
            / (self.plate_material.E_eff * self.geometry.s_p * self.geometry.h)
        )

    @property
    def zeta_plate(self) -> float:
        """compute the transverse shear ratio for the plate"""
        _Aarray = self.Aarray_plate
        A11 = _Aarray[0]
        A66 = _Aarray[3]
        old_zeta = A66 / A11 * (self.geometry.b / self.geometry.h) ** 2
        return 1.0 / old_zeta

    @property
    def zeta_stiff(self) -> float:
        """compute the transverse shear ratio for the stiffener"""
        _Aarray = self.Aarray_stiff
        A11 = _Aarray[0]
        A66 = _Aarray[3]
        old_zeta = A66 / A11 * (self.geometry.h_w / self.geometry.t_w) ** 2
        return 1.0 / old_zeta

    @property
    def old_affine_exx(self):
        _Darray = self.old_Darray_plate
        D11 = _Darray[0]
        D22 = _Darray[2]
        _Aarray = self.Aarray_plate
        A11 = _Aarray[0]
        exx_T = (
            np.pi ** 2
            * np.sqrt(D11 * D22)
            / self.geometry.b ** 2
            / (1 + self.delta)
            / A11
        )
        # N11 = exx_T * A11
        # print(f"{N11=}")
        # print(f"{A11=}"); exit()
        return exx_T

    @property
    def affine_exx(self):
        """
        Solve exx such that lambda = lambda_min*
        Just estimate the overall buckling mode and require the user to adjust for local modes case
            when the stiffeners are stronger
        TODO : could estimate based on local mode too?
        """
        _Darray = self.Darray_plate
        D11 = _Darray[0]
        D22 = _Darray[2]
        exx_T = (
            np.pi ** 2
            * np.sqrt(D11 * D22)
            / self.geometry.b ** 2
            / (1 + self.delta)
            / self.A11_eff
        )
        return exx_T

    @property
    def intended_Nxx(self) -> float:
        """
        intended Nxx in linear static analysis
        """
        N11 = self.affine_exx * self.A11_eff
        # print(f"{N11=}")
        # print(f"{self.A11_eff=}")
        return N11

    @property
    def centroid(self) -> float:
        E_S = self.stiffener_material.E_eff
        A_W = self.geometry.area_w
        A_B = self.geometry.area_b
        tb = self.geometry.t_b
        hw = self.geometry.h_w
        h = self.geometry.h
        A_S = self.geometry.area_S
        E_P = self.plate_material.E_eff
        A_P = self.geometry.area_P
        num_stiff = self.geometry.num_stiff

        # modulus weighted centroid zcen
        _z_base = (tb + h) / 2.0
        _z_wall = (hw + h) / 2.0
        z_cen = (
            E_S
            * (A_B * _z_base + A_W * _z_wall)
            * num_stiff
            / (E_S * A_S * num_stiff + E_P * A_P)
        )
        # print(f"{z_cen=}")
        if isinstance(z_cen, np.ndarray):
            z_cen = z_cen[0]
        z_cen = float(z_cen)
        return z_cen

    @property
    def old_gamma(self) -> float:
        """stiffener to plate bending stiffness ratio"""
        if self.geometry.num_stiff == 0:
            return 0.0
        # get the stiffener bending stiffeness EI about modulus weighted centroid
        E_S = self.stiffener_material.E_eff
        A_W = self.geometry.area_w
        A_B = self.geometry.area_b
        wb = self.geometry.w_b
        tb = self.geometry.t_b
        # print(f"tb = {tb}")
        tw = self.geometry.t_w
        hw = self.geometry.h_w
        h = self.geometry.h
        A_S = self.geometry.area_S
        E_P = self.plate_material.E_eff
        A_P = self.geometry.area_P
        num_stiff = self.geometry.num_stiff
        _Darray = self.old_Darray_plate
        D11 = _Darray[0]

        # modulus weighted centroid zcen
        _z_base = (tb + h) / 2.0
        _z_wall = (hw + h) / 2.0
        z_cen = (
            E_S
            * (A_B * _z_base + A_W * _z_wall)
            * num_stiff
            / (E_S * A_S * num_stiff + E_P * A_P)
        )
        z_s = (
            E_S * (A_B * _z_base + A_W * _z_wall) / (E_S * A_S)
        )  # w/o base this is just h/2
        I_S = (wb ** 3 * tb + tw * hw ** 3) / 12.0
        EI_s = (
            E_S * I_S + E_S * A_S * (z_s - z_cen) ** 2
        )  # dominant term should be (z_s - z_cen)^2 due to offset center
        return (
            EI_s / self.geometry.s_p / D11
        )  # TODO : temporarily multiply by n_stiff+1?

    @property
    def gamma(self) -> float:
        """stiffener to plate bending stiffness ratio"""
        if self.geometry.num_stiff == 0:
            return 0.0
        # get the stiffener bending stiffeness EI about modulus weighted centroid
        E_S = self.stiffener_material.E_eff
        A_W = self.geometry.area_w
        A_B = self.geometry.area_b
        wb = self.geometry.w_b
        tb = self.geometry.t_b
        # print(f"tb = {tb}")
        tw = self.geometry.t_w
        hw = self.geometry.h_w
        h = self.geometry.h
        A_S = self.geometry.area_S
        E_P = self.plate_material.E_eff
        A_P = self.geometry.area_P
        num_stiff = self.geometry.num_stiff
        _Darray = self.Darray_plate
        D11 = _Darray[0]

        # modulus weighted centroid zcen
        _z_base = (tb + h) / 2.0
        _z_wall = (hw + h) / 2.0
        z_cen = (
            E_S
            * (A_B * _z_base + A_W * _z_wall)
            * num_stiff
            / (E_S * A_S * num_stiff + E_P * A_P)
        )
        z_s = (
            E_S * (A_B * _z_base + A_W * _z_wall) / (E_S * A_S)
        )  # w/o base this is just h/2
        I_S = (wb ** 3 * tb + tw * hw ** 3) / 12.0
        EI_s = (
            E_S * I_S + E_S * A_S * (z_s - z_cen) ** 2
        )  # dominant term should be (z_s - z_cen)^2 due to offset center
        return (
            EI_s / self.geometry.s_p / D11
        )  # TODO : temporarily multiply by n_stiff+1?

    @property
    def old_affine_exy(self):
        """
        get the exy so that lambda = kx_0y_0 the affine buckling coefficient for pure shear load
        out of the buckling analysis!
        """
        _Darray = self.old_Darray_plate
        D11 = _Darray[0]
        D22 = _Darray[2]
        _Aarray = self.Aarray_plate
        A66 = _Aarray[3]
        exy_T = 0.5 * (
            np.pi ** 2 * (D11 * D22 ** 3) ** 0.25 / self.geometry.b ** 2 / A66
        )
        return exy_T

    @property
    def affine_exy(self):
        """
        get the exy so that lambda = kx_0y_0 the affine buckling coefficient for pure shear load
        out of the buckling analysis!
        """
        _Darray = self.Darray_plate
        D11 = _Darray[0]
        D22 = _Darray[2]
        _Aarray = self.Aarray_plate
        A66 = _Aarray[3]
        # 0.5 *
        exy_T = np.pi ** 2 * (D11 * D22 ** 3) ** 0.25 / self.geometry.b ** 2 / A66

        # need N12 = this in order to get lambda = N12,cr*
        # dim_fact = np.pi ** 2 * (D11 * D22 ** 3) ** 0.25 / self.geometry.b ** 2
        # print(f"{dim_fact=}")
        # exit()
        return exy_T

    @property
    def intended_Nxy(self) -> float:
        """
        intended Nxx in linear static analysis
        """
        # may need to change to
        N12 = self.affine_exy * self.Aarray_plate[3]
        print(f"{N12=}")
        print(f"{self.Aarray_plate[3]=}")
        return N12

    def _test_broadcast(self):
        return
        # test bcast
        if self.comm.rank == 0:
            my_var = 0
        else:
            my_var = None
        print(f"pre myvar{self._index} = {my_var} on rank {self.comm.rank}")
        my_var = self.comm.bcast(my_var, root=0)
        print(f"post myvar{self._index} = {my_var} on rank {self.comm.rank}")
        self._index += 1

    def pre_analysis(
        self,
        # explicit mesh creation settings
        nx_plate=30,
        ny_plate=30,
        nz_stiff=10,
        nx_stiff_mult=1,
        exx=0.0,
        eyy=0.0,
        exy=0.0,
        clamped=False,
        _make_rbe=True,
        _explicit_poisson_exp=False,
        side_support: bool = True,
        # caps2tacs method settings
        use_caps=False,
        global_mesh_size=0.1,  # caps settings
        edge_pt_min=5,
        edge_pt_max=40,
    ):
        """
        Generate a stiffened plate mesh with CQUAD4 elements
        create pure axial, pure shear, or combined loading displacement
        """

        if self.geometry.num_stiff == 0:
            _make_rbe = False

        self._exx = exx
        self._exy = exy
        self._eyy = eyy

        self._test_broadcast()

        self._use_caps = use_caps

        if use_caps:
            # use caps2tacs to generate a stiffened panel
            tacs_model = caps2tacs.TacsModel.build(
                csm_file=self.csm_file, comm=self.comm, active_procs=[0]
            )
            tacs_aim = tacs_model.tacs_aim
            self._tacs_aim = tacs_aim

            tacs_model.mesh_aim.set_mesh(
                edge_pt_min=edge_pt_min,
                edge_pt_max=edge_pt_max,
                global_mesh_size=global_mesh_size,
                max_surf_offset=0.01,
                max_dihedral_angle=15,
            ).register_to(tacs_model)

            null_mat = caps2tacs.Isotropic.null().register_to(tacs_model)

            # set Stiffened plate geometry class into the CSM geometry
            tacs_aim.set_config_parameter("stiff_base", 0)
            tacs_aim.set_design_parameter("a", self.geometry.a)
            tacs_aim.set_design_parameter("b", self.geometry.b)
            tacs_aim.set_design_parameter("num_stiff", self.geometry.num_stiff)
            # tacs_aim.set_design_parameter("w_b", self.geometry.w_b)
            tacs_aim.set_design_parameter("h_w", self.geometry.h_w)

            # set shell properties with CompDescripts
            # auto makes shell properties (need thickDVs so that the compDescripts get written out from tacsAIM)
            caps2tacs.ThicknessVariable(
                caps_group="panel", value=self.geometry.h, material=null_mat
            ).register_to(tacs_model)
            # caps2tacs.ThicknessVariable(caps_group="rib", value=self.geometry.rib_h, material=null_mat).register_to(tacs_model)
            # caps2tacs.ThicknessVariable(caps_group="base", value=self.geometry.h+self.geometry.t_b, material=null_mat).register_to(tacs_model)
            caps2tacs.ThicknessVariable(
                caps_group="stiff", value=self.geometry.t_w, material=null_mat
            ).register_to(tacs_model)

            # add v,theta_z constraint to stiffener corner nodes - since they are tied off here to ribs
            # hope to produce more realistic shear modes, TODO : figure out whether this should be here
            # this isn't compatible with BCs for the
            # caps2tacs.PinConstraint(caps_constraint="stCorner", dof_constraint=26).register_to(tacs_model)

            self._test_broadcast()

            # run the pre analysis to build tacs input files
            tacs_aim._no_constr_override = True
            tacs_model.setup(include_aim=True)
            tacs_model.pre_analysis()

        elif self.comm.rank == 0:  # make the bdf file without CAPS
            if os.path.exists(self.bdf_file):
                os.remove(self.bdf_file)

            fp = open(self.bdf_file, "w")
            fp.write("$ Input file for a square axial/shear-disp BC plate\n")
            fp.write("SOL 103\nCEND\nBEGIN BULK\n")

            nodes_dict = []
            written_nodes = []
            elem_dicts = []
            node_id = 0
            elem_id = 0

            class Node:
                TOL = 1e-10

                def __init__(self, x, y, z, id):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.id = id

                @classmethod
                def in_tol(cls, x1, x2):
                    return abs(x1 - x2) < cls.TOL

                def same_location(self, x, y, z):
                    return (
                        self.in_tol(x, self.x)
                        and self.in_tol(y, self.y)
                        and self.in_tol(z, self.z)
                    )

                @property
                def node_dict(self) -> dict:
                    return {
                        "x": self.x,
                        "y": self.y,
                        "z": self.z,
                        "id": self.id,
                    }

            class Element:
                def __init__(self, id: int, part_id: int, nodes: list):
                    self.id = id
                    self.part_id = part_id
                    self.nodes = nodes

            class Mesh:
                def __init__(self):
                    self.nodes = []
                    self.elements = []

                @property
                def max_node_id(self) -> int:
                    if len(self.nodes) == 0:
                        return 0
                    else:
                        return max([node.id for node in self.nodes])

                @property
                def max_elem_id(self) -> int:
                    if len(self.elements) == 0:
                        return 0
                    else:
                        return max([elem.id for elem in self.elements])

                def add_node_at(self, x, y, z):
                    _existing_node = False
                    for node in self.nodes:
                        if node.same_location(x, y, z):
                            _existing_node = True
                            break
                    if _existing_node:
                        _id = node.id
                    else:
                        _id = self.max_node_id + 1
                    my_node = Node(x, y, z, _id)
                    if not _existing_node:  # add new node if didn't exist yet
                        self.nodes += [my_node]
                    return my_node

                def add_element(self, x_list, y_list, z_list, part_id: int):
                    """each x_list, y_list, z_list are the coordinates of the element points (e.g. length-4 list if quad)"""
                    nodes = []
                    for i in range(len(x_list)):
                        node = self.add_node_at(x_list[i], y_list[i], z_list[i])
                        nodes += [node]

                    elem_id = self.max_elem_id + 1
                    my_elem = Element(id=elem_id, part_id=part_id, nodes=nodes)
                    self.elements += [my_elem]
                    return my_elem

                @property
                def node_dicts(self) -> list:
                    return [node.node_dict for node in self.nodes]

            # make a new mesh object
            self.mesh = Mesh()

            N = self.geometry.num_local
            # make each local section and stiffener
            print(f"{N=}")
            stiff_locations = []
            for ilocal in range(N):
                # extra leftover
                if ilocal == 0:
                    ystart = 0
                elif ilocal == 1:
                    ystart += self.geometry.boundary_s_p
                else:
                    ystart += self.geometry.s_p

                stiff_locations += [ystart]

                # print(f"{ystart=}, {ilocal=}")
                # if ilocal == N-1:
                #     exit()

                Ly = (
                    self.geometry.s_p
                    if not (ilocal in [0, N - 1])
                    else self.geometry.boundary_s_p
                )
                # print(f"y from {ystart} to {ystart+Ly}")
                Lx = self.geometry.a
                Lz_stiff = self.geometry.h_w

                # make the local section of the plate
                dx = Lx / (nx_plate - 1)
                dy = Ly / (ny_plate - 1)
                for iy in range(ny_plate - 1):
                    y1 = dy * iy + ystart
                    y2 = y1 + dy
                    for ix in range(nx_plate - 1):
                        x1 = dx * ix
                        x2 = x1 + dx

                        self.mesh.add_element(
                            x_list=[x1, x2, x2, x1],
                            y_list=[y1, y1, y2, y2],
                            z_list=[0.0] * 4,
                            part_id=1,
                        )

                # make the stiffener if not last local section
                if ilocal == N - 1:
                    continue
                dz = Lz_stiff / (nz_stiff - 1)
                dx_stiff = dx / nx_stiff_mult
                ystiffener = ystart + Ly

                for iz in range(nz_stiff - 1):
                    z1 = iz * dz
                    z2 = z1 + dz
                    for ix in range(nx_plate - 1):
                        x_start = dx * ix
                        for istiff_mult in range(nx_stiff_mult):
                            x1 = x_start + istiff_mult * dx_stiff
                            x2 = x1 + dx_stiff

                            self.mesh.add_element(
                                x_list=[x1, x2, x2, x1],
                                y_list=[ystiffener] * 4,
                                z_list=[z1, z1, z2, z2],
                                part_id=2,  # for stiffener
                            )

            # write out all of the nodes
            for node in self.mesh.nodes:
                spc = " "
                coord_disp = 0
                coord_id = 0
                seid = 0
                fp.write(
                    "%-8s%16d%16d%16.9e%16.9e*       \n"
                    % ("GRID*", node.id, coord_id, node.x, node.y)
                )
                fp.write(
                    "*       %16.9e%16d%16s%16d        \n"
                    % (node.z, coord_disp, spc, seid)
                )

            # use the nodes_dict to write the CQUAD4 elements
            for elem in self.mesh.elements:
                fp.write(
                    "%-8s%8d%8d%8d%8d%8d%8d\n"
                    % (
                        "CQUAD4",
                        elem.id,
                        elem.part_id,
                        elem.nodes[0].id,
                        elem.nodes[1].id,
                        elem.nodes[2].id,
                        elem.nodes[3].id,
                    )
                )
            # fp.close()

        self._test_broadcast()

        # read the bdf file to get the boundary nodes
        # and then add custom BCs for axial or shear
        if self.comm.rank == 0:

            # read the BDF file
            hdl = open(self.bdf_file, "r")
            lines = hdl.readlines()
            hdl.close()

            next_line = False
            nodes = []

        if self.comm.rank == 0:

            if self._use_caps:
                for line in lines:
                    chunks = line.split(" ")
                    non_null_chunks = [_ for _ in chunks if not (_ == "" or _ == "\n")]
                    if next_line:
                        next_line = False
                        z_chunk = non_null_chunks[1].strip("\n")
                        node_dict["z"] = float(z_chunk)
                        nodes += [node_dict]
                        continue

                    if "GRID*" in line:
                        next_line = True
                        y_chunk = non_null_chunks[3].strip("*")

                        node_dict = {
                            "id": int(non_null_chunks[1]),
                            "x": float(non_null_chunks[2]),
                            "y": float(y_chunk),
                            "z": None,
                        }
            else:
                nodes = self.mesh.node_dicts  # copy the list

            # make nodes dict for nodes on the boundary
            boundary_nodes = []

            def in_tol(val1, val2):
                return abs(val1 - val2) < 1e-5

            for node_dict in nodes:
                x_left = in_tol(node_dict["x"], 0.0)
                x_right = in_tol(node_dict["x"], self.geometry.a)
                y_bot = in_tol(node_dict["y"], 0.0)
                y_top = in_tol(node_dict["y"], self.geometry.b)

                # no longer enforce xy-plane since also want to constraint stringers like plate perimeter too!
                # otherwise the stringers buckle strangely in shear. They need to be fixed to the rib/spar like the plate perimeter
                # Update: changed this so that the stiffeners don't receive in-plane compressive disp BCs, but do receive shear perimeter disps
                xy_plane = in_tol(node_dict["z"], 0.0)

                on_bndry = x_left or x_right or y_bot or y_top
                # on_bndry = on_bndry and xy_plane

                if on_bndry:
                    node_dict["xleft"] = x_left
                    node_dict["xright"] = x_right
                    node_dict["ybot"] = y_bot
                    node_dict["ytop"] = y_top
                    node_dict["xy_plane"] = xy_plane or self._compress_stiff_override

                    # print(f"boundary node dict = {node_dict}")

                    boundary_nodes += [node_dict]

            # need to read in the ESP/CAPS dat file
            # then append write the SPC cards to it
            # Set up the plate BCs so that it has u = uhat, for shear disp control
            # u = eps * y, v = eps * x, w = 0
            if self._use_caps:
                fp = open(self.dat_file, "r")
                dat_lines = fp.readlines()
                fp.close()

        self._test_broadcast()

        self._xi = None
        self._eta = None
        self._zeta = None
        self.num_nodes = None

        if self.comm.rank == 0:

            # non-dimensional xyz coordinates of the plate
            self._xi = []
            self._eta = []
            self._zeta = []

            for node in nodes:
                self._xi += [node["x"] / self.geometry.a]
                self._eta += [node["y"] / self.geometry.b]
                self._zeta += [node["z"] / self.geometry.h_w]
            self._xi = np.array(self._xi)
            self._eta = np.array(self._eta)
            self._zeta = np.array(self._zeta)
            self.num_nodes = len(nodes)

        self._xi = self.comm.bcast(self._xi, root=0)
        self._eta = self.comm.bcast(self._eta, root=0)
        self._zeta = self.comm.bcast(self._zeta, root=0)
        self.num_nodes = self.comm.bcast(self.num_nodes, root=0)
        self.comm.Barrier()

        if self.comm.rank == 0:

            if self._use_caps:
                post = False
                pre_lines = []
                post_lines = []
                for line in dat_lines:
                    if "MAT" in line:
                        post = True
                    if post:
                        post_lines += [line]
                    else:
                        pre_lines += [line]

            # make RBE2 elements
            all_rbe_control_nodes = []
            if _make_rbe:
                # open the bdf file to get the max # CQUAD4 elements
                fp1 = open(self.bdf_file, "r")
                lines_bdf = fp1.readlines()
                fp1.close()
                eid = 0
                for line in lines_bdf:
                    if "CQUAD4" in line:
                        chunks = line.split(" ")
                        no_empty_chunks = [_ for _ in chunks if not (_ == "")]
                        # print(no_empty_chunks)
                        _eid = int(no_empty_chunks[1])
                        if _eid > eid:
                            eid = _eid

                fp1 = open(self.bdf_file, "a")
                N = self.geometry.num_stiff + 1
                for iy in range(1, self.geometry.num_stiff + 1):
                    # yval = iy * self.geometry.b / N
                    yval = stiff_locations[iy]
                    for xval in [0, self.geometry.a]:
                        rbe_nodes = []
                        rbe_control_node = None

                        for node_dict in boundary_nodes:
                            if in_tol(node_dict["x"], xval) and in_tol(
                                node_dict["y"], yval
                            ):
                                zval = node_dict["z"]
                                xy_plane = node_dict["xy_plane"]
                                if node_dict["xy_plane"]:
                                    rbe_control_node = node_dict["id"]
                                    all_rbe_control_nodes += [rbe_control_node]
                                else:
                                    rbe_nodes += [node_dict["id"]]

                        # write the RBE element
                        eid += 1
                        # print(f"{rbe_control_node=}")
                        # print(f"{rbe_nodes=}")
                        # exit()
                        if len(rbe_nodes) > 0:
                            # print(f"writing {rbe_nodes=}")
                            # exit()
                            fp1.write(
                                "%-8s%8d%8d%8d"
                                % ("RBE2", int(eid), int(rbe_control_node), 23)
                            )
                            for rbe_node in rbe_nodes:
                                fp1.write("%8d" % (rbe_node))
                            fp1.write("\n")
                fp1.close()

            if self._use_caps:
                fp = open(self.dat_file, "w")
                for line in pre_lines:
                    fp.write(line)

            if _explicit_poisson_exp:
                eyy_poisson = -1.0 * self.intended_Nxx / self.A12_eff
                # print(f"{self.A12_eff=}")
                # print(f"{eyy_poisson=}")
                # exit()

            # add displacement control boundary conditions
            for node_dict in boundary_nodes:
                # still only apply BCs to xy plane, use RBEs to ensure this now
                # if not node_dict["xy_plane"]: continue
                x = node_dict["x"]
                y = node_dict["y"]
                z = node_dict["z"]
                nid = node_dict["id"]
                u = 0.5 * exy * y
                v = 0.5 * exy * x

                if _explicit_poisson_exp:
                    vpoisson = eyy_poisson * y

                # print(f"boundary node = {nid}, x {x}, y {y}, z {x}")

                # only enforce compressive displacements to plate, not stiffeners
                # TODO : maybe I need to do this for the stiffener too, but unclear
                # if node_dict["xy_plane"]:
                if node_dict["xright"] or exy != 0:
                    u -= exx * x
                elif node_dict["ytop"]:
                    v -= eyy * y
                elif node_dict["xleft"] or node_dict["ybot"]:
                    pass

                # check on boundary
                if clamped or (node_dict["xleft"] and node_dict["ybot"]):
                    fp.write(
                        "%-8s%8d%8d%8s%8.6f\n" % ("SPC", 1, nid, "3456", 0.0)
                    )  # w = theta_x = theta_y
                else:
                    if (
                        node_dict["id"] in all_rbe_control_nodes
                    ):  # no rotation of the rbe control node to stop rbe element
                        fp.write(
                            "%-8s%8d%8d%8s%8.6f\n" % ("SPC", 1, nid, "346", 0.0)
                        )  # w = theta_z = 0
                    else:
                        simply_supported = False
                        if side_support:
                            simply_supported = True
                        else:  # only support uniaxial / xx ends
                            simply_supported = node_dict["xleft"] or node_dict["xright"]

                        if simply_supported:  # then all edges simply supported
                            fp.write(
                                "%-8s%8d%8d%8s%8.6f\n" % ("SPC", 1, nid, "36", 0.0)
                            )  # w = theta_z = 0
                # TODO : maybe I need to do this for the stiffener too for exx case, but unclear
                # if node_dict["xy_plane"] or exy != 0:
                if exy != 0 or node_dict["xleft"] or node_dict["xright"]:
                    fp.write(
                        "%-8s%16d%16d%16s%16.9f\n" % ("SPC*", 1, nid, "1", u)
                    )  # u = eps_xy * y
                # add to boundary nodes left and right of plate on the bottom of plate for v=0 axial case or all edges
                # also all sides of exy case
                if (
                    exy != 0.0
                    or node_dict["ybot"]
                    # or (node_dict["xy_plane"] and not node_dict["ytop"])
                ):

                    fp.write(
                        "%-8s%16d%16d%16s%16.9f\n" % ("SPC*", 1, nid, "2", v)
                    )  # v = eps_xy * x

                if (
                    _explicit_poisson_exp
                    and exx != 0
                    and not (node_dict["xy_plane"])
                    and (node_dict["xleft"] or node_dict["xright"])
                ):
                    # xy_plane = node_dict["xy_plane"]
                    # print(f"{xy_plane=}")
                    # print(f"{x=} {y=} {z=}")
                    # print(f"{vpoisson=}"); exit()
                    fp.write(
                        "%-8s%16d%16d%16s%16.9f\n" % ("SPC*", 1, nid, "2", vpoisson)
                    )  # vpoisson on stiffener ends

            if self._use_caps:
                for line in post_lines:
                    fp.write(line)
                fp.close()
            else:  # not use caps
                # write material and property cards
                fp.write(
                    "MAT1*                 1              0.                              0. *0      \n"
                )
                fp.write(
                    "*0                   0.                                                 *1      \n"
                )
                fp.write("*1                   1.              0.              0. \n")
                fp.write("$ Femap Property  : panel\n")
                fp.write(
                    "PSHELL*               1               1           1.E-2               1 *0      \n"
                )
                fp.write("*0                   1.               1 0.8333333333333 \n")
                if self.geometry.num_stiff > 0:
                    fp.write("$ Femap Property  : stiff\n")
                    fp.write(
                        "PSHELL*               2               1  4.472135955E-4               1 *0      \n"
                    )
                    fp.write(
                        "*0                   1.               1 0.8333333333333 \n"
                    )

                fp.write("ENDDATA\n")
                fp.close()

        self.comm.Barrier()

    def post_analysis(self):
        """no derivatives here so just clear capsLock file"""
        # remove the capsLock file after done with analysis
        if self.comm.rank == 0:
            if self._use_caps and os.path.exists(self.caps_lock):
                os.remove(self.caps_lock)

        if not self._use_caps:
            Xpts = self.bucklingProb.Xpts.getArray()
            _xi = Xpts[0::3]
            _eta = Xpts[1::3]
            _zeta = Xpts[2::3]

            all_xi = self.comm.gather(_xi, root=0)
            all_eta = self.comm.gather(_eta, root=0)
            all_zeta = self.comm.gather(_zeta, root=0)
            all_num_nodes = self.comm.gather(_xi.shape[0], root=0)

            if self.comm.rank == 0:
                xi_list = [xi for xi in all_xi if xi is not None]
                self._xi = np.concatenate(xi_list) / self.geometry.a

                eta_list = [eta for eta in all_eta if eta is not None]
                self._eta = np.concatenate(eta_list) / self.geometry.b

                zeta_list = [zeta for zeta in all_zeta if zeta is not None]
                self._zeta = np.concatenate(zeta_list) / (self.geometry.h_w if self.geometry.num_stiff > 0 else 1.0)

                self.num_nodes = sum(
                    [num_nodes for num_nodes in all_num_nodes if num_nodes is not None]
                )

            else:
                self._xi = None
                self._eta = None
                self._zeta = None
                self.num_nodes = None

            # broadcast across all processors
            self._xi = self.comm.bcast(self._xi, root=0)
            self._eta = self.comm.bcast(self._eta, root=0)
            self._zeta = self.comm.bcast(self._zeta, root=0)
            self.num_nodes = self.comm.bcast(self.num_nodes, root=0)

            # save eigenvectors that are aggregated across all modes
            for imode in range(self.num_modes):
                all_eigvecs = self.comm.gather(self._eigenvectors[imode], root=0)
                if self.comm.rank == 0:
                    eigvec_list = [
                        eigvec for eigvec in all_eigvecs if eigvec is not None
                    ]
                    self._eigenvectors[imode] = np.concatenate(eigvec_list)
                else:
                    self._eigenvectors[imode] = None
                self._eigenvectors[imode] = self.comm.bcast(
                    self._eigenvectors[imode], root=0
                )

    def _elemCallback(self):
        """element callback to set the stiffener, base, panel material properties"""

        def elemCallBack(
            dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs
        ):
            # Set constitutive properties
            # rho = 4540.0  # density, kg/m^3
            # # E = 70e9  # elastic modulus, Pa 118e9
            # # nu = 0.33  # poisson's ratio
            # ys = 1050e6  # yield stress, Pa
            # kappa = 6.89
            # specific_heat = 463.0

            ref_axis = None
            if "panel" in compDescript:
                material = self.plate_material
                thickness = self.geometry.h
            elif "base" in compDescript:
                material = self.stiffener_material
                thickness = self.geometry.h + self.geometry.t_b
            elif "stiff" in compDescript:
                material = self.stiffener_material
                thickness = self.geometry.t_w
            elif "rib" in compDescript:
                material = self.plate_material
                thickness = self.geometry.rib_h
                ref_axis = np.array([0, 1, 0], dtype=TACS.dtype)
            else:
                raise AssertionError(
                    "elem does not belong to oneof the main components"
                )

            if ref_axis is None:
                ref_axis = material.ref_axis

            # if E22 not provided, isotropic
            isotropic = material._E22 is None

            # Setup property and constitutive objects
            if isotropic:
                mat = constitutive.MaterialProperties(E=material.E11, nu=material.nu12)

                # Set one thickness dv for every component
                con = constitutive.IsoShellConstitutive(mat, t=thickness)

            else:  # orthotropic
                # if single ply we do this
                # print(f"{len(material.ply_angles)=}")
                if len(material.ply_angles) == 1 or len(material.ply_angles) == 2: # sym
                    util = CompositeMaterialUtility(
                        E11=material.E11, E22=material.E22, nu12=material.nu12, G12=material.G12
                    ).rotate_ply(material._ply_angles[0])
                    G12 = util.G12
                    E11 = util.E11
                    E22 = util.E22
                    nu12 = util.nu12
                    # print(f"{G12=} {E11=} {E22=} {nu12=}")


                    # assume G23, G13 = G12
                    G23 = G12
                    G13 = G12

                    # to prevent stiffener crippling, make stiffeners stronger in transverse shear
                    # otherwise with composites, can dominate the modes (doesn't affect in-plane of stiffener, just trying this out)
                    # does actually affect the eigenvalues.. don't do it
                    # if "stiff" in compDescript:
                    #     G23 *= 5
                    #     G13 *= 5


                    ortho_prop = constitutive.MaterialProperties(
                        E1=util.E11,
                        E2=util.E22,
                        nu12=util.nu12,
                        G12=util.G12,
                        G23=G23,
                        G13=G13,
                    )

                else:
                    raise AssertionError("Doesn't support multi-ply laminate right now, needs verification.")

                ortho_ply = constitutive.OrthotropicPly(thickness, ortho_prop)

                # make sure it is a symmetric laminate by doing the symmetric number of plies
                # right now assume it is symmetric

                con = constitutive.CompositeShellConstitutive(
                    [ortho_ply],
                    np.array([thickness], dtype=dtype),
                    np.array([0], dtype=dtype),
                    tOffset=0.0,
                )

                # how to make sure it is a symmetric laminate?
                # con = constitutive.CompositeShellConstitutive(
                #     [ortho_ply] * material.num_plies,
                #     np.array(material.get_ply_thicknesses(thickness), dtype=dtype),
                #     np.array(material.rad_ply_angles, dtype=dtype),
                #     tOffset=0.0,
                # )

            # For each element type in this component,
            # pass back the appropriate tacs element object
            elemList = []
            for descript in elemDescripts:
                # if ref_axis is None:
                #     transform = None
                # else:
                #     transform = elements.ShellRefAxisTransform(ref_axis)
                transform = None
                if descript in ["CQUAD4", "CQUADR"]:
                    elem = elements.Quad4Shell(transform, con)
                elif descript in ["CQUAD9", "CQUAD"]:
                    elem = elements.Quad9Shell(transform, con)
                else:
                    raise AssertionError("Non CQUAD4 Elements in this plate?")

                elem.setComplexStepGmatrix(True)

                elemList.append(elem)

            # Add scale for thickness dv
            scale = [100.0]
            return elemList, scale

        return elemCallBack

    def run_static_analysis(self, base_path=None, write_soln=False):
        """
        run a linear static analysis on the flat plate with either isotropic or composite materials
        return the average stresses in the plate => to compute in-plane loads Nx, Ny, Nxy
        """

        # Instantiate FEAAssembler
        # os.chdir(self._tacs_aim.root_analysis_dir)
        FEAAssembler = pyTACS(self.dat_file, comm=self.comm)
        self.comm.Barrier()

        # Set up constitutive objects and elements
        FEAAssembler.initialize(self._elemCallback())

        # set complex step Gmatrix into all elements through assembler
        # FEAAssembler.assembler.setComplexStepGmatrix(True)

        # debug the static problem first
        SP = FEAAssembler.createStaticProblem(name="static")
        SP.solve()
        if write_soln:
            if base_path is None:
                base_path = os.getcwd()
            static_folder = os.path.join(base_path, self.static_folder_name)
            if not os.path.exists(static_folder) and self.comm.rank == 0:
                os.mkdir(static_folder)
            SP.writeSolution(outputDir=static_folder)

        # test the average stresses routine
        # compNum = 0 is the panel, 1 is stiffener component
        avgStresses = FEAAssembler.assembler.getAverageStresses(compNum=0)
        return avgStresses

    def run_buckling_analysis(
        self,
        sigma=30.0,
        num_eig=5,
        write_soln=False,
        derivatives=False,
        base_path=None,
    ):
        """
        run a linear buckling analysis on the flat plate with either isotropic or composite materials
        return the sorted eigenvalues of the plate => would like to include M
        """

        # test bcast
        self._test_broadcast()

        # os.chdir(self._tacs_aim.root_analysis_dir)

        # Instantiate FEAAssembler
        FEAAssembler = pyTACS(self.dat_file, comm=self.comm)

        # Set up constitutive objects and elements
        FEAAssembler.initialize(self._elemCallback())

        # set complex step Gmatrix into all elements through assembler
        # FEAAssembler.assembler.setComplexStepGmatrix(True)

        # Setup buckling problem
        self.bucklingProb = FEAAssembler.createBucklingProblem(
            name="buckle", sigma=sigma, numEigs=num_eig
        )
        self.bucklingProb.setOption("printLevel", 2)

        # exit()

        # solve and evaluate functions/sensitivities
        funcs = {}
        funcsSens = {}
        self.bucklingProb.solve()
        self.bucklingProb.evalFunctions(funcs)
        if derivatives:
            self.bucklingProb.evalFunctionsSens(funcsSens)
        if write_soln:
            if base_path is None:
                base_path = os.getcwd()
            buckling_folder = os.path.join(base_path, self.buckling_folder_name)
            if not os.path.exists(buckling_folder) and self.comm.rank == 0:
                os.mkdir(buckling_folder)
            self.bucklingProb.writeSolution(outputDir=buckling_folder)

        # save the eigenvectors for MAC and return errors from function
        self._eigenvectors = []
        self._eigenvalues = []
        self._num_modes = num_eig
        errors = []
        for imode in range(num_eig):
            eigval, eigvec = self.bucklingProb.getVariables(imode)
            self._eigenvectors += [eigvec]
            self._eigenvalues += [eigval]
            error = self.bucklingProb.getModalError(imode)
            errors += [error]
        self._errors = errors

        if self.comm.rank == 0:
            pprint(funcs)
            # pprint(funcsSens)

        self._solved_buckling = True
        self._alphas = {}

        # return the eigenvalues here
        return np.array([funcs[key] for key in funcs]), np.array(errors)

    @property
    def nondim_X(self):
        """non-dimensional X matrix for Gaussian Process model"""
        if self.comm.rank == 0:
            return np.concatenate(
                [
                    np.expand_dims(self._xi, axis=-1),
                    np.expand_dims(self._eta, axis=-1),
                    np.expand_dims(self._zeta, axis=-1),
                ],
                axis=1,
            )
        else:
            return None

    def get_eigenvector(self, imode, uvw=False):
        # convert eigenvectors to w coordinates only, 6 dof per shell
        eigvector = self._eigenvectors[imode]
        if uvw:
            uvw_subvector = np.concatenate(
                [eigvector[0::6], eigvector[1::6], eigvector[2::6]], axis=0
            )
            return uvw_subvector
        else:
            return eigvector[2::6]

    @property
    def num_modes(self) -> int:
        """number of eigenvalues or modes that were recorded"""
        return self._num_modes

    @property
    def eigenvectors(self):
        return [self.get_eigenvector(imode) for imode in range(self.num_modes)]

    @property
    def eigenvalues(self):
        return self._eigenvalues

    def is_non_crippling_mode(self, imode):
        # ensure that the majority of the eigenvector magnitude is due to w displacement
        # and not v displacement of stiffener, split into these two sub-vectors w_sub and total eigenvector
        eigvector = self._eigenvectors[imode]
        w_subvector = eigvector[2::6]
        uvw_subvector = np.concatenate(
            [eigvector[0::6], eigvector[1::6], eigvector[2::6]], axis=0
        )

        w_mag = np.sqrt(np.dot(w_subvector, w_subvector))
        full_mag = np.sqrt(np.dot(uvw_subvector, uvw_subvector))
        mag_frac = w_mag / full_mag
        return mag_frac > 0.9

    def _in_tol(self, val1, val2, tol=1e-5):
        return np.abs(val1 - val2) < tol

    def is_local_mode(self, imode, just_check_local=False, local_mode_tol=0.5):
        """check if its a local mode by comparing the inf-norm (or max) w displacements along the stiffeners to the overall plate"""
        N = self.geometry.num_stiff + 1
        w = self._eigenvectors[imode][2::6]  # get only the w displacement entries

        # trim out and remove RBE elements if need be
        w = w[: self.num_nodes]

        # compute max w displacement in the stiffeners
        mask = None
        for i in range(1, self.geometry.num_stiff + 1):
            # require at the panel surface only but under stiffener
            _mask = np.logical_and(
                self._in_tol(self._eta, i / N), self._in_tol(self._zeta, 0.0, tol=1e-5)
            )
            if mask is None:
                mask = _mask
            else:
                mask = np.logical_or(mask, _mask)
        
        # check for low relative deflections underneath the stiffeners
        w_stiff = w[mask]
        w_stiff_max = np.max(np.abs(w_stiff))
        w_max = max([np.max(np.abs(w)), 1e-13]) # in overall plate
        low_stiff_deflection = w_stiff_max / w_max < local_mode_tol

        # also check the middle of the plate in case you have even # stiffeners
        middle_dist = np.abs(0.5 - self._eta)
        min_middle_dist = np.min(middle_dist) # sometimes no edge right at the middle
        middle_plate_mask = np.logical_and(
            self._in_tol(self._eta, 0.5 + min_middle_dist), self._in_tol(self._zeta, 0.0, tol=1e-5)
        )
        w_middle = w[middle_plate_mask]
        w_middle_max = np.max(np.abs(w_middle))
        low_middle_deflection = w_middle_max / w_max < local_mode_tol

        # print(f"local_mode check: {imode=} {w_middle_max=} {w_stiff_max=} {w_max=}", flush=True)

        
        if just_check_local:
            return low_stiff_deflection or low_middle_deflection
        else:  # also checks for non stiffener crippling modes
            return (low_stiff_deflection or low_middle_deflection) and self.is_non_crippling_mode(imode)

    def is_global_mode(self, imode, just_check_global=False, local_mode_tol=0.20):
        """check that the mode is global i.e. the max w displacement occurs in the"""
        ERROR_TOL = 1e-5
        if abs(self._errors[imode]) > ERROR_TOL:
            return False  # reject this mode
        if just_check_global:
            return not self.is_local_mode(
                imode, just_check_local=True, local_mode_tol=local_mode_tol
            )
        else:
            return not self.is_local_mode(
                imode, just_check_local=True, local_mode_tol=local_mode_tol
            ) and self.is_non_crippling_mode(imode)

    @property
    def global_modes(self) -> list:
        return [imode for imode in range(self.num_modes) if self.is_global_mode(imode)]

    @property
    def global_mode_eigenvalues(self) -> list:
        return [self._eigenvalues[imode] for imode in self.global_modes]

    @property
    def min_global_mode_eigenvalue(self) -> float:
        if len(self.global_mode_eigenvalues) == 0:
            return None
        else:
            return np.min(np.abs(np.array(self.global_mode_eigenvalues)))

    def print_mode_classification(self):
        """for each mode print out whether it's a global, local, or stiffener crippling mode"""
        for imode in range(self.num_modes):
            if self.is_local_mode(imode):
                print(f"\tMode {imode} is local")
            elif self.is_global_mode(imode):
                print(f"\tMode {imode} is global")
            else:
                print(f"\tMode {imode} is stiffener crippling")
        return

    def cosine_mode_similarity(self, vec1, vec2):
        """cosine mode similarity metric for modal assurance criterion"""

        assert vec1.shape[0] == vec2.shape[0]
        # make both unit vectors
        unit_vec1 = vec1 / np.linalg.norm(vec1)
        unit_vec2 = vec2 / np.linalg.norm(vec2)

        # absolute value done since mode shapes are truly unsigned (form not sign matters)
        return np.abs(np.dot(unit_vec1, unit_vec2))

    def get_mac_global_mode(
        self,
        axial: bool = True,
        min_similarity: float = 0.3,
        local_mode_tol: float = 0.5,
    ):
        """
        (better modal assurance criterion method to help with mode distortion at higher gamma)
        use more strict modal assurance criterion to determine whether each 'global mode' matches the CF modes
        well enough (otherwise our global mode heuristic) will sometimes pick up global-local mixing modes
        """

        self._MAC_msg = None
        found_mode = False
        self._min_global_imode = None
        self._min_global_eigval = None
        self._min_global_mode_shape = None

        if self.comm.rank == 0:  # only do this on root proc

            obs_eigval = None

            # already eliminates purely local and stiffener crippling modes
            # but some mix global-local modes not caught yet at this point.
            for imode in range(self.num_modes):

                # ensure it is a global mode otherwise continue
                if not self.is_global_mode(imode, local_mode_tol=local_mode_tol):
                    continue

                # this gets only w defl as (N,) shape vector where there are N mesh nodes
                obs_eigvec = self.get_eigenvector(imode, uvw=False)
                obs_eigval = self._eigenvalues[imode]
                N = obs_eigvec.shape[0]

                # get xyz coords in non-dimensional form
                nondim_X = self.nondim_X
                assert nondim_X.shape[0] == N

                max_similarity = 0.0
                max_sim_m = -1

                # check whether matches well enough with previously established modes
                if axial:

                    # loop over known global axial modes
                    for m in range(1, 25 + 1):
                        n = 1
                        known_eigfunc = lambda xi, eta: np.sin(m * np.pi * xi) * np.sin(
                            n * np.pi * eta
                        )
                        known_eigvec = np.array(
                            [
                                known_eigfunc(nondim_X[i, 0], nondim_X[i, 1])
                                for i in range(N)
                            ]
                        )

                        # cosine similarity
                        similarity = self.cosine_mode_similarity(
                            obs_eigvec, known_eigvec
                        )
                        if similarity > max_similarity:
                            max_similarity = similarity
                            max_sim_m = m

                    if max_similarity > min_similarity:
                        found_mode = True
                        break

                else:  # shear

                    # # loop over various global shear modes
                    # for m in range(1,25+1):

                    #     # low rho0 case

                    #     # high rho0 case
                    #     for angle_deg in range(5, 85, 20):
                    break
                    # just assume that rejecting local, crippling modes is enough for now with shear

        # now save the MAC results to report later
        if axial and found_mode:
            if self.comm.rank == 0:
                # now store the max similarity
                self._min_global_imode = imode
                self._min_global_eigval = obs_eigval.real
                self._min_global_mode_shape = max_sim_m
                self._MAC_msg = f"MAC identified global mode {imode} with shape (m,n)=({max_sim_m},1) and {max_similarity=:.4f}"
                print(self._MAC_msg)

            # broadcast results
            # this is bugging out on my machine when I broadcast too much => pyTACS fails
            # self._min_global_imode = self.comm.bcast(self._min_global_imode, root=0)
            # self._min_global_eigval = self.comm.bcast(self._min_global_eigval, root=0)
            # self._MAC_msg = self.comm.bcast(self._MAC_msg, root=0)

        if axial and not found_mode:
            self._MAC_msg = "Global mode similar to CF modes not found.."

        shear = not (axial)
        if shear and self.comm.rank == 0:
            if obs_eigval is not None:
                self._min_global_imode = imode
                self._min_global_eigval = np.abs(obs_eigval.real)
                self._MAC_msg = f"No shear mode MAC rn just heuristic - first assumed global mode {imode} taken"
            else:
                self._min_global_eigval = None
                self._MAC_msg = "No shear global modes found with global-local check"

            if self.comm.rank == 0:
                print(self._MAC_msg)

        return self._min_global_eigval

    @property
    def min_global_mode_index(self) -> int:
        return self._min_global_imode

    @property
    def min_global_eigmode(self) -> np.ndarray:
        if self.min_global_mode_index is not None:
            return self._eigenvectors[self.min_global_mode_index].real
        else:
            return None

    def get_matching_global_mode(
        self, 
        other_plate_nondim_X:np.ndarray, 
        other_plate_eigmode:np.ndarray,
        min_similarity:float=0.7,
        local_mode_tol:float=0.7,
    ):
        """
        use another plate's nondim_X and eigenmode for nearby rho0, gamma solution for mode tracking
        Have to use interpolation to track across changing meshes
        """
        from scipy.interpolate import NearestNDInterpolator
        import matplotlib.pyplot as plt
        # print(f"{other_plate_nondim_X=} {other_plate_nondim_X.shape=}")
        xi1 = other_plate_nondim_X[:,0].astype(np.double)
        eta1 = other_plate_nondim_X[:,1].astype(np.double)
        phi1 = other_plate_eigmode[2::6].astype(np.double) # w component

        self._min_global_mode_shape = 1

        # now use interp2d to build a 2d function
        interp = NearestNDInterpolator(list(zip(xi1, eta1)), phi1)

        matching_imode = None


        # first get the most similar mode
        mode_sim_vec = []
        for imode in range(self.num_modes):

            # not checking local modes here (check after)

            # get current eigenmode data
            xi2 = self.nondim_X[:,0].astype(np.double)
            eta2 = self.nondim_X[:,1].astype(np.double)
            phi2 = self._eigenvectors[imode][2::6].astype(np.double) # w component
            # print(f"{phi2.shape=}")

            # get meshgrid format of new mesh
            xi2_unique = np.unique(np.round(xi2, 5))
            eta2_unique = np.unique(np.round(eta2, 5))
            XI2, ETA2 = np.meshgrid(xi2_unique, eta2_unique)
            PHI2 = np.zeros(XI2.shape)
            nxi = XI2.shape[1]
            neta = XI2.shape[0]
            # dxi = np.diff(xi2_unique)[0]
            # deta = np.diff(eta2_unique)[0]
            for ixi in range(nxi):
                for ieta in range(neta):
                    # find the closest point to the unique xi value
                    distances = (xi2 - xi2_unique[ixi])**2 + (eta2 - eta2_unique[ieta])**2
                    ind = np.argmin(distances)
                    # mask = np.logical_and(
                    #     np.abs(xi2 - xi2_unique[ixi]) < 0.01,
                    #     np.abs(eta2 - eta2_unique[ieta]) < 0.01
                    # )
                    # print(f"{dxi=} {deta=} {mask=} {np.sum(mask)=} {mask.shape=}")
                    # phi2_val = phi2[mask][0]
                    PHI2[ieta, ixi] = phi2[ind]

            # now interpolate phi1 onto new mesh
            PHI1_interp = interp(XI2, ETA2)

            # flatten both and make unit vectors
            phi2_vec = PHI2.flatten()
            phi1_interp_vec = PHI1_interp.flatten()

            # now take element wise dot product of two matrices
            cosine_sim = self.cosine_mode_similarity(phi1_interp_vec, phi2_vec)
            mode_sim_vec += [cosine_sim]

        # now determine the mode that is the most similar
        mode_sim_vec = np.array(mode_sim_vec)
        most_sim_imode = np.argmax(mode_sim_vec)
        cosine_sim = mode_sim_vec[most_sim_imode]
        if cosine_sim > min_similarity:
            matching_imode = most_sim_imode
            if self.comm.rank == 0:
                print(f"found mode {imode} to be matching with similarity {cosine_sim}\n")

        if matching_imode is not None:
            self._min_global_imode = matching_imode
            self._MAC_msg = f"MAC reverse rho0 tracking successful with sim {cosine_sim}"
            return self._eigenvalues[matching_imode]
        else:
            self._MAC_msg = f"MAC reverse rho0 tracking unsuccessful no modes similar enough (max sim {cosine_sim})"
            return None

    def get_nondim_slopes(self, imode:int, xedge:bool=True, m=1, n=1):
        """
        get nondim slopes for axial buckling mode shapes at x and y edges to see if solution is close to clamped BC or not
        this happens if gamma is high enough that the slopes at +x and -x go smaller
        """
        eigvec = self._eigenvectors[imode]
        w_eigvec = eigvec[2::6]
        wmax = np.max(np.abs(w_eigvec))

        yedge = not(xedge)
        if xedge: # get slopes at xedge ocmpared to SS slope sin(m*pi*x/a) is m*pi/a max slope at middle
            # but do so in nondim xi coords so sin(m*pi*xi)
            SS_slope = m * np.pi
            
            # get points in panel nearest x = 0 but not 0
            xvals_in_panel = self._xi[np.abs(self._zeta) < 1e-7]
            unique_xvals = np.unique(xvals_in_panel)
            dxi = np.diff(unique_xvals)[0]

            # get closest point to middle at (dx,1/2) in xi,eta space
            dist = np.abs(self._xi - dxi) + np.abs(self._eta - 0.5) + np.abs(self._zeta)
            ind = np.argmin(dist)

            act_slope = np.abs(w_eigvec[ind] / wmax / dxi)

        if yedge: # get slopes at xedge ocmpared to SS slope sin(n*pi*x/b) is n*pi/b max slope at middle
            # but do so in nondim eta coords so sin(n*pi*xi)
            SS_slope = n * np.pi
            
            # get points in panel nearest y = 0 but not 0
            yvals_in_panel = self._eta[np.abs(self._zeta) < 1e-7]
            deta = np.diff(np.unique(yvals_in_panel))[0]
            # get closest point to middle at (1/2,deta) in xi,eta space
            dist = np.abs(self._xi - 0.5) + np.abs(self._eta - deta) + np.abs(self._zeta)
            ind = np.argmin(dist)

            act_slope = np.abs(w_eigvec[ind] / wmax / deta)

        nd_slope = act_slope / SS_slope
        nd_slope = nd_slope.real
        if self.comm.rank == 0:
            edge_type = "xedge" if xedge else "yedge"
            print(f"{edge_type}: ND_slope={nd_slope} with equiv 1.0 SS, 0.0 ~clamped")

        return act_slope / SS_slope



    def N11_plate(self, exx) -> float:
        """axial load carried by plate, need to account for composite lamiante  case with E_eff"""
        return exx * self.plate_material.E_eff * self.geometry.h

    def N11_stiffener(self, exx) -> float:
        return exx * self.stiffener_material.E_eff * self.geometry.t_w

    def predict_crit_load_no_centroid(
        self, exx=0.0, exy=0.0, output_global=False, return_all=False
    ):

        # haven't treated the combined case yet
        assert exx == 0.0 or exy == 0.0

        if exx != 0.0:
            # N11_plate = self.intended_Nxx
            # _Darray = self.Darray_plate
            # D11 = _Darray[0]; D22 = _Darray[2]

            gamma = self.gamma_no_centroid
            rho0 = self.affine_aspect_ratio_no_centroid
            xi = self.xi_plate_no_centroid

            lam_star_global = min(
                [
                    (1 + gamma) * m1 ** 2 / rho0 ** 2 + rho0 ** 2 / m1 ** 2 + 2 * xi
                    for m1 in range(1, 50)
                ]
            )

            return lam_star_global, "global"  # temp

        else:  # exy != 0.0

            # high aspect ratio soln
            rho0 = self.affine_aspect_ratio_no_centroid
            gamma = self.gamma_no_centroid
            xi = self.xi_plate_no_centroid

            def high_AR_resid(s2):
                s1 = (1.0 + 2.0 * s2 ** 2 * xi + s2 ** 4 + gamma) ** 0.25
                term1 = s2 ** 2 + s1 ** 2 + xi / 3
                term2 = (
                    (3 + xi) / 9.0 + 4.0 / 3.0 * s1 ** 2 * xi + 4.0 / 3.0 * s1 ** 4
                ) ** 0.5
                return term1 - term2

            s2_bar = fsolve(high_AR_resid, 1.0)[0]
            s1_bar = (1.0 + 2.0 * s2_bar ** 2 * xi + s2_bar ** 4 + gamma) ** 0.25
            print(f"{s1_bar=}, {s2_bar=}")

            N12cr_highAR = (
                (
                    1.0
                    + gamma
                    + s1_bar ** 4
                    + 6 * s1_bar ** 2 * s2_bar ** 2
                    + s2_bar ** 4
                    + 2 * xi * (s1_bar ** 2 + s2_bar ** 2)
                )
                / 2.0
                / s1_bar ** 2
                / s2_bar
            )
            N12cr_lowAR = N12cr_highAR / rho0 ** 2
            return np.max([N12cr_highAR, N12cr_lowAR]), "global"

    def predict_crit_load(
        self, axial: bool = True, output_global=False, return_all=False
    ):

        # haven't treated the combined case yet
        # assert exx == 0.0 or exy == 0.0

        if axial:
            # N11_plate = self.intended_Nxx
            # _Darray = self.Darray_plate
            # D11 = _Darray[0]; D22 = _Darray[2]

            lam_star_global = min(
                [
                    (1 + self.gamma) * m1 ** 2 / self.affine_aspect_ratio ** 2
                    + self.affine_aspect_ratio ** 2 / m1 ** 2
                    + 2 * self.xi_plate
                    for m1 in range(1, 50)
                ]
            )

            return lam_star_global, "global"  # temp

        else:  # exy != 0.0

            # high aspect ratio soln
            rho0 = self.affine_aspect_ratio
            gamma = self.gamma
            xi = self.xi_plate

            def high_AR_resid(s2):
                s1 = (1.0 + 2.0 * s2 ** 2 * xi + s2 ** 4 + gamma) ** 0.25
                term1 = s2 ** 2 + s1 ** 2 + xi / 3
                term2 = (
                    (3 + xi) / 9.0 + 4.0 / 3.0 * s1 ** 2 * xi + 4.0 / 3.0 * s1 ** 4
                ) ** 0.5
                return term1 - term2

            s2_bar = fsolve(high_AR_resid, 1.0)[0]
            s1_bar = (1.0 + 2.0 * s2_bar ** 2 * xi + s2_bar ** 4 + gamma) ** 0.25
            print(f"{s1_bar=}, {s2_bar=}")

            N12cr_highAR = (
                (
                    1.0
                    + gamma
                    + s1_bar ** 4
                    + 6 * s1_bar ** 2 * s2_bar ** 2
                    + s2_bar ** 4
                    + 2 * xi * (s1_bar ** 2 + s2_bar ** 2)
                )
                / 2.0
                / s1_bar ** 2
                / s2_bar
            )
            N12cr_lowAR = N12cr_highAR / rho0 ** 2
            return np.max([N12cr_highAR, N12cr_lowAR]), "global"

    def size_stiffener(self, gamma, nx, nz, safety_factor=10, shear=False):
        lam_stiff0, lam_global0, _ = self.predict_crit_load(
            exx=(not shear) * self.affine_exx,
            exy=shear * self.affine_exy,
        )

        gamma0 = self.gamma
        hw = self.geometry.h_w
        tw = self.geometry.t_w
        SAR0 = self.geometry.stiff_AR
        factor1 = (gamma / gamma0 * tw * hw ** 3) ** 2
        factor2 = SAR0 ** 2 * lam_stiff0 / safety_factor / lam_global0
        hw_max = np.power(factor1 * factor2, 0.125)
        tw_new = tw * gamma / gamma0 * hw ** 3 / hw_max ** 3
        return hw_max, tw_new

    def __str__(self):
        mystr = f"Stiffened panel analysis object '{self._name}':\n"
        mystr += str(self.geometry)
        mystr += "Plate material:\n"
        mystr += str(self.plate_material)
        mystr += "Stiffener material:\n"
        mystr += str(self.stiffener_material)
        mystr += "\nAdvanced parameters in the stiffened panel analysis obj\n"
        mystr += f"\trho_0 = {self.affine_aspect_ratio}\n"
        mystr += f"\txi plate = {self.xi_plate}\n"
        mystr += f"\txi stiff = {self.xi_stiff}\n"
        mystr += f"\tgamma = {self.gamma}\n"
        mystr += f"\tdelta = {self.delta}\n"
        mystr += f"\tzeta plate = {self.zeta_plate}\n"
        mystr += f"\tzeta stiff = {self.zeta_stiff}\n"
        mystr += f"\tstiffAR = {self.geometry.stiff_AR}\n"
        mystr += "Mesh + Case Settings\n"
        mystr += f"\texx = {self._exx:.5e}\n"
        mystr += f"\texy = {self._exy:.5e}\n"
        mystr += f"\teyy = {self._eyy:.5e}\n"
        mystr += f"\tnum nodes = {self.num_nodes}\n"
        mystr += "In plane loads\n"
        Nxx = self.A11_eff * self._exx
        Nxy = self.Aarray_plate[-1] * self._exy
        mystr += f"\t{Nxx=}\n"
        mystr += f"\t{Nxy=}\n"
        mystr += self._MAC_msg + "\n"
        return mystr

__all__ = ["StiffenedPlateAnalysis"]

import numpy as np
from tacs import pyTACS, constitutive, elements, utilities, caps2tacs
import os
from pprint import pprint
from .stiffened_plate_geometry import StiffenedPlateGeometry
from .composite_material import CompositeMaterial

dtype = utilities.BaseUI.dtype

class StiffenedPlateAnalysis:
    def __init__(
        self,
        comm,
        geometry:StiffenedPlateGeometry,
        plate_material:CompositeMaterial,
        stiffener_material:CompositeMaterial,
        name=None,  # use the plate name to differentiate plate folder names
        compress_stiff=False, # whether to compress stiffeners to in axial case (TODO : figure this out => need to study static analysis)
    ):
        self.comm = comm
        self.geometry = geometry
        self.plate_material = plate_material
        self.stiffener_material = stiffener_material

        self._compress_stiff_override = compress_stiff

        # geometry properties
        self._name = name
        self._tacs_aim = None

    @property
    def buckling_folder_name(self) -> str:
        if self._name:
            return "buckling-" + self._name
        else:
            return "buckling"
        
    @property
    def static_folder_name(self) -> str:
        if self._name:
            return "static-" + self._name
        else:
            return "static"

    @property
    def bdf_file(self) -> str:
        return self._bdf_file

    @bdf_file.setter
    def bdf_file(self, new_file: str):
        self._bdf_file = new_file

    @property
    def csm_file(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir_path, "_stiffened_panel.csm")
    
    @property
    def caps_lock(self):
        if self._tacs_aim is not None:
            tacs_dir = self._tacs_aim.root_analysis_dir
            scratch_dir = os.path.dirname(tacs_dir)
            #caps_dir = os.path.dirname(scratch_dir)
            return os.path.join(scratch_dir, "capsLock")
        
    @property
    def dat_file(self):
        return self._tacs_aim.root_dat_file
    
    @property
    def bdf_file(self):
        tacs_dir = self._tacs_aim.root_analysis_dir
        return os.path.join(tacs_dir, "tacs.bdf")
    
    @property
    def affine_exx(self):
        """
        Solve exx such that lambda = lambda_min*
        Just estimate the overall buckling mode and require the user to adjust for local modes case
            when the stiffeners are stronger
        TODO : could estimate based on local mode too?
        """
        material = self.plate_material
        nu21 = material.nu12 * material.E22 / material.E11
        denom = 1.0 - material.nu12 * nu21
        D11 = material.E11 * self.geometry.h**3 / 12.0 / denom
        D22 = material.E22 * self.geometry.h**3 / 12.0 / denom
        exx_T = (
            np.pi**2 * np.sqrt(D11 * D22) / self.geometry.b**2 / self.geometry.h / material.E11
        )
        return exx_T
    
    @property
    def affine_eyy(self):
        """TODO : write this eqn out"""
        return None

    @property
    def affine_exy(self):
        """
        get the exy so that lambda = kx_0y_0 the affine buckling coefficient for pure shear load
        out of the buckling analysis!
        """
        material = self.plate_material
        nu21 = material.nu12 * material.E22 / material.E11
        denom = 1.0 - material.nu12 * nu21
        D11 = material.E11 * self.geometry.h**3 / 12.0 / denom
        D22 = material.E22 * self.geometry.h**3 / 12.0 / denom
        exy_T = (
            np.pi**2
            * (D11 * D22**3) ** 0.25
            / self.geometry.b**2
            / self.geometry.h
            / material.G12
        )
        return exy_T

    def pre_analysis(
            self, 
            global_mesh_size=0.1,
            exx=0.0, 
            eyy=0.0,
            exy=0.0, 
            clamped=False,
            edge_pt_min=5,
            edge_pt_max=40,
        ):
        """
        Generate a stiffened plate mesh with CQUAD4 elements
        create pure axial, pure shear, or combined loading displacement
        """

        # use caps2tacs to generate a stiffened panel
        tacs_model = caps2tacs.TacsModel.build(
            csm_file=self.csm_file, 
            comm=self.comm, 
            active_procs=[0]
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
        tacs_aim.set_design_parameter("a", self.geometry.a)
        tacs_aim.set_design_parameter("b", self.geometry.b)
        tacs_aim.set_design_parameter("num_stiff", self.geometry.num_stiff)
        tacs_aim.set_design_parameter("w_b", self.geometry.w_b)
        tacs_aim.set_design_parameter("h_w", self.geometry.h_w)

        # set shell properties with CompDescripts
        # auto makes shell properties (need thickDVs so that the compDescripts get written out from tacsAIM)
        caps2tacs.ThicknessVariable(caps_group="panel", value=self.geometry.h, material=null_mat).register_to(tacs_model)
        caps2tacs.ThicknessVariable(caps_group="base", value=self.geometry.h+self.geometry.t_b, material=null_mat).register_to(tacs_model)
        caps2tacs.ThicknessVariable(caps_group="stiff", value=self.geometry.t_w, material=null_mat).register_to(tacs_model)

        # add v,theta_z constraint to stiffener corner nodes - since they are tied off here to ribs
        # hope to produce more realistic shear modes, TODO : figure out whether this should be here
        # this isn't compatible with BCs for the 
        #caps2tacs.PinConstraint(caps_constraint="stCorner", dof_constraint=26).register_to(tacs_model)

        # run the pre analysis to build tacs input files
        tacs_aim._no_constr_override = True
        tacs_model.setup(include_aim=True)
        tacs_model.pre_analysis()


        # read the bdf file to get the boundary nodes
        # and then add custom BCs for axial or shear
        if self.comm.rank == 0:

            # read the BDF file
            hdl = open(self.bdf_file, "r")
            lines = hdl.readlines()
            hdl.close()

            next_line = False
            nodes = []

            for line in lines:
                chunks = line.split(" ")
                non_null_chunks = [_ for _ in chunks if not(_ == '' or _ == '\n')]
                if next_line:
                    next_line = False
                    z_chunk = non_null_chunks[1].strip('\n')
                    node_dict["z"] = float(z_chunk)
                    nodes += [node_dict]
                    continue

                if "GRID*" in line:
                    next_line = True
                    y_chunk = non_null_chunks[3].strip("*")

                    node_dict = {
                        "id" : int(non_null_chunks[1]),
                        "x" : float(non_null_chunks[2]),
                        "y" : float(y_chunk),
                        "z" : None,
                    }

            # make nodes dict for nodes on the boundary
            boundary_nodes = []
            def in_tol(val1,val2):
                return abs(val1-val2) < 1e-5
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
                #on_bndry = on_bndry and xy_plane

                if on_bndry:
                    node_dict["xleft"] = x_left
                    node_dict["xright"] = x_right
                    node_dict["ybot"] = y_bot
                    node_dict["ytop"] = y_top
                    node_dict["xy_plane"] = xy_plane or self._compress_stiff_override

                    #print(f"boundary node dict = {node_dict}")

                    boundary_nodes += [node_dict]

            # need to read in the ESP/CAPS dat file
            # then append write the SPC cards to it
                # Set up the plate BCs so that it has u = uhat, for shear disp control
                # u = eps * y, v = eps * x, w = 0
            fp = open(self.dat_file, "r")
            dat_lines = fp.readlines()
            fp.close()

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

            fp = open(self.dat_file, "w")
            for line in pre_lines:
                fp.write(line)

            for node_dict in boundary_nodes:
                x = node_dict["x"]
                y = node_dict["y"]
                nid = node_dict["id"]
                u = exy * y
                v = exy * x

                # only enforce compressive displacements to plate, not stiffeners
                # TODO : maybe I need to do this for the stiffener too, but unclear
                if node_dict["xy_plane"]: 
                    if node_dict["xright"] or exy != 0:
                        u -= exx * x
                    elif node_dict["ytop"]:
                        v -= eyy * y
                    elif node_dict["xleft"] or node_dict["ybot"]:
                        pass

                # check on boundary
                if clamped or (node_dict["xleft"] and node_dict["ybot"]):
                    fp.write(
                        "%-8s%8d%8d%8s%8.6f\n"
                        % ("SPC", 1, nid, "3456", 0.0)
                    )  # w = theta_x = theta_y
                else:
                    fp.write(
                        "%-8s%8d%8d%8s%8.6f\n"
                        % ("SPC", 1, nid, "36", 0.0)
                    )  # w = theta_x = theta_y
                # TODO : maybe I need to do this for the stiffener too for exx case, but unclear
                if node_dict["xy_plane"] or exy != 0: 
                    if exy != 0 or node_dict["xleft"] or node_dict["xright"]:
                        fp.write(
                            "%-8s%8d%8d%8s%8.6f\n" % ("SPC", 1, nid, "1", u)
                        )  # u = eps_xy * y
                    if exy != 0.0 or node_dict["ybot"]:
                        fp.write(
                            "%-8s%8d%8d%8s%8.6f\n" % ("SPC", 1, nid, "2", v)
                        )  # v = eps_xy * x

            for line in post_lines:
                fp.write(line)
            fp.close()

        self.comm.Barrier()

    def post_analysis(self):
        """no derivatives here so just clear capsLock file"""
        # remove the capsLock file after done with analysis
        os.remove(self.caps_lock)

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

            if "panel" in compDescript:
                material = self.plate_material
                thickness = self.geometry.h
            elif "base" in compDescript:
                material = self.stiffener_material
                thickness = self.geometry.h + self.geometry.t_b
            elif "stiff" in compDescript:
                material = self.stiffener_material
                thickness = self.geometry.t_w
            else:
                raise AssertionError("elem does not belong to oneof the main components")


            # if E22 not provided, isotropic
            isotropic = material._E22 is None

            # Setup property and constitutive objects
            if isotropic:
                mat = constitutive.MaterialProperties(E=material.E11, nu=material.nu12)

                # Set one thickness dv for every component
                con = constitutive.IsoShellConstitutive(mat, t=thickness)

            else:  # orthotropic
                # assume G23, G13 = G12
                G23 = material.G12 if material._G23 is None else material._G23
                G13 = material.G12 if material._G13 is None else material._G13
                ortho_prop = constitutive.MaterialProperties(
                    E1=material.E11,
                    E2=material.E22,
                    nu12=material.nu12,
                    G12=material.G12,
                    G23=G23,
                    G13=G13,
                )

                ortho_ply = constitutive.OrthotropicPly(thickness, ortho_prop)

                # one play composite constitutive model
                con = constitutive.CompositeShellConstitutive(
                    [ortho_ply],
                    np.array([thickness], dtype=dtype),
                    np.array([0], dtype=dtype),
                    tOffset=0.0,
                )
            # For each element type in this component,
            # pass back the appropriate tacs element object
            elemList = []
            for descript in elemDescripts:
                transform = None
                if descript in ["CQUAD4", "CQUADR"]:
                    elem = elements.Quad4Shell(transform, con)
                elif descript in ["CQUAD9", "CQUAD"]:
                    elem = elements.Quad9Shell(transform, con)
                else:
                    raise AssertionError("Non CQUAD4 Elements in this plate?")

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
        FEAAssembler = pyTACS(self.dat_file, comm=self.comm)

        # Set up constitutive objects and elements
        FEAAssembler.initialize(self._elemCallback())

        # set complex step Gmatrix into all elements through assembler
        FEAAssembler.assembler.setComplexStepGmatrix(True)

        # debug the static problem first
        SP = FEAAssembler.createStaticProblem(name="static")
        SP.solve()
        if write_soln:
            if base_path is None:
                base_path = os.getcwd()
            static_folder = os.path.join(base_path, self.static_folder_name)
            if not os.path.exists(static_folder):
                os.mkdir(static_folder)
            SP.writeSolution(outputDir=static_folder)

        # test the average stresses routine
        avgStresses = FEAAssembler.assembler.getAverageStresses()
        print(f"avg Stresses = {avgStresses}")
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

        # Instantiate FEAAssembler
        FEAAssembler = pyTACS(self.dat_file, comm=self.comm)

        # Set up constitutive objects and elements
        FEAAssembler.initialize(self._elemCallback())

        # set complex step Gmatrix into all elements through assembler
        FEAAssembler.assembler.setComplexStepGmatrix(True)

        # Setup buckling problem
        bucklingProb = FEAAssembler.createBucklingProblem(
            name="buckle", sigma=sigma, numEigs=num_eig
        )
        bucklingProb.setOption("printLevel", 2)

        # exit()

        # solve and evaluate functions/sensitivities
        funcs = {}
        funcsSens = {}
        bucklingProb.solve()
        bucklingProb.evalFunctions(funcs)
        if derivatives:
            bucklingProb.evalFunctionsSens(funcsSens)
        if write_soln:
            if base_path is None:
                base_path = os.getcwd()
            buckling_folder = os.path.join(base_path, self.buckling_folder_name)
            if not os.path.exists(buckling_folder):
                os.mkdir(buckling_folder)
            bucklingProb.writeSolution(outputDir=buckling_folder)

        # save the eigenvectors for MAC and return errors from function
        self._eigenvectors = []
        self._eigenvalues = []
        self._num_modes = num_eig
        errors = []
        for imode in range(num_eig):
            eigval, eigvec = bucklingProb.getVariables(imode)
            self._eigenvectors += [eigvec]
            self._eigenvalues += [eigval]
            error = bucklingProb.getModalError(imode)
            errors += [error]

        if self.comm.rank == 0:
            pprint(funcs)
            # pprint(funcsSens)

        self._solved_buckling = True
        self._alphas = {}

        # return the eigenvalues here
        return np.array([funcs[key] for key in funcs]), np.array(errors)
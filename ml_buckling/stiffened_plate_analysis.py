__all__ = ["StiffenedPlateAnalysis"]

import numpy as np
from tacs import pyTACS, constitutive, elements, utilities, caps2tacs, TACS
import os
from pprint import pprint
from .stiffened_plate_geometry import StiffenedPlateGeometry
from .composite_material import CompositeMaterial
from .composite_material_utility import CompositeMaterialUtility
from typing_extensions import Self

dtype = utilities.BaseUI.dtype

def exp_kernel1(xp, xq, sigma_f, L):
    # xp, xq are Nx1, Nx1 vectors
    return sigma_f**2 * np.exp(-0.5 * (xp - xq).T @ (xp - xq) / L**2)

class StiffenedPlateAnalysis:
    def __init__(
        self,
        comm,
        geometry:StiffenedPlateGeometry,
        plate_material:CompositeMaterial,
        stiffener_material:CompositeMaterial,
        name=None,  # use the plate name to differentiate plate folder names
        _compress_stiff=False, # whether to compress stiffeners to in axial case (TODO : figure this out => need to study static analysis)
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

    @classmethod
    def copy(cls, analysis:Self, name=None):
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
    def Darray_stiff(self) -> float:
        """array [D11,D12,D22,D66] for the stiffener"""
        zL = 0
        _Darray = np.zeros((4,))
        ply_thicknesses = self.stiffener_material.get_ply_thicknesses(self.geometry.t_w)
        for iply,ply_angle in enumerate(self.stiffener_material.ply_angles):
            ply_thick = ply_thicknesses[iply]
            zU = zL + ply_thick
            util = CompositeMaterialUtility(
                E11=self.stiffener_material.E11,
                E22=self.stiffener_material.E22,
                nu12=self.stiffener_material.nu12,
                G12=self.stiffener_material.G12
            ).rotate_ply(ply_angle)

            nu_denom = (1 - util.nu12 * util.nu21)
            Q11 = util.E11 / nu_denom
            Q22 = util.E22 / nu_denom
            Q12 = util.nu12 * Q22
            Q66 = util.G12

            _Darray[0] += 1.0/3 * Q11 * (zU**3 - zL**3)
            _Darray[1] += 1.0/3 * Q12 * (zU**3 - zL**3)
            _Darray[2] += 1.0/3 * Q22 * (zU**3 - zL**3)
            _Darray[3] += 1.0/3 * Q66 * (zU**3 - zL**3)

            zL = zU * 1.0
        return _Darray
    
    @property
    def xi_stiff(self):
        _Darray = self.Darray_stiff
        D11 = _Darray[0]; D12 = _Darray[1]; D22 = _Darray[2]; D66 = _Darray[3]
        return (D12 + 2 * D66) / np.sqrt(D11 * D22)
    
    @property
    def gen_poisson_stiff(self):
        _Darray = self.Darray_stiff
        D11 = _Darray[0]; D12 = _Darray[1]; D22 = _Darray[2]
        return 1/self.xi_stiff * D12/np.sqrt(D11 * D22)
    
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
    
    def _test_broadcast(self):
        return
        # test bcast
        my_var = 0
        print(f"pre myvar{self._index} = {my_var} on rank {self.comm.rank}")
        my_var = self.comm.bcast(my_var, root=0)
        print(f"post myvar{self._index} = {my_var} on rank {self.comm.rank}")
        self._index += 1

    def pre_analysis(
            self, 
            global_mesh_size=0.1,
            exx=0.0, 
            eyy=0.0,
            exy=0.0, 
            clamped=False,
            edge_pt_min=5,
            edge_pt_max=40,
            _make_rbe=True
        ):
        """
        Generate a stiffened plate mesh with CQUAD4 elements
        create pure axial, pure shear, or combined loading displacement
        """

        self._test_broadcast()

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
        # caps2tacs.ThicknessVariable(caps_group="rib", value=self.geometry.rib_h, material=null_mat).register_to(tacs_model)
        caps2tacs.ThicknessVariable(caps_group="base", value=self.geometry.h+self.geometry.t_b, material=null_mat).register_to(tacs_model)
        caps2tacs.ThicknessVariable(caps_group="stiff", value=self.geometry.t_w, material=null_mat).register_to(tacs_model)

        # add v,theta_z constraint to stiffener corner nodes - since they are tied off here to ribs
        # hope to produce more realistic shear modes, TODO : figure out whether this should be here
        # this isn't compatible with BCs for the 
        #caps2tacs.PinConstraint(caps_constraint="stCorner", dof_constraint=26).register_to(tacs_model)

        self._test_broadcast()

        # run the pre analysis to build tacs input files
        tacs_aim._no_constr_override = True
        tacs_model.setup(include_aim=True)
        tacs_model.pre_analysis()

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
                        no_empty_chunks = [_ for _ in chunks if not(_ == '')]
                        #print(no_empty_chunks)
                        _eid = float(no_empty_chunks[1])
                        if _eid > eid:
                            eid = _eid

                fp1 = open(self.bdf_file, "a")
                N = self.geometry.num_stiff + 1
                for iy in range(1,self.geometry.num_stiff+1):
                    yval = iy * self.geometry.b / N
                    for xval in [0, self.geometry.a]:
                        rbe_nodes = []
                        rbe_control_node = None

                        for node_dict in boundary_nodes:                        
                            if in_tol(node_dict["x"],xval) and in_tol(node_dict["y"], yval):
                                zval = node_dict["z"]
                                xy_plane = node_dict["xy_plane"]
                                if node_dict["xy_plane"]:
                                    rbe_control_node = node_dict["id"]
                                    all_rbe_control_nodes += [rbe_control_node]
                                else:
                                    rbe_nodes += [node_dict["id"]]

                        # write the RBE element
                        eid += 1
                        # also could do 123456 or 123 (but I don't really want no rotation here I don't think)
                        fp1.write("%-8s%8d%8d%8d" % ("RBE2", eid, rbe_control_node, 23)) #123456
                        for rbe_node in rbe_nodes:
                            fp1.write("%8d" % (rbe_node))
                        fp1.write("\n")
                fp1.close()

            fp = open(self.dat_file, "w")
            for line in pre_lines:
                fp.write(line)
      
            # add displacement control boundary conditions
            for node_dict in boundary_nodes:
                # still only apply BCs to xy plane, use RBEs to ensure this now
                #if not node_dict["xy_plane"]: continue
                x = node_dict["x"]
                y = node_dict["y"]
                nid = node_dict["id"]
                u = exy * y
                v = exy * x

                # only enforce compressive displacements to plate, not stiffeners
                # TODO : maybe I need to do this for the stiffener too, but unclear
                #if node_dict["xy_plane"]: 
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
                    if node_dict["id"] in all_rbe_control_nodes: # no rotation of the rbe control node to stop rbe element 
                        fp.write(
                            "%-8s%8d%8d%8s%8.6f\n"
                            % ("SPC", 1, nid, "346", 0.0)
                        )  # w = theta_z = 0
                    else:
                        fp.write(
                            "%-8s%8d%8d%8s%8.6f\n"
                            % ("SPC", 1, nid, "36", 0.0)
                        )  # w = theta_z = 0
                # TODO : maybe I need to do this for the stiffener too for exx case, but unclear
                #if node_dict["xy_plane"] or exy != 0: 
                if exy != 0 or node_dict["xleft"] or node_dict["xright"]:
                    fp.write(
                        "%-8s%8d%8d%8s%8.6f\n" % ("SPC", 1, nid, "1", u)
                    )  # u = eps_xy * y
                # add to boundary nodes left and right of plate on the bottom of plate for v=0 axial case or all edges 
                # also all sides of exy case
                if exy != 0.0 or node_dict["ybot"] or (node_dict["xy_plane"] and not node_dict["ytop"]): 
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
        if self.comm.rank == 0 and os.path.exists(self.caps_lock):
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
                raise AssertionError("elem does not belong to oneof the main components")
            
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

                # make sure it is a symmetric laminate by doing the symmetric number of plies
                # right now assume it is symmetric

                # print(f"num plies = {material.num_plies}")
                # print(f"ply thicknesses = {material.get_ply_thicknesses(thickness)}")
                # print(f"ply angles = {material.rad_ply_angles}")
                # exit()

                # how to make sure it is a symmetric laminate?
                con = constitutive.CompositeShellConstitutive(
                    [ortho_ply]*material.num_plies,
                    np.array(material.get_ply_thicknesses(thickness), dtype=dtype),
                    np.array(material.rad_ply_angles, dtype=dtype),
                    tOffset=0.0,
                )

            # For each element type in this component,
            # pass back the appropriate tacs element object
            elemList = []
            for descript in elemDescripts:
                if ref_axis is None:
                    transform = None
                else:
                    transform = elements.ShellRefAxisTransform(ref_axis)
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
        #os.chdir(self._tacs_aim.root_analysis_dir)
        FEAAssembler = pyTACS(self.dat_file, comm=self.comm)
        self.comm.Barrier()

        # Set up constitutive objects and elements
        FEAAssembler.initialize(self._elemCallback())

        # set complex step Gmatrix into all elements through assembler
        #FEAAssembler.assembler.setComplexStepGmatrix(True)

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
        avgStresses = FEAAssembler.assembler.getAverageStresses()
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

        #os.chdir(self._tacs_aim.root_analysis_dir)

        # Instantiate FEAAssembler
        FEAAssembler = pyTACS(self.dat_file, comm=self.comm)

        # Set up constitutive objects and elements
        FEAAssembler.initialize(self._elemCallback())

        # set complex step Gmatrix into all elements through assembler
        #FEAAssembler.assembler.setComplexStepGmatrix(True)

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
            if not os.path.exists(buckling_folder) and self.comm.rank == 0:
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
    
    def predict_crit_load(self, exx=0, exy=0, eyy=0, mode_loop=True):
        assert exy == 0 and eyy == 0 #haven't written these yet
        assert self.geometry.num_stiff == 1 # hasn't been implemented for >1 stiffeners yet

        # axial mode case
        # -----------------------------
        # get the stiffener bending stiffeness EI about modulus weighted centroid
        E_S = self.stiffener_material.E_eff
        A_W = self.geometry.area_w
        A_B = self.geometry.area_b
        wb = self.geometry.w_b
        tb = self.geometry.t_b
        #print(f"tb = {tb}")
        tw = self.geometry.t_w
        hw = self.geometry.h_w
        a = self.geometry.a
        b = self.geometry.b
        A_S = self.geometry.area_S
        E_P = self.plate_material.E_eff
        A_P = self.geometry.area_P
        I_P = self.geometry.h**3 / 12.0

        # modulus weighted centroid zcen
        z_cen = E_S * (A_B * tb/2.0 + A_W * hw/2.0) / (E_S * A_S + E_P * A_P)
        z_s = E_S * (A_B * tb/2.0 + A_W * hw/2.0) / (E_S * A_S)
        I_S = (wb**3 * tb + tw*hw**3) / 12.0
        EI_s = E_S * I_S + E_S * A_S * (z_s - z_cen)**2

        D11 = self.plate_material.Q11 * I_P
        D22 = self.plate_material.Q22 * I_P
        D12 = self.plate_material.Q12 * I_P
        D66 = self.plate_material.Q66 * I_P
        # print(f"D11 = {D11}, D22 = {D22}, D12 = {D12}")

        delta = A_S / A_P

        # global mode
        if mode_loop:
            N11_crit_global = 1e10
            for m1_star in range(1,51):
                _N11_crit_global = 2 * a * np.pi**2 / m1_star**2 / b / (0.5 + delta) * (0.25 * (D11 * m1_star**4 * b / a**3 + D22 * a/ b**3) + m1_star**2 / a * (1.0/b * (D12/2.0 + D66) + m1_star**2 / 2.0 / a**2 * EI_s))
                if _N11_crit_global < N11_crit_global:
                    N11_crit_global = _N11_crit_global
        else:
            m1_star = a/b * (D22 / (D11 + 2*EI_s / b))**0.25
            N11_crit_global = 2 * a * np.pi**2 / m1_star**2 / b / (0.5 + delta) * (0.25 * (D11 * m1_star**4 * b / a**3 + D22 * a/ b**3) + m1_star**2 / a * (1.0/b * (D12/2.0 + D66) + m1_star**2 / 2.0 / a**2 * EI_s))

        # print(f"N11 crit global = {N11_crit_global}")

        # local mode
        if mode_loop:
            N11_crit_local = 1e10
            for m2_star in range(1,51):
                _N11_crit_local = (m2_star**2 / a**2 * D11 + 16.0 * a**2 / m2_star**2 / b**4 * D22 + 8.0/b**2 * (D12 + 2.0 * D66)) * np.pi**2
                if _N11_crit_local < N11_crit_local:
                    N11_crit_local = _N11_crit_local
        else:
            m2_star = 2.0 * a / b * (D22 / D11) ** 0.25
            N11_crit_local = (m2_star**2 / a**2 * D11 + 16.0 * a**2 / m2_star**2 / b**4 * D22 + 8.0/b**2 * (D12 + 2.0 * D66)) * np.pi**2

        
        if N11_crit_global < N11_crit_local:
            N11_crit = N11_crit_global
        else:
            N11_crit = N11_crit_local

        s11_app = exx * E_P

        # compute current N11, should it be effective E11 here?
        # N11 = exx * self.plate_material.E11 * self.geometry.h
        N11 = exx * E_P * self.geometry.h

        _lambda = N11_crit / N11
        return _lambda
    
    MAC_THRESHOLD = 0.1  # 0.6

    @property
    def nondim_X(self):
        """non-dimensional X matrix for Gaussian Process model"""
        return np.concatenate([
            np.expand_dims(self._xi, axis=-1),
            np.expand_dims(self._eta, axis=-1),
            np.expand_dims(self._zeta, axis=-1)
        ], axis=1,
        )
    
    def get_eigenvector(self, imode, uvw=False):
        # convert eigenvectors to w coordinates only, 6 dof per shell
        # print(f"ndof in eigenvector = {self._eigenvectors[imode].shape[0]}")
        eigvector = self._eigenvectors[imode]
        if uvw:
            uvw_subvector = np.concatenate([eigvector[0::6], eigvector[1::6], eigvector[2::6]], axis=0)
            return uvw_subvector
        else:
            return eigvector[2::6]
    
    def interpolate_eigenvectors(self, X_test, compute_covar=False):
        """
        interpolate the eigenvector from this object the nominal plate to a new mesh in non-dim coordinates
        """
        X_train = self.nondim_X
        num_train = int(self.num_nodes)
        num_test = X_test.shape[0]
        # default hyperparameters
        sigma_n = 1e-4
        sigma_f = 1.0
        L = 0.4
        _kernel = lambda xp, xq: exp_kernel1(xp, xq, sigma_f=sigma_f, L=L)
        K_train = sigma_n**2 * np.eye(num_train) + np.array(
            [
                [_kernel(X_train[i, :], X_train[j, :]) for i in range(num_train)]
                for j in range(num_train)
            ]
        )
        K_test = np.array(
            [
                [_kernel(X_train[i, :], X_test[j, :]) for i in range(num_train)]
                for j in range(num_test)
            ]
        )

        if not compute_covar:
            _interpolated_eigenvectors = []
            for imode in range(self.num_modes):
                phi = self.get_eigenvector(imode)
                if self._saved_alphas:  # skip linear solve in this case
                    alpha = self._alphas[imode]
                else:
                    alpha = np.linalg.solve(K_train, phi)
                    self._alphas[imode] = alpha
                phi_star = K_test @ alpha
                _interpolated_eigenvectors += [phi_star]
            self._saved_alphas = True
            return _interpolated_eigenvectors
        else:
            raise AssertionError(
                "Haven't written part of extrapolate eigenvector to get the conditional covariance yet."
            )
    
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
        uvw_subvector = np.concatenate([eigvector[0::6], eigvector[1::6], eigvector[2::6]], axis=0)

        w_mag = np.sqrt(np.dot(w_subvector, w_subvector))
        full_mag = np.sqrt(np.dot(uvw_subvector, uvw_subvector))
        mag_frac = w_mag / full_mag
        #print(f"w_mag {w_mag}, full mag {full_mag}, mag frac {mag_frac}")
        return mag_frac > 0.9

    @classmethod
    def mac_permutation(
        cls, nominal_plate : Self, new_plate : Self, num_modes: int
    ) -> dict:
        """
        compute the permutation of modes in the new plate that correspond to the modes in the nominal plate
        using Gaussian Process interpolation for modal assurance criterion
        """
        eigenvalues = [None for _ in range(num_modes)]
        permutation = {}
        nominal_interp_modes = nominal_plate.interpolate_eigenvectors(
            X_test=new_plate.nondim_X
        )
        new_modes = new_plate.eigenvectors

        # _debug = False
        # if _debug:
        #     for imode, interp_mode in enumerate(nominal_interp_modes):
        #         interp_mat = new_plate._vec_to_plate_matrix(interp_mode)
        #         import matplotlib.pyplot as plt

        #         plt.imshow(interp_mat.astype(np.double))
        #         plt.show()

        for imode, nominal_mode in enumerate(nominal_interp_modes):
            if imode >= num_modes:  # if larger than number of nominal modes to compare
                break

            # skip crippling modes
            if not nominal_plate.is_non_crippling_mode(imode): continue

            nominal_mode_unit = nominal_mode / np.linalg.norm(nominal_mode)

            similarity_list = []
            for inew,new_mode in enumerate(new_modes):

                # skip crippling modes
                if not new_plate.is_non_crippling_mode(inew): continue
                new_mode_unit = new_mode / np.linalg.norm(new_mode)
                # compute cosine similarity with the unit vectors
                similarity_list += [
                    abs(np.dot(nominal_mode_unit, new_mode_unit).astype(np.double))
                ]

            # compute the maximum similarity index
            # if _debug:
            #     print(f"similarity list imode {imode} = {similarity_list}")
            jmode_star = np.argmax(np.array(similarity_list))
            permutation[imode] = jmode_star
            if similarity_list[jmode_star] > cls.MAC_THRESHOLD:  # similarity threshold
                eigenvalues[imode] = new_plate.eigenvalues[jmode_star]

        # check the permutation map is one-to-one

        # print to the terminal about the modal criterion
        print(f"0-based mac criterion permutation map")
        print(
            f"\tbetween nominal plate {nominal_plate._name} and new plate {new_plate._name}"
        )
        print(f"\tthe permutation map is the following::\n")
        for imode in permutation:
            print(f"\t nominal {imode} : new {permutation[imode]}")
        eigenvalues = np.real(eigenvalues)

        return eigenvalues, permutation



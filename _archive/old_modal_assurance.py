   
class OldModalAssurance:
    @classmethod
    def mac_permutation(
        cls, nominal_plate, new_plate, num_modes: int
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
            if not nominal_plate.is_non_crippling_mode(imode):
                continue

            nominal_mode_unit = nominal_mode / np.linalg.norm(nominal_mode)

            similarity_list = []
            for inew, new_mode in enumerate(new_modes):

                # skip crippling modes
                if not new_plate.is_non_crippling_mode(inew):
                    continue
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
        K_train = sigma_n ** 2 * np.eye(num_train) + np.array(
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
        
    def eigenvalue_correction_factor(self, in_plane_loads, axial:bool=True):
        if axial:
            # since we compute exx BCs so that we wanted N11_intended and lambda1 = N11,cr/N11_intended
            #  but we actually get lambda2 = N11,cr/N11_achieved
            #  we can find lambda1 = lambda2 * N11_achieved / N11_intended using our loading correction factor on the eigenvalue
            N11_intended = self.intended_Nxx
            print(f"{N11_intended=}")
            N11_achieved = -in_plane_loads[0]
            return np.real(N11_achieved / N11_intended)
        else: # shear, TODO
            pass

    def write_geom(
        self,
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
            name="buckle", sigma=10.0, numEigs=20
        )
        self.bucklingProb.setOption("printLevel", 2)

        # exit()

        if base_path is None:
            base_path = os.getcwd()
        buckling_folder = os.path.join(base_path, self.buckling_folder_name)
        if not os.path.exists(buckling_folder) and self.comm.rank == 0:
            os.mkdir(buckling_folder)
        self.bucklingProb.writeSolution(outputDir=buckling_folder)
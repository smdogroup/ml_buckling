def relu(x):
    return max([0.0, x])


def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))


def soft_abs(x, rho=10):
    return 1.0 / rho * np.log(np.exp(rho * x) + np.exp(-rho * x))


sigma_n = 1e-1  # 1e-1 was old value

# this one was a pretty good model except for high gamma, middle rho0 for one region of the design
kernel_option = args.kernel  # best is 7 right now
if kernel_option == 1:

    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]

        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2_1 = 1.0 + 0.2 * xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2_2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3_1 = 1.0 + 0.2 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        kernel3_2 = xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)

        # log(rho_0) direction
        # idea here is to combine linear kernel on (rho0, gamma, zeta) for rho0 outside [-1,1] the tails
        #    using weaker linear functions in gamma, zeta directions (by + const)
        #    also the rho0 of course is actually bilinear
        # then use weak SE term to cover the oscillations in rho0 and in this example we use regular
        #     linear kernels without a constant term because then it couples the gamma = 0 to high gamma data too much
        #     and messes up the mode switching zones
        #     put this in the paper.

        # * np.exp(-0.5 * (xp[3] + xq[3]) / 1.0) # 0.2, # * np.exp(-0.5 * (d1 ** 2 / (0.2 ** 2))
        kernel123 = (
            0.1 + soft_relu(-xp[1]) * soft_relu(-xq[1])
        ) * kernel2_1 * kernel3_1 + 0.1 * np.exp(
            -0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2)
        ) * soft_relu(
            1 - abs(xp[1])
        ) * soft_relu(
            1 - abs(xq[1])
        ) * kernel2_2 * kernel3_2

        return kernel0 * kernel123

elif kernel_option == 2:

    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 0.1 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1 = (
            soft_relu(-xp[1]) * soft_relu(-xq[1])
            + 0.1
            + 0.1  # * np.exp(-0.5 * (xp[3] + xq[3]) / 1.0) # 0.2
            # * np.exp(-0.5 * (d1 ** 2 / (0.2 ** 2))
            * np.exp(-0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2))
            * soft_relu(1 - abs(xp[1]))
            * soft_relu(1 - abs(xq[1]))
        )
        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        return kernel0 * kernel1 * kernel2 * kernel3

elif kernel_option == 3:

    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1_1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1
        kernel1_2 = (
            0.1  # * np.exp(-0.5 * (xp[3] + xq[3]) / 1.0) # 0.2
            # * np.exp(-0.5 * (d1 ** 2 / (0.2 ** 2))
            * np.exp(-0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2))
            * soft_relu(1 - abs(xp[1]))
            * soft_relu(1 - abs(xq[1]))
        )
        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = (1.0 + 0.2 * xp[2] * xq[2]) ** 2 + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = 1.0 + 0.5 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        return (
            kernel1_1 * kernel0 * kernel2 * kernel3
        )  # + kernel1_2  # add perturbations separately

elif kernel_option == 4:
    # this one works great with the xi dimension and doesn't have any funky d/dgamma slopes becoming negative
    # at higher xi values

    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1_1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1
        kernel1_2 = (
            0.1  # * np.exp(-0.5 * (xp[3] + xq[3]) / 1.0) # 0.2
            # * np.exp(-0.5 * (d1 ** 2 / (0.2 ** 2))
            * np.exp(-0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2))
            * soft_relu(1 - abs(xp[1]))
            * soft_relu(1 - abs(xq[1]))
        )
        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = (1.0 + 0.2 * xp[2] * xq[2]) ** 2 + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = 1.0 + 0.5 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        return (
            kernel1_1 + kernel0 + kernel2 + kernel3
        )  # + kernel1_2  # add perturbations separately

elif kernel_option == 5:
    # didn't work that well and I think it's the multiplication
    # it was not related to the SE term on dimension 1
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 0.1 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1

        kernel1_2 = 1.0 + 0.03 * np.exp(
            -0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2)
        ) * soft_relu(1 - abs(xp[1])) * soft_relu(
            1 - abs(xq[1])
        )  # * np.exp(-0.5 * d3 ** 2 / 9.0)

        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)

        return kernel0 * kernel1 * kernel2 * kernel3 * kernel1_2

elif kernel_option == 6:
    # yeah that helps, this is a reasonable model => but there's too much oscillation from the SE kernel term
    # maybe try and add xi, gamma, zeta product to the kernel1_2 term
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1_1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1

        kernel1_2 = 1.0 + 0.01 * np.exp(
            -0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2)
        ) * soft_relu(1 - abs(xp[1])) * soft_relu(
            1 - abs(xq[1])
        )  # * np.exp(-0.5 * d3 ** 2 / 9.0)

        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = (1.0 + 0.2 * xp[2] * xq[2]) ** 2 + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = 1.0 + 0.5 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)

        return (kernel0 + kernel1_1 + kernel2 + kernel3) * kernel1_2

elif kernel_option == 7:
    # This one worked great! Captured more of the local data with a tuned value of the
    # SE coefficient to 0.02 and the linear terms are of a different style from the
    # bilinear and SE part of rho0.
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1_1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1

        kernel1_2 = 1.0 + 0.02 * np.exp(-0.5 * (d1 ** 2 / 0.2 ** 2)) * soft_relu(
            1 - abs(xp[1])
        ) * soft_relu(
            1 - abs(xq[1])
        )  # * np.exp(-0.5 * d3 ** 2 / 9.0)

        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = (1.0 + 0.2 * xp[2] * xq[2]) ** 2 + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = 1.0 + 0.5 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)

        return (
            kernel1_1 * (kernel0 + kernel2 + kernel3)
            + kernel1_2 * kernel0 * kernel2 * kernel3
        )

elif kernel_option == 8:
    # This one worked great! Captured more of the local data with a tuned value of the
    # SE coefficient to 0.02 and the linear terms are of a different style from the
    # bilinear and SE part of rho0.
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1_1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1

        kernel1_2 = 1.0 + 0.02 * np.exp(-0.5 * (d1 ** 2 / 0.2 ** 2)) * soft_relu(
            1 - abs(xp[1])
        ) * soft_relu(
            1 - abs(xq[1])
        )  # * np.exp(-0.5 * d3 ** 2 / 9.0)

        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = (1.0 + 0.2 * xp[2] * xq[2]) ** 2 + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = 1.0 + 0.5 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)

        return (
            kernel1_1 * (kernel0 + kernel2 + kernel3)
            + kernel1_2 * kernel0 * kernel2 * kernel3
        )

elif kernel_option == 9:
    # This one worked great! Captured more of the local data with a tuned value of the
    # SE coefficient to 0.02 and the linear terms are of a different style from the
    # bilinear and SE part of rho0.
    def kernel(xp, xq, theta, debug=False):
        # xp, xq are Nx1,Mx1 vectors (ln(1+xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        BL_kernel = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1
        if debug:
            print(f"BL_kernel = {BL_kernel}")
        SE_kernel = (
            0.02
            * np.exp(-0.5 * (d1 ** 2 / 0.2 ** 2))
            * soft_relu(1 - soft_abs(xp[1]))
            * soft_relu(1 - soft_abs(xq[1]))  # * np.exp(-0.5 * d3 ** 2 / 9.0)
        )
        if debug:
            print(f"SE kernel = {SE_kernel}")
        gamma_kernel = 1.0 + 0.1 * xp[3] * xq[3]
        if debug:
            print(f"gamma vals = {xp[3]}, {xq[3]}")
        if debug:
            print(f"gamma kernel = {gamma_kernel}")
        xi_kernel = 0.1 * xp[0] * xq[0]
        if debug:
            print(f"xi kernel = {xi_kernel}")

        inner_kernel = (
            BL_kernel * (1.0 + 0.1 * xp[3] * xq[3]) + SE_kernel + 0.1 * xp[0] * xq[0]
        )
        if debug:
            print(f"inner kernel = {inner_kernel}")
        return inner_kernel * (1 + 0.01 * xp[2] * xq[2])

elif kernel_option == 10:
    # want to get lower RMSE
    # this is latest one as of 8/22/2024
    # zeta

    def kernel(xp, xq, theta, debug=False):
        # xp, xq are Nx1,Mx1 vectors (ln(1+xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        BL_kernel = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1
        if debug:
            print(f"BL_kernel = {BL_kernel}")
        # 0.02 was factor here before
        SE_factor = 0.05 * np.exp(-0.5 * (d1 ** 2 / 0.2 ** 2 + d3 ** 2 / 0.3 ** 2))
        SE_kernel = (
            SE_factor
            * soft_relu(1 - soft_abs(xp[1]))
            * soft_relu(1 - soft_abs(xq[1]))  # * np.exp(-0.5 * d3 ** 2 / 9.0)
            * soft_relu(0.3 - xp[3])
            * soft_relu(0.3 - xq[3])  # correlate only low values of gamma together
        )
        if debug:
            print(f"SE kernel = {SE_kernel}")
        gamma_kernel = 1.0 + 0.1 * xp[3] * xq[3]
        if debug:
            print(f"gamma vals = {xp[3]}, {xq[3]}")
        if debug:
            print(f"gamma kernel = {gamma_kernel}")
        xi_kernel = 0.1 * xp[0] * xq[0]
        if debug:
            print(f"xi kernel = {xi_kernel}")

        # now have quadratic gamma kernel
        inner_kernel = (
            BL_kernel * (1.0 + 0.1 * xp[3] * xq[3]) ** 2
            + SE_kernel
            + 0.1 * xp[0] * xq[0]
            + 0.02 * xp[2] * xq[2]
        )
        if debug:
            print(f"inner kernel = {inner_kernel}")
        return inner_kernel

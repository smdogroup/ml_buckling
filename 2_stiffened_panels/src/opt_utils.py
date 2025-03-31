import ml_buckling as mlb
import numpy as np

def gamma_rho_resid_wrapper(
    comm,
    plate_material:mlb.CompositeMaterial,
    stiff_material:mlb.CompositeMaterial,
    nstiff:int,
    stiff_AR:float,
    b:float,
    h:float,
    rho0:float,
    gamma:float,
):    
    target_log_gamma = np.log10(1.0+gamma)
    target_log_rho0 = np.log10(rho0)

    if nstiff > 0:
        def gamma_rho0_resid(x):
            # x is [log10(h_w), log10(AR)]
            h_w = 10**x[0]
            AR = 10**x[1]
            t_w = h_w / stiff_AR

            a = AR * b

            geometry = mlb.StiffenedPlateGeometry(
                a=a, b=b, h=h, num_stiff=nstiff, h_w=h_w, t_w=t_w
            )
            stiff_analysis = mlb.StiffenedPlateAnalysis(
                comm=comm,
                geometry=geometry,
                stiffener_material=stiff_material,
                plate_material=plate_material,
            )

            pred_log_rho0 = np.log10(stiff_analysis.affine_aspect_ratio)
            pred_log_gamma = np.log10(1.0 + stiff_analysis.gamma)
            # scale up gamma resid since tends to under-solve that one
            return [target_log_rho0 - pred_log_rho0, (target_log_gamma - pred_log_gamma)]
    else: # nstiff == 0
        def gamma_rho0_resid(x):
            # x is [log10(h_w), log10(AR)]
            h_w = 10**x[0]
            AR = 10**x[1]
            t_w = h_w / stiff_AR

            a = AR * b

            geometry = mlb.StiffenedPlateGeometry(
                a=a, b=b, h=h, num_stiff=nstiff, h_w=h_w, t_w=t_w
            )
            stiff_analysis = mlb.StiffenedPlateAnalysis(
                comm=comm,
                geometry=geometry,
                stiffener_material=stiff_material,
                plate_material=plate_material,
            )

            pred_log_rho0 = np.log10(stiff_analysis.affine_aspect_ratio)
            pred_log_gamma = np.log10(1.0 + stiff_analysis.gamma)
            return [target_log_rho0 - pred_log_rho0, np.log10(h_w) + 7]
        
    return gamma_rho0_resid
from tools.interpolators import fast_interpolation
import pandas as pd
import math

def compute_aero_forces(velocity, alpha_rad, beta_rad, p_rps, q_rps, r_rps, delta_e_l_deg, delta_e_r_deg, qbar_pa, ac_params):
    '''
    Compute aerodynamic forces and moments based on the current state and aircraft parameters.
    
    Parameters:
    velocity : True airspeed of the aircraft [m/s]
    alpha_rad : Angle of attack [rad]
    beta_rad : Sideslip angle [rad]
    p_rps, q_rps, r_rps: Body roll, pitch, and yaw rates [rad/s]
    delta_e_l_deg, delta_e_r_deg: Left and right elevon deflections [deg] -> Positive = Trailing edge down
    qbar_pa : Dynamic pressure [Pa]
    ac_params : Aircraft parameters (dictionary containing reference geometry and aero tables)
    
    Returns:
    drag_N : Aerodynamic drag force [N]
    side_N : Aerodynamic side force [N]
    lift_N : Aerodynamic lift force [N]
    roll_Nm : Aerodynamic rolling moment [Nm]
    pitch_Nm : Aerodynamic pitching moment [Nm]
    yaw_Nm : Aerodynamic yawing moment [Nm]
    '''
    
    cfd_csv = "databases/cfd_sweep.csv"         # Path to CFD data CSV
    vsp_csv = "databases/vsp_derivatives.csv"   # Path to Open
    
    alpha_deg = math.degrees(alpha_rad)
    delta_e_l = math.radians(delta_e_l_deg)
    delta_e_r = math.radians(delta_e_r_deg)      
    
    if not hasattr(compute_aero_forces, "cfd_data"):
        # Load CFD Alpha Sweep (Expects columns: alpha_deg, CL, CD, Cm)
        compute_aero_forces.cfd_data = pd.read_csv(cfd_csv)
        
        # Load OpenVSP Derivatives (Expects 2 columns: Derivative, Value)
        # Converts it into a fast lookup dictionary
        vsp_df = pd.read_csv(vsp_csv)
        compute_aero_forces.vsp_dict = dict(zip(vsp_df['Derivative'], vsp_df['Value']))

    # Access the cached data
    cfd = compute_aero_forces.cfd_data
    vsp = compute_aero_forces.vsp_dict
    
    # Extract reference geometry
    S = ac_params["S_m2"]  # Wing Area
    b = ac_params["b_m"]   # Wingspan
    c = ac_params["c_m"]   # Mean Aerodynamic Chord
    
    # Elevons
    delta_e = (delta_e_l + delta_e_r) / 2.0  # Symmetric deflection (Pitch)
    delta_a = (delta_e_l - delta_e_r) / 2.0  # Asymmetric deflection (Roll)
    
    # Normalize body rates
    if velocity > 0:
        p_norm = (p_rps * b) / (2.0 * velocity)
        q_norm = (q_rps * c) / (2.0 * velocity)
        r_norm = (r_rps * b) / (2.0 * velocity)
    else:
        p_norm, q_norm, r_norm = 0.0, 0.0, 0.0

    # Interpolate CFD data
    CL_interp = fast_interpolation(cfd['Alpha_deg'].values, cfd['CL'].values, alpha_deg)
    CD_interp = fast_interpolation(cfd['Alpha_deg'].values, cfd['CD'].values, alpha_deg)
    Cm_interp = fast_interpolation(cfd['Alpha_deg'].values, cfd['Cm'].values, alpha_deg)

    # Calculate coefficients
    # Longitudinal (CFD + control + damping)
    CL_tot = CL_interp + (vsp.get("CL_de", 0.0) * delta_e)
    CD_tot = CD_interp 
    Cm_tot = Cm_interp + (vsp.get("Cm_q", 0.0) * q_norm) + (vsp.get("Cm_de", 0.0) * delta_e)

    # Lateral (OpenVSP)
    CY_tot = (vsp.get("CY_beta", 0.0) * beta_rad) + \
             (vsp.get("CY_p", 0.0) * p_norm) + \
             (vsp.get("CY_r", 0.0) * r_norm)
    
    Cl_tot = (vsp.get("Cl_beta", 0.0) * beta_rad) + \
             (vsp.get("Cl_p", 0.0) * p_norm) + \
             (vsp.get("Cl_r", 0.0) * r_norm) + \
             (vsp.get("Cl_da", 0.0) * delta_a)
             
    Cn_tot = (vsp.get("Cn_beta", 0.0) * beta_rad) + \
             (vsp.get("Cn_p", 0.0) * p_norm) + \
             (vsp.get("Cn_r", 0.0) * r_norm) + \
             (vsp.get("Cn_da", 0.0) * delta_a)

    # 5. Dimensionalize into Forces and Moments
    qS = qbar_pa * S
    
    drag_N = qS * CD_tot
    side_N = qS * CY_tot
    lift_N = qS * CL_tot
    
    roll_Nm  = qS * b * Cl_tot
    pitch_Nm = qS * c * Cm_tot
    yaw_Nm   = qS * b * Cn_tot

    return drag_N, side_N, lift_N, roll_Nm, pitch_Nm, yaw_Nm
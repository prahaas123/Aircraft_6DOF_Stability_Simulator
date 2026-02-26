import numpy as np
import math
from tools.interpolators import fast_interpolation
from databases.aero_db import compute_aero_forces
from databases.prop_db import compute_prop_forces

def physics_6dof(time, state, ac_params, atmos_mod):
    '''
    Function for the 6DOF simulation for the aircraft.
        
    Parameters:
    time : Current time in seconds.
    state : Current state vector of the aircraft.
    ac_params : Aircraft parameters.
    
    Returns:
    dx : Time derivative of the state vector.
    '''
    dx = np.zeros(12) # LHS of equations
    
    # States
    u_b_ms = state[0]       # body frame velocity in x direction (forward) [m/s]
    v_b_ms = state[1]       # body frame velocity in y direction (side) [m/s]
    w_b_ms = state[2]       # body frame velocity in z direction (down) [m/s]
    p_b_rps = state[3]      # body frame roll rate [rad/s]
    q_b_rps = state[4]      # body frame pitch rate [rad/s]
    r_b_rps = state[5]      # body frame yaw rate [rad/s]
    phi_rad = state[6]      # roll angle [rad]
    theta_rad = state[7]    # pitch angle [rad]
    psi_rad = state[8]      # yaw angle [rad]
    p1_n_m = state[9]       # inertial frame position in x direction [m]
    p2_n_m = state[10]      # inertial frame position in y direction [m]
    p3_n_m = state[11]      # inertial frame position in z direction [m]
    
    # Pre-computed trig angles
    c_phi = math.cos(phi_rad)
    s_phi = math.sin(phi_rad)
    c_theta = math.cos(theta_rad)
    s_theta = math.sin(theta_rad)
    t_theta = math.tan(theta_rad)
    c_psi = math.cos(psi_rad)
    s_psi = math.sin(psi_rad)
    
    # Aircraft parameters
    m_kg = ac_params["m_kg"]         # mass [kg]
    Jxx_kgm2 = ac_params["Jxx_kgm2"] # moment of inertia around x-axis [kg*m^2]
    Jyy_kgm2 = ac_params["Jyy_kgm2"] # moment of inertia around y-axis [kg*m^2]
    Jzz_kgm2 = ac_params["Jzz_kgm2"] # moment of inertia around z-axis [kg*m^2]
    Jxz_kgm2 = ac_params["Jxz_kgm2"] # product of inertia [kg*m^2]
    
    # Altitude and air properties
    h_m = -p3_n_m  # altitude [m]
    rho_interp_kgm3 = fast_interpolation(atmos_mod["alt_m"], atmos_mod["rho_kgm3"], h_m)
    c_interp_ms = fast_interpolation(atmos_mod["alt_m"], atmos_mod["c_ms"], h_m)
    true_airspeed_ms = math.sqrt(u_b_ms**2 + v_b_ms**2 + w_b_ms**2)
    qbar_pa = 0.5 * rho_interp_kgm3 * true_airspeed_ms**2  # dynamic pressure [Pa]
    
    alpha_rad = math.atan2(w_b_ms, u_b_ms)  # angle of attack [rad]
    beta_rad = math.asin(v_b_ms / true_airspeed_ms) if true_airspeed_ms != 0 else 0.0  # sideslip angle [rad
    s_alpha = math.sin(alpha_rad)
    c_alpha = math.cos(alpha_rad)
    s_beta = math.sin(beta_rad)
    c_beta = math.cos(beta_rad)
        
    # Gravity force
    g = 9.81
    gx_b_mps2 = -g * s_theta
    gy_b_mps2 = g * s_phi * c_theta
    gz_b_mps2 = g * c_phi * c_theta
    
    # Aerodynamic forces and moments
    drag_N, side_N, lift_N, roll_Nm, pitch_Nm, yaw_Nm = compute_aero_forces(true_airspeed_ms, alpha_rad, beta_rad, p_b_rps, q_b_rps, r_b_rps, -5.0, -5.0, qbar_pa, ac_params)
     
    # Propulsion forces and moments
    prop_thrust_N, prop_lift_N, prop_torque_Nm, prop_moment_Nm = compute_prop_forces(true_airspeed_ms, qbar_pa, 100)
    
    # External forces and moments (REVIEW)
    Fx_b_N = -(drag_N * c_alpha * c_beta) - (side_N * c_alpha * s_beta) + (lift_N * s_alpha) + prop_thrust_N
    Fy_b_N = -(drag_N * s_beta) + (side_N * c_beta)
    Fz_b_N = -(drag_N * s_alpha * c_beta) - (side_N * s_alpha * s_beta) - (lift_N * c_alpha) - prop_lift_N
    Mx_b_Nm = roll_Nm + prop_torque_Nm
    My_b_Nm = pitch_Nm + prop_moment_Nm
    Mz_b_Nm = yaw_Nm
    
    # Translational equations
    dx[0] = (Fx_b_N / m_kg) + gx_b_mps2 - (w_b_ms * q_b_rps) + (v_b_ms * r_b_rps)     # du/dt
    dx[1] = (Fy_b_N / m_kg) + gy_b_mps2 - (u_b_ms * r_b_rps) + (w_b_ms * p_b_rps)     # dv/dt
    dx[2] = (Fz_b_N / m_kg) + gz_b_mps2 - (v_b_ms * p_b_rps) + (u_b_ms * q_b_rps)     # dw/dt
    
    # Rotational equations
    den = Jxx_kgm2 * Jzz_kgm2 - Jxz_kgm2**2
    dx[3] = (Jxz_kgm2 * (Jxx_kgm2 - Jyy_kgm2 + Jzz_kgm2) * p_b_rps * q_b_rps - \
            (Jzz_kgm2 * (Jzz_kgm2 - Jyy_kgm2) + Jxz_kgm2**2) * q_b_rps * r_b_rps + \
            (Jzz_kgm2 * Mx_b_Nm) + (Jxz_kgm2 * Mz_b_Nm)) / den                        # dp/dt
    dx[4] = ((Jzz_kgm2 - Jxx_kgm2) * p_b_rps * r_b_rps - \
            (Jxz_kgm2 * (p_b_rps**2 - r_b_rps**2)) + My_b_Nm) / Jyy_kgm2              # dq/dt
    dx[5] = ((Jxx_kgm2 * (Jxx_kgm2 - Jyy_kgm2)  + Jxz_kgm2**2) * p_b_rps * q_b_rps - \
            (Jxz_kgm2 * (Jxx_kgm2 - Jyy_kgm2 + Jzz_kgm2) * q_b_rps * r_b_rps) + \
            (Jxx_kgm2 * Mz_b_Nm) + (Jxz_kgm2 * Mx_b_Nm)) / den                        # dr/dt
    
    # Kinematic equations
    dx[6] = p_b_rps + (s_phi * t_theta * q_b_rps) + \
            (c_phi * t_theta * r_b_rps)                       # dphi/dt
    dx[7] = (c_phi * q_b_rps) - \
            (s_phi * r_b_rps)                                 # dtheta/dt
    dx[8] = (s_phi / c_theta * q_b_rps) + \
            (c_phi / c_theta * r_b_rps)                       # dpsi/dt
    
    # Position equations
    dx[9] = (c_theta * c_psi * u_b_ms) + \
            (((s_phi * s_theta * c_psi) - (c_phi * s_psi)) * v_b_ms) + \
            (((c_phi * s_theta * c_psi) + (s_phi * s_psi)) * w_b_ms)     # dp1/dt
    dx[10] = (c_theta * s_psi * u_b_ms) + \
             (((s_phi * s_theta * s_psi) + (c_phi * c_psi)) * v_b_ms) + \
             (((c_phi * s_theta * s_psi) - (s_phi * c_psi)) * w_b_ms)    # dp2/dt
    dx[11] = (-s_theta * u_b_ms) + \
             ((s_phi * c_theta * v_b_ms) + \
             (c_phi * c_theta * w_b_ms))                                 # dp3/dt
             
    return dx
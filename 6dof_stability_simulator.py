import numpy as np
import pandas as pd
import ussa1976
import math
import matplotlib.pyplot as plt
import flightgear_python
import time

def main():
    # -----------------------------------------------------------------
    # PART 1: SIMULATION SETUP
    # -----------------------------------------------------------------
    
    # Atmospheric model
    atmosphere = ussa1976.compute()
    alt_m = atmosphere["z"].values
    rho_kgm3 = atmosphere["rho"].values
    c_ms = atmosphere["cs"].values
    g_ms2 = 9.81
    atmos_mod = {"alt_m": alt_m, "rho_kgm3": rho_kgm3, "c_ms": c_ms, "g_ms2": g_ms2}

    # Aircraft model
    plane_model = {
        "S_m2": 0.090,        # Wing Area [m^2]
        "b_m": 0.9,           # Wingspan [m]
        "c_m": 0.12,          # Mean Aerodynamic Chord [m]
        "m_kg": 0.4,          # Mass [kg]
        "Jxx_kgm2": 0.25,     # Moment of inertia around x-axis [kg*m^2]
        "Jyy_kgm2": 0.35,     # Moment of inertia around y-axis [kg*m^2]
        "Jzz_kgm2": 0.50,     # Moment of inertia around z-axis [kg*m^2]
        "Jxz_kgm2": 0.001     # Product of inertia [kg*m^2]
    }

    # Initial conditions
    u0_b_ms = 10.0      # body frame velocity in x direction (forward) [m/s]
    v0_b_ms = 0.0       # body frame velocity in y direction (side) [m/s]
    w0_b_ms = 0.0       # body frame velocity in z direction (down) [m/s]
    p0_b_rps = 0.0      # body frame roll rate [rad/s]
    q0_b_rps = 0.0      # body frame pitch rate [rad/s]
    r0_b_rps = 0.0      # body frame yaw rate [rad/s]
    phi0_rad = 0.0      # roll angle [rad]
    theta0_rad = 0.0    # pitch angle [rad]
    psi0_rad = 0.0      # yaw angle [rad]
    p10_n_m = 0.0       # inertial frame position in x direction [m]
    p20_n_m = 0.0       # inertial frame position in y direction [m]
    p30_n_m = -50.0     # inertial frame position in z direction [m]

    x0 = np.array([
        u0_b_ms,
        v0_b_ms,
        w0_b_ms,
        p0_b_rps,
        q0_b_rps,
        r0_b_rps,
        phi0_rad,
        theta0_rad,
        psi0_rad,
        p10_n_m,
        p20_n_m,
        p30_n_m
    ])

    x0 = x0.transpose(); nx0 = x0.size

    # Time conditions
    t0_s = 0.0        # initial time [s]
    tf_s = 30.0       # final time [s]
    h_s = 0.001        # step size [s]

    # -----------------------------------------------------------------
    # PART 2: NUMERICALLY APPROXIMATED SOLUTIONS TO THE EQUATIONS
    # -----------------------------------------------------------------

    # Numerical integrations
    t_s = np.arange(t0_s, tf_s + h_s, h_s); nt_s = t_s.size
    x = np.empty((nx0, nt_s), dtype=float)
    x[:, 0] = x0
    t_s, x = forward_euler(physics_6dof, t_s, x, h_s, plane_model, atmos_mod)

    # True airspeed
    true_airspeed_ms = np.zeros((nt_s, 1))
    for i, element in enumerate(t_s):
        true_airspeed_ms[i, 0] = math.sqrt(x[0, i]**2 + x[1, i]**2 + x[2, i]**2)

    # Atmospheric properties
    altitude_m = np.zeros((nt_s, 1))
    cs_ms = np.zeros((nt_s, 1))
    rho_kgm3 = np.zeros((nt_s, 1))
    for i, element in enumerate(t_s):
        altitude_m[i, 0] = -x[11, i]
        cs_ms[i, 0] = fast_interpolation(atmos_mod["alt_m"], atmos_mod["c_ms"], altitude_m[i, 0])
        rho_kgm3[i, 0] = fast_interpolation(atmos_mod["alt_m"], atmos_mod["rho_kgm3"], altitude_m[i, 0])

    # Angle of attack (Alpha)
    alpha_rad = np.zeros((nt_s, 1))
    for i, element in enumerate(t_s):
        alpha_rad[i, 0] = math.atan2(x[2, i], x[0, i]) 

    # Sideslip angle (Beta)
    beta_rad = np.zeros((nt_s, 1))
    for i, element in enumerate(t_s):
        if x[0, i] == 0 and true_airspeed_ms[i, 0] == 0:
            v_over_vt = 0
        else:
            v_over_vt = x[1, i] / true_airspeed_ms[i, 0]
        beta_rad[i, 0] = math.asin(v_over_vt)

    # Mach number
    mach_number = np.zeros((nt_s, 1))
    for i, element in enumerate(t_s):
        mach_number[i, 0] = true_airspeed_ms[i, 0] / cs_ms[i, 0]

    # -----------------------------------------------------------------
    # PART 3: PLOTTING RESULTS
    # -----------------------------------------------------------------

    rad2deg = 180.0 / math.pi

    fig, axs = plt.subplots(4, 3, figsize=(16, 12))
    fig.suptitle('6-DOF Aircraft Simulation State Vectors', fontsize=16)

    # Translational velocities
    axs[0, 0].plot(t_s, x[0, :], 'b')
    axs[0, 0].set_title('u (Forward Velocity)')
    axs[0, 0].set_ylabel('m/s')

    axs[0, 1].plot(t_s, x[1, :], 'b')
    axs[0, 1].set_title('v (Side Velocity)')
    axs[0, 1].set_ylabel('m/s')

    axs[0, 2].plot(t_s, x[2, :], 'b')
    axs[0, 2].set_title('w (Down Velocity)')
    axs[0, 2].set_ylabel('m/s')

    # Angular rates (deg/s)
    axs[1, 0].plot(t_s, x[3, :] * rad2deg, 'r')
    axs[1, 0].set_title('p (Roll Rate)')
    axs[1, 0].set_ylabel('deg/s')

    axs[1, 1].plot(t_s, x[4, :] * rad2deg, 'r')
    axs[1, 1].set_title('q (Pitch Rate)')
    axs[1, 1].set_ylabel('deg/s')

    axs[1, 2].plot(t_s, x[5, :] * rad2deg, 'r')
    axs[1, 2].set_title('r (Yaw Rate)')
    axs[1, 2].set_ylabel('deg/s')

    # Euler angles (degrees)
    axs[2, 0].plot(t_s, x[6, :] * rad2deg, 'g')
    axs[2, 0].set_title('Phi (Roll Angle)')
    axs[2, 0].set_ylabel('deg')

    axs[2, 1].plot(t_s, x[7, :] * rad2deg, 'g')
    axs[2, 1].set_title('Theta (Pitch Angle)')
    axs[2, 1].set_ylabel('deg')

    axs[2, 2].plot(t_s, x[8, :] * rad2deg, 'g')
    axs[2, 2].set_title('Psi (Yaw Angle)')
    axs[2, 2].set_ylabel('deg')

    # Positions
    axs[3, 0].plot(t_s, x[9, :], 'k')
    axs[3, 0].set_title('North Position')
    axs[3, 0].set_ylabel('m')

    axs[3, 1].plot(t_s, x[10, :], 'k')
    axs[3, 1].set_title('East Position')
    axs[3, 1].set_ylabel('m')

    # Altitude
    axs[3, 2].plot(t_s, -x[11, :], 'k')
    axs[3, 2].set_title('Altitude')
    axs[3, 2].set_ylabel('m')

    # Apply grid and X-axis label to all subplots
    for ax in axs.flat:
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.3, hspace=0.6)
    plt.show()

    # -----------------------------------------------------------------
    # PART 4: FLIGHTGEAR SIMULATION
    # -----------------------------------------------------------------

    # def fdm_callback(fdm_data, event_pipe):
    #     if event_pipe.poll():
    #         state_dict = event_pipe.recv()
    #         fdm_data.lat_rad = state_dict['lat_rad']
    #         fdm_data.lon_rad = state_dict['lon_rad']
    #         fdm_data.alt_m = state_dict['alt_m']
    #         fdm_data.phi_rad = state_dict['phi_rad']
    #         fdm_data.theta_rad = state_dict['theta_rad']
    #         fdm_data.psi_rad = state_dict['psi_rad']
    #     return fdm_data

    # print("Connecting to FlightGear...")
    # fdm_conn = flightgear_python.fg_if.FDMConnection(fdm_version=24)
    # fdm_event_pipe = fdm_conn.connect_rx('localhost', 5501, fdm_callback)
    # fdm_conn.connect_tx('localhost', 5502)
    # fdm_conn.start()

    # R_earth_m = 6378137.0  
    # lat0_rad = math.radians(43.456)
    # lon0_rad = math.radians(-80.383)

    # try:
    #     for i in range(nt_s):
    #         p_north_m = x[9, i]
    #         p_east_m = x[10, i]
    #         current_state = {
    #             'lat_rad': lat0_rad + (p_north_m / R_earth_m),
    #             'lon_rad': lon0_rad + (p_east_m / (R_earth_m * math.cos(lat0_rad))),
    #             'alt_m': -x[11, i],         
    #             'phi_rad': x[6, i],         
    #             'theta_rad': x[7, i],       
    #             'psi_rad': x[8, i]          
    #         }
    #         fdm_event_pipe.send(current_state)
    #         time.sleep(h_s)
    # except KeyboardInterrupt:
    #     pass

    # fdm_conn.stop()
    # print("Playback complete.")

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
    beta_rad = math.asin(v_b_ms / true_airspeed_ms) if true_airspeed_ms != 0 else 0.0  # sideslip angle [rad]
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
    
    # External forces and moments
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

def compute_prop_forces(true_airpseed_ms, qbar_pa, throttle):
    '''
    Computes forces and moments from the propulsion system
    
    Parameters:
    true_airspeed_ms:
    qbar_pa:
    throttle:
    
    Returns
    prop_thrust_N:
    prop_lift_N:
    prop_torque_Nm:
    prop_moment_Nm:
    '''
    prop_thrust_N = 1.4
    prop_lift_N = 0
    prop_torque_Nm = 0
    prop_moment_Nm = 0
    return prop_thrust_N, prop_lift_N, prop_torque_Nm, prop_moment_Nm

def fast_interpolation(x, y, x_new):
    """
    Fast interpolation function for 1D data.

    Parameters:
    x : The x-coordinates of the data points.
    y : The y-coordinates of the data points.
    x_new : The new x-coordinate at which to interpolate.

    Returns:
    The interpolated y-coordinate corresponding to x_new.
    """
    if x_new <= x[0]:
        return y[0]
    elif x_new >= x[-1]:
        return y[-1]
    
    # Binary search to find the correct interval
    low = 0
    high = len(x) - 1
    
    while low <= high:
        mid = (low + high) // 2
        
        if x[mid] < x_new:
            low = mid + 1
        elif x[mid] > x_new:
            high = mid - 1
        else:
            return y[mid]
    
    # Linear interpolation
    x0, x1 = x[high], x[low]
    y0, y1 = y[high], y[low]
    
    return y0 + (y1 - y0) * ((x_new - x0) / (x1 - x0))

def forward_euler(f, t_s, x, h_s, *args):
    """
    Forward Euler method for numerical integration.

    Parameters:
    f : The function representing the RHS of the differential equation, f(t, x).
    t_s : Vector of time points at which to evaluate the solution.
    x : Numerically approximated solution to the DE 'f'.
    h_s : The step sizes for each time interval in seconds.

    Returns:
    t_s : Vector of time points at which the solution is approximated.
    x : The approximated solution at each time point in t_s.
    """
    for i in range(1, len(t_s)):
        x[:, i] = x[:, i-1] + h_s * f(t_s[i-1], x[:, i-1], *args)

    return t_s, x

if __name__ == "__main__":
    main()
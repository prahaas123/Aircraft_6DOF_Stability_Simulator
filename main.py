from tools import numerical_integration
from tools.interpolators import fast_interpolation
from equations import physics_6dof
import numpy as np
import ussa1976
import math
import matplotlib.pyplot as plt
import flightgear_python
import time

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
    "S_m2": 0.5,          # Wing Area [m^2]
    "b_m": 2.0,           # Wingspan [m]
    "c_m": 0.25,          # Mean Aerodynamic Chord [m]
    "m_kg": 5.0,          # Mass [kg]
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
t_s, x = numerical_integration.forward_euler(physics_6dof.physics_6dof, t_s, x, h_s, plane_model, atmos_mod)

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
    if x[0, i] == 0 and x[2, i] == 0:
        w_over_v = 0
    else:
        w_over_v = x[2, i] / x[0, i]
    alpha_rad[i, 0] = math.atan(w_over_v)

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
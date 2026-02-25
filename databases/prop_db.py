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
    prop_thrust_N = 10
    prop_lift_N = 0
    prop_torque_Nm = 0
    prop_moment_Nm = 0
    return prop_thrust_N, prop_lift_N, prop_torque_Nm, prop_moment_Nm
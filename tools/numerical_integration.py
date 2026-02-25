import numpy as np

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
# Aircraft 3-DOF Simulation and Stability Analysis

This repository contains Python-based tools for simulating the static and dynamic stability behavior of aircraft.

## Scripts

### **`3dof_longitudinal_stability.py`**

* A nonlinear 3-DOF flight dynamics simulator focused on longitudinal motion.
* **Trim & Integration:** Automatically searches for approximate trim conditions and uses the RK45 method for time-domain integration.
* **Aerodynamics:** Uses 2D interpolation on the provided aerodynamic databases.
* **Visualization:** Generates matplotlib subplots displaying velocity, angle of attack ($\alpha$), pitch rate, pitch angle ($\theta$), and flight path angle ($\gamma$) over time.

### **`3dof_lateral_stability.py`**

* A nonlinear 3-DOF flight dynamics simulator focused on lateral-directional motion.
* **Time-Domain Integration:** Uses the RK45 method to solve the lateral equations of motion subject to initial disturbances in roll or yaw rates.
* **Aerodynamics:** Interpolates non-dimensional stability derivatives from the provided aerodynamic databases.
* **Visualization:** Generates matplotlib subplots displaying sideslip angle ($\beta$), roll rate, yaw rate, roll angle ($\phi$), and heading ($\psi$).

### **`dynamic_modes.py`**

* A linearized dynamic stability analysis tool.
* **State-Space Matrices:** Constructs the longitudinal and lateral A-matrices based on mass properties, reference geometry, and stability derivatives.
* **Eigenvalue Calculation:** Computes the eigenvalues and eigenvectors to find the natural frequencies and damping ratios of the aircraft's dynamic modes.
* **Mode Approximations:** Calculates classical approximations for the Phugoid, Short Period, Dutch Roll, Roll, and Spiral modes.
* **Root Locus Plotting:** Outputs a root locus plot comparing the full system eigenvalues against the classical approximations.

## `databases/` Directory

This folder contains the CSV input files used by the scripts:

* **`cfd_sweep.csv`**: Contains baseline aerodynamic coefficients (CL, CD, Cm) evaluated across a sweep of AoA ($\alpha$) and airspeeds.
* **`vsp_derivatives.csv`**: Contains non-dimensional stability derivatives exported from OpenVSP/VSPAERO.
* **`prop_db.csv`**: Database containing dynamic propulsion data (thrust and torque).

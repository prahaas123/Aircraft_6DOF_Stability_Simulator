# Aircraft 6-DOF Simulation and Stability Analysis

This repository contains Python-based tools for simulating the static and dynamic stability behaviour of aircraft.

## Scripts

### **`6dof_stability_simulator.py`**

* A full nonlinear 6-DOF flight dynamics simulator.
* **Time-Domain Integration:** Uses a Forward Euler numerical integration to solve the rigid body equations of motion over time.
* **Aerodynamics:** Airspeed, alpha and beta sweeps from a CFD tool, along with stability derivatives, are taken as inputs
* **Visualization:** Automatically generates matplotlib subplots showing translational and angular velocities, Euler angles, and position data. It also includes an optional module visualize the animation via the `flightgear_python` library.

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

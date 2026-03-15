# Aircraft 6-DOF Stability Simulator

A complete 6-Degree-of-Freedom (6-DOF) flight dynamics simulation environment built in Python. This tool helps validate the stability of the conceptual aircraft designs by directly integrating **OpenVSP / STAR-CCM+** aerodynamic databases with a forward Euler numerical physics engine, and visualizing the results in **FlightGear**.

## Key Features

* **6-DOF Physics Engine**: Solves the full non-linear equations of motion for a rigid body in flight, including translational accelerations, rotational rates, and Euler angle kinematics.
* **Numerical Integration**: High-fidelity, stable integration at fast time steps (e.g., 100 Hz / 0.01s) to eliminate numerical divergence.
* **Control Surface Mixing**: Built-in elevon mixing matrix for blended-wing/delta-wing configurations.
* **Real-Time FlightGear Visualization**: Broadcasts telemetry over UDP directly into FlightGear to animate your custom aircraft's flight trajectory in 3D.
* **Comprehensive Data Plotting**: Automatically generates a 4x3 Matplotlib dashboard of all 12 state vectors (velocities, rates, angles, positions).

## Project Structure

```text
Aircraft_6DOF_Stability_Simulator/
├── databases/
│   ├── cfd_sweep.csv           # Raw CFD lift/drag/pitch vs Alpha tables
│   └── vsp_derivatives.csv     # Extracted stability & control derivatives
├── tools/
│   ├── numerical_integration.py # RK4 and Forward Euler math engines
│   └── interpolators.py         # Fast table lookups for aerodynamics
├── equations/
│   └── physics_6dof.py          # The core 6-DOF equations of motion
├── aero_db.py                   # Translates CFD/Derivatives into physical forces (N, Nm)
├── main.py                      # Top-level execution, plotting, and FlightGear sim
└── README.md
```

## Prerequisites & Installation

1. **Python Packages**:
   Install the required libraries via pip:

```bash
pip install numpy pandas matplotlib pyvista flightgear-python ussa1976
```

2. **FlightGear**:
   Download and install [FlightGear](https://www.flightgear.org/) for 3D visualization.

## How to Use

1. **Generate Aerodynamic Data**
If you change your aircraft's geometry or center of gravity, generate the required AoA sweeps and the stability derivatives using any CFD tool of choice. Make sure the databases are `.csv` files and are formatted exactly like `cfd_sweep.csv` and `vsp_derivatives.csv` in the `databases/` folder.

2. **Configure Aircraft Parameters**
Open `main.py` and adjust your aircraft's mass properties and initial conditions:

```python
plane_model = {
    "S_m2": 0.5,          # Wing Area
    "b_m": 2.0,           # Wingspan
    "c_m": 0.25,          # Mean Aerodynamic Chord
    "m_kg": 5.0,          # Mass
    "Jxx_kgm2": 0.25,     # Roll Inertia
    "Jyy_kgm2": 0.35,     # Pitch Inertia
    "Jzz_kgm2": 0.50      # Yaw Inertia
}
```

3. **Launch FlightGear (Listener Mode)**
Before running the Python simulator, open your terminal and launch FlightGear with network FDM enabled. This tells FlightGear to turn off its internal physics and wait for UDP packets from Python on port `5501` and `5502`.

```bash
fgfs --fdm=network,localhost,5501,5502,udp --aircraft=c172p --lat=43.456 --lon=-80.383 --altitude=50
```

4. **Run the Simulator**
With FlightGear waiting on the runway, run the main simulator script:

```bash
python main.py
```

**What happens next:**
1. The forward Euler integrator calculates the entire 10-second flight path instantly.
2. A Matplotlib dashboard pops up showing the 12 state variables over time.
3. In the background, Python connects to FlightGear and plays back the trajectory in real-time.

## Coordinate Systems
* **Positions (Flat Earth):** North, East, Down (NED).
* **Body Rates:** $p$ (roll), $q$ (pitch), $r$ (yaw) in radians/second.
* **Euler Angles:** $\phi$ (roll), $\theta$ (pitch), $\psi$ (yaw) following standard aerospace conventions.
* **OpenVSP Derivatives:** Kept in standard non-dimensional *per radian* format for seamless integration with the simulator's body rate normalization.

## Usage
Feel free to use this simulator, but always give credit to the creator.

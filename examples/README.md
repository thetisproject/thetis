# Thetis Examples

This directory contains self-contained example simulations showcasing different Thetis capabilities—from 2D 
shallow-water flows to 3D baroclinic dynamics, sediment transport, tracer cases, and inverse modeling.

Each subfolder includes a Python script (e.g. `channel2d.py`), which you can run directly or in parallel.

---

## 📚 Overview of Examples

| Folder                   | Short Description                                                                   |
|--------------------------|-------------------------------------------------------------------------------------|
| **balzano**              | Classic wetting–drying test case with sloping bathymetry and periodic free surface. |
| **baroclinic_channel**   | 3D baroclinic channel flow with stratification.                                     |
| **baroclinic_eddies**    | Simulation of baroclinic eddies in a channel.                                       |
| **bottomFriction**       | Tests bottom friction formulations in shallow-water flow.                           |
| **channel2d**            | 2D channel flow; useful for basic shallow-water tests.                              |
| **channel3d**            | 3D version of channel flow with vertical structure.                                 |
| **channel_inversion**    | Calibration of bathymetry/Manning coefficient/forcings using channel observations.  |
| **columbia_plume**       | Oregon coastal plume with realistic river discharge.                                |
| **cylinder_eddies**      | Flow around a cylinder, illustrating eddy shedding.                                 |
| **discrete_turbines**    | Simulating flow interaction with tidal turbines.                                    |
| **dome**                 | Idealized dome-shaped bathymetry to test internal waves.                            |
| **freshwaterCylinder**   | Freshwater release into denser ambient fluid around cylinder.                       |
| **geostrophicGyre**      | Steady geostrophic circulation in a closed basin.                                   |
| **headland_inversion**   | Calibration of bottom friction around a headland using velocity observations.       |
| **idealizedEstuary**     | Idealized estuarine flow with salt intrusion.                                       |
| **katophillips**         | Shows implementation of Kato-Phillips internal wave closure.                        |
| **lockExchange**         | Classic lock-exchange buoyant flow test.                                            |
| **nonhydrostatic_cases** | Non-hydrostatic shallow-water wave examples.                                        |
| **north_sea**            | Simplified North Sea tidal model.                                                   |
| **overflow**             | Dense overflow flow (e.g., sinking plume).                                          |
| **reaction**             | Advection-reaction example case.                                                    |
| **rhineROFI**            | Rhône River Estuary example (ROFI).                                                 |
| **sediment_meander_2d**  | Sediment transport in a 2D meandering channel.                                      |
| **sediment_trench_2d**   | Sediment resuspension over a trench.                                                |
| **stommel2d**            | Classic 2D Stommel gyre model.                                                      |
| **stommel3d**            | 3D extension of the Stommel gyre.                                                   |
| **tidal_barrage**        | Tidal barrage with flow-through structure.                                          |
| **tidalfarm**            | Optimisation of an array of turbines in tidal flow.                                 |
| **tohoku_inversion**     | Calibration (inversion) of model for Tōhoku tsunami event.                          |
| **tracerBox**            | Passive tracer in a box flow test.                                                  |
| **waveEq2d**             | 2D linear wave equation solver.                                                     |
| **waveEq3d**             | 3D extension of the wave equation solver.                                           |

---

## 🚀 Running an Example

```bash  
  cd channel2d
  python channel2d.py
  # or for parallel MPI runs:
  mpiexec -n 4 python channel2d.py
```

---


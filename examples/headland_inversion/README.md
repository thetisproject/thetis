# Headland Channel - Friction Field Calibration

## Forward run

The forward run is not included here, instead, only the time series `.hdf5` files at each station are provided. 

The forward run used to generate this uses the same model configuration as provided by `model_config.py` which 
configures the `solver object`, but with a friction field based on a sea bed particle sizes as shown below:

![Sea Bed Particle Sizes](images/seabed_classification.png)

The idealised headland is 20km long and 6km wide, with a coastline depth of 3m and a main channel depth of 40m. The left 
and right boundaries are forced by a sinusoidal elevation function, emulating a single tidal signal. A viscosity sponge 
is used at the left hand boundary to provide some model stability.

## Inversion run

### Full Nodal Flexibility

```sh
source ~/firedrake/bin/activate
make invert CASE=NodalFreedom
```

The inversion problem is currently run from a `Makefile`. User arguments are specified here i.e. fields to optimise 
(just Manning for this example) and then the Makefile runs the scripts with the inputs provided.

The solver object is set up using `construct_solver` and then initial values for each field (in this case we only 
optimise for bed friction) are specified. The station manager, `StationObservationManager` , is then defined, which is 
from the modified version of the inversion tools provided by Thetis. In this case, as the calibration points have been 
defined by a forward run, we can use `load_observation_data` to interpolate the observation time series to the model 
time and also stores the time series data to disk. The station manager will also include the variable that is being 
optimised for, in this case, the velocity components.

We can then set up the inversion manager, `InversionManager`, again from the modified version of the inversion tools of 
Thetis. The station manager is the first argument and the two key other arguments are the penalty parameters and cost 
scaling. The cost function is regularised by an additional term, in this case, the Hessian (second derivative) of the 
elevation field. This prevents overfitting of the Manning field i.e. having a highly variable field. The penalty 
parameters are the regularisation parameters that control the strength of the regularisation. Higher values increase the 
weight of the penalty term, leading to a smoother friction field, while lower values allow more variability. The cost 
scaling normalises the regularisation term by the local mesh element size, so that the degree of penalization adapts to 
the local mesh resolution. In regions with finer mesh resolution, the scaling ensures that higher variability in 
friction is allowed, whilst in regions of lower resolution less variability is allowed to prevent overfitting to sparse 
data points. The cost function is then defined using the inversion manager, which is the Hessian regularised L2 norm. 
The actual class, `HessianRecoverer2D`, for calculating this loss can be found in `thetis.diagnostics.py`.

The forward model is then run (`solver_obj.iterate`) with the cost function embedded via a callback, which is an 
important step for the adjoint. Passing the cost function callback effectively embeds the dependency of the model state 
on the control variables into the cost function, forming the reduced functional. The reduced functional is thus the 
original cost function plus the regularization term, modified by the model equations. Running the model gives us the 
baseline cost and more importantly, calculates the gradients of the cost function with respect to the control variables 
(friction) which are fed into the adjoint method. The annotation process is then stopped, which has been recording the 
computations related to the cost function and its derivatives.

Optimisation parameters are then defined, which are the maximum number of iterations and the tolerance for the 
optimisation convergence criterion (threshold for the relative change in the cost function, below which the optimisation 
process will terminate). The optimisation is run by calling `inv_manager.minimize`. The L-BFGS-B algorithm is used as it
is suitable for bound-constrained problems, which are specified earlier alongside the penalty parameters for the cost 
functional. These are the minimum and maximum values of bed friction allowed.

The remainder of the script performs file saving and preparation for visualisation in ParaView.

### Constant Bed Friction

```sh
source ~/firedrake/bin/activate
make invert CASE=Constant
```

For a constant bed friction, there are some differences which are enforced by changing the case entry, as explained 
below.

Firstly, we do not need penalty parameters in the inversion problem as we will not have any variation across the field
and thus there is no smoothing required. Now, instead of adding the Manning field as a `Control`, we will define the 
friction through a `Constant`, and then project this `Constant` onto the Manning field. This `Constant` then becomes our
`Control`, and as it is a `Constant`, it cannot vary. This is inherently dealt with by `Firedrake` and `pyadjoint`. 

The standard implementation of `InversionManager` in Thetis is not flexible in dealing with the `Control` not being 
a `Function` i.e. the Manning, bathymetry etc., so we this is where modifications start to come in. 
We need to export the Manning at each iteration, so we need to extract the mesh from the `StationObservationManager`, 
extract the function space and create a `Function` to assign the `Control` value to. This can then be exported to `.vtu` 
at each iteration.

### Region Based Bed Friction

```sh
source ~/firedrake/bin/activate
make invert CASE=Regions
```

For region-based bed friction, the `InversionManager` has again been modified so that we can export things 
correctly. Here, we need to create a mapping that relates the Manning values to the regions of the mesh. Again, there is 
no need for penalising the Hessian. We also need to use a `Constant` for each area, and each `Constant` is one 
`Control`. The adjoint works on graphical connections, so we need to ensure that the adjoint process can back-propagate 
through the various layers to our `Controls`. This is why we need to store our values as `Functions` on our `mesh` 
`FunctionSpace`. 

We can define four regions, for example, in the following manner:

```
mask_values = [np.logical_and(x < 50e3, y < 6e3).astype(float),
               np.logical_and(x < 50e3, y >= 6e3).astype(float),
               np.logical_and(x >= 50e3, y < 6e3).astype(float),
               np.logical_and(x >= 50e3, y >= 6e3).astype(float)]
               
# Create Function objects to store the coefficients
m_true = [Constant(i+1., domain=mesh2d) for i in range(len(mask_values))]
masks = [Function(V) for _ in range(len(m_true))]

# Assign the mask values to the Functions
for i, mask_ in enumerate(masks):
    mask_.dat.data[:] = mask_values[i]
```

Importantly, the masks i.e. our regions, will remain consistent. The only thing that will change will be the values 
associated with each mask. This means we can define each mask by assigning its values from `NumPy` operators using 
`mask.dat.data[:] = mask_values[i]`. In the case of bed particle size mapping, this is important, because it would be 
challenging to have to define a series of `conditional` or other operators to define each area. Note that if we 
included this assignment of the masks at each iteration of the adjoint, it would not work as there is no graphical 
connection when we do assignments with `function.dat.data[:] = values`. We can then define our mapping function as 
follows:

```
def update_n(n, m):
    # Reset n to zero
    n.assign(0)
    # Add the weighted masks to n
    for m_, mask_ in zip(m, masks):
        n += m_ * mask_
```

We then iterate through each mask (region) and then add the corresponding value of Manning (n) friction. We now have a 
mapping that `PyAdjoint` can understand.

As for the `InversionManager`, we can then provide this mapping `update_n` function which allows us to export the 
`Control` and `Gradient` fields correctly, rather than having `m` outputs for `m` controls. We can then run the forward,
inverse and plotting scripts in order.

### Independent Point Scheme

```sh
source ~/firedrake/bin/activate
make invert CASE=IndependentPointsScheme
```

The independent point scheme approach works in the same way as the region-based approach, where we have a mapping 
function which tells us how the Manning field changes with respect to our input independent point values. We can use the
same `InversionManager` updates, and the only thing we need to do is change the masks we generate. For a linear 
interpolation of the points, this mapping is generated as follows:

```
# Get domain limits, define independent points and specify their values
lx, ly = np.max(x), np.max(y)
points = [(lx/4, ly/4), (lx/4, 3*ly/4), (lx/2, ly/4), (lx/2, 3*ly/4),
          (3*lx/4, ly/4), (3*lx/4, 3*ly/4)]
m_true = [Constant(0.01*(i+1), domain=mesh2d) for i in range(len(points))]
M = len(m_true)

# Use Python's numpy to create arrays for the interpolation points
interp_x = np.array([p[0] for p in points])
interp_y = np.array([p[1] for p in points])
points = np.column_stack((interp_x, interp_y))

# Create the interpolators, use nearest neighbour interpolation outside the convex 
# hull of the linear interpolator
linear_interpolator = LinearNDInterpolator(points, np.eye(len(points)))
nearest_interpolator = NearestNDInterpolator(points, np.eye(len(points)))

# Apply the interpolators to the mesh coordinates to get linear coefficients 
# (these do not depend on the magnitude of the points)
linear_coefficients = linear_interpolator(coordinates)
nan_mask = np.isnan(linear_coefficients).any(axis=1)
linear_coefficients[nan_mask] = nearest_interpolator(coordinates[nan_mask])

# Create Function objects to store the coefficients
masks = [Function(V) for _ in range(len(points))]

# Assign the linear coefficients to the masks
for i, mask in enumerate(masks):
    mask.dat.data[:] = linear_coefficients[:, i]
```

Now, instead of masks with 0/1 values, we have masks which describe the contribution of each point to the rest of the 
domain. Note that this will only work for linear interpolation, as we cannot generate static coefficients for non-linear
mappings (RBF, quadratic, cubic etc.). In those cases, we would need to 'annotate' the interpolation functions for 
`PyAdjoint` to track the gradient through. 

## Post-processing

```sh
source ~/firedrake/bin/activate
make plot CASE=NodalFreedom
make plot CASE=Constant
make plot CASE=Regions
make plot CASE=IndependentPointsScheme
```

To plot the progress, we can use the Makefile to run `plot_velocity_progress.py`. This plots the velocity over time at 
each of the station locations for each iteration of the optimisation, relative to the ground truth from the forward run.

## Running in parallel

The default settings run these scripts in parallel, however we can leverage parallel processing to accelerate the 
simulations by partioning the mesh. To do so, simply provide the number of processors you would like to use after the 
PARALLEL option, e.g.:

```sh
source ~/firedrake/bin/activate
make invert CASE=IndependentPointsScheme PARALLEL=4
```

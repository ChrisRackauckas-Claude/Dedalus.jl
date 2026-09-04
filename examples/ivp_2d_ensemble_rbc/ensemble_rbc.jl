"""
Dedalus script simulating an ensemble of 2D horizontally-periodic Rayleigh-Benard
convection simulations with 10 different random initial conditions. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_ensemble.jl` script can be used to
produce plots from the saved data. It should take about 20 cpu-minutes to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)^(-1/2)
    nu = (Rayleigh / Prandtl)^(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    \$ mpiexec -n 4 julia ensemble_rbc.jl
    \$ mpiexec -n 4 julia plot_ensemble.jl snapshots/*.h5
"""

using Dedalus
using Logging

logger = Logging.current_logger()

# Parameters
Lx, Lz = 4, 1
Nx, Nz = 256, 64
Rayleigh = 2e6
Prandtl = 1
dealias = 3/2
stop_sim_time = 50
timestepper = RK222
max_timestep = 0.125
dtype = Float64

# Bases
ncoord = Coordinate("n")
coords = CartesianCoordinates("x", "z")
dist = Distributor((ncoord, coords); dtype=dtype)
nbasis = CardinalBasis(ncoord; size=10)
xbasis = RealFourier(coords["x"], Nx; bounds=(0, Lx), dealias=dealias)
zbasis = ChebyshevT(coords["z"], Nz; bounds=(0, Lz), dealias=dealias)

# Fields
p = Field(dist; name="p", bases=(nbasis, xbasis, zbasis))
b = Field(dist; name="b", bases=(nbasis, xbasis, zbasis))
u = VectorField(dist, coords; name="u", bases=(nbasis, xbasis, zbasis))
tau_p = Field(dist; name="tau_p", bases=(nbasis,))
tau_b1 = Field(dist; name="tau_b1", bases=(nbasis, xbasis))
tau_b2 = Field(dist; name="tau_b2", bases=(nbasis, xbasis))
tau_u1 = VectorField(dist, coords; name="tau_u1", bases=(nbasis, xbasis))
tau_u2 = VectorField(dist, coords; name="tau_u2", bases=(nbasis, xbasis))

# Substitutions
kappa = (Rayleigh * Prandtl)^(-1/2)
nu = (Rayleigh / Prandtl)^(-1/2)
x, z = local_grids(dist, xbasis, zbasis)
ex, ez = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
lift = A -> Lift(A, lift_basis, -1)
grad_func = A -> gradient(A, coords)
grad_u = grad_func(u) + ez * lift(tau_u1)  # First-order reduction
grad_b = grad_func(b) + ez * lift(tau_b1)  # First-order reduction
integ_func = A -> integrate(A, coords)

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2]; namespace=@locals)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
add_equation!(problem, "dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
add_equation!(problem, "b(z=0) = Lz")
add_equation!(problem, "u(z=0) = 0")
add_equation!(problem, "b(z=Lz) = 0")
add_equation!(problem, "u(z=Lz) = 0")
add_equation!(problem, "integ(p) = 0") # Pressure gauge

# Solver
solver = build_solver(problem, timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3) # Random noise
b["g"] .*= z .* (Lz .- z) # Damp noise at walls
b["g"] .+= Lz .- z # Add linear background

# Analysis
vorticity = -divergence(skew(u))
snapshots = add_file_handler(solver.evaluator, "snapshots"; sim_dt=0.25, max_writes=50)
add_task!(snapshots, b; name="buoyancy")
add_task!(snapshots, vorticity; name="vorticity")
add_task!(snapshots, average(b, "n"); name="ensemble buoyancy")
add_task!(snapshots, average(vorticity, "n"); name="ensemble vorticity")

# CFL
cfl = CFL(solver; initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
          max_change=1.5, min_change=0.5, max_dt=max_timestep)
add_velocity!(cfl, u)

# Flow properties
flow = GlobalFlowProperty(solver; cadence=10)
add_property!(flow, sqrt(u * u) / nu; name="Re")

# Main loop
try
    @info "Starting main loop"
    while solver.proceed
        timestep = compute_timestep(cfl)
        step!(solver, timestep)
        if (solver.iteration - 1) % 10 == 0
            max_Re = flow_max(flow, "Re")
            @info "Iteration=$(solver.iteration), Time=$(solver.sim_time), dt=$(timestep), max(Re)=$(max_Re)"
        end
    end
catch e
    @error "Exception raised, triggering end of main loop."
    rethrow(e)
finally
    log_stats(solver)
end

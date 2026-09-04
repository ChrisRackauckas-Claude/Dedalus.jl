"""
Dedalus script simulating 2D centrifugal convection in an annulus. This script
demonstrates solving an initial value problem in the annulus. It can be ran serially
or in parallel, and uses the built-in analysis framework to save data snapshots to
HDF5 files. The `plot_polar.jl` and `plot_scalars.jl` scripts can be used to produce
plots from the saved data. The simulation should take roughly 10 cpu-minutes to run.

The problem is non-dimensionalized using the mean radius L = (Ri + Ro)/2 and the
freefall time, so the resulting thermal diffusivity and viscosity are related to the
Prandtl and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)^(-1/2)
    nu = (Rayleigh / Prandtl)^(-1/2)

The radii ratio is given by eta = Ro/Ri. Since the problem is 2D, the Coriolis force
and Rossby number drop out of the problem.

For incompressible hydro in the annulus, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    \$ mpiexec -n 4 julia centrifugal_convection.jl
    \$ mpiexec -n 4 julia plot_polar.jl snapshots/*.h5
    \$ julia plot_scalars.jl scalars/*.h5
"""

using Dedalus
using Logging

logger = Logging.current_logger()

# Parameters
Nphi, Nr = 256, 64
eta = 3
Rayleigh = 1e6
Prandtl = 1
dealias = 3/2
stop_sim_time = 30
timestepper = RK222
max_timestep = 0.125
safety = 0.5
dtype = Float64

# Derived parameters
Ri = 2 / (1 + eta)
Ro = 2 * eta / (1 + eta)

# Bases
coords = PolarCoordinates("phi", "r")
dist = Distributor(coords; dtype=dtype)
annulus = AnnulusBasis(coords; shape=(Nphi, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
annulus_edge = outer_edge(annulus)

# Fields
p = Field(dist; name="p", bases=(annulus,))
b = Field(dist; name="b", bases=(annulus,))
u = VectorField(dist, coords; name="u", bases=(annulus,))
tau_p = Field(dist; name="tau_p")
tau_b1 = Field(dist; name="tau_b1", bases=(annulus_edge,))
tau_b2 = Field(dist; name="tau_b2", bases=(annulus_edge,))
tau_u1 = VectorField(dist, coords; name="tau_u1", bases=(annulus_edge,))
tau_u2 = VectorField(dist, coords; name="tau_u2", bases=(annulus_edge,))

# Substitutions
kappa = (Rayleigh * Prandtl)^(-1/2)
nu = (Rayleigh / Prandtl)^(-1/2)
phi, r = local_grids(dist, annulus)
rvec = VectorField(dist, coords; bases=(radial_basis(annulus),))
rvec["g"][2] = r
lift_basis = derivative_basis(annulus, 1)
lift = A -> Lift(A, lift_basis, -1)
grad_u = gradient(u) + rvec * lift(tau_u1)  # First-order reduction
grad_b = gradient(b) + rvec * lift(tau_b1)  # First-order reduction
g = rvec * 2 * (eta - 1) / (eta + 1)

# Problem
problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2]; namespace=@locals)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
add_equation!(problem, "dt(u) - nu*div(grad_u) + grad(p) + b*g + lift(tau_u2) = - u@grad(u)")
add_equation!(problem, "b(r=Ri) = 0")
add_equation!(problem, "u(r=Ri) = 0")
add_equation!(problem, "b(r=Ro) = 1")
add_equation!(problem, "u(r=Ro) = 0")
add_equation!(problem, "integ(p) = 0") # Pressure gauge

# Solver
solver = build_solver(problem, timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3) # Random noise
b["g"] .*= (r .- Ri) .* (Ro .- r) # Damp noise at walls
b["g"] .+= log.(r ./ Ri) ./ log(Ro / Ri) # Add conductive background

# Analysis
snapshots = add_file_handler(solver.evaluator, "snapshots"; sim_dt=0.1, max_writes=20)
add_task!(snapshots, -divergence(skew(u)); name="vorticity")
add_task!(snapshots, b; name="buoyancy")
scalars = add_file_handler(solver.evaluator, "scalars"; sim_dt=0.01)
add_task!(scalars, integrate(0.5 * DotProduct(u, u)); name="KE")

# CFL
cfl = CFL(solver; initial_dt=max_timestep, max_dt=max_timestep, safety=safety,
          cadence=10, threshold=0.1, max_change=1.5, min_change=0.5)
add_velocity!(cfl, u)

# Flow properties
flow = GlobalFlowProperty(solver; cadence=10)
add_property!(flow, sqrt(DotProduct(u, u)) / nu; name="Re")

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

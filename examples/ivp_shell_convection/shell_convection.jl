"""
Dedalus script simulating Boussinesq convection in a spherical shell. This script
demonstrates solving an initial value problem in the shell. It can be ran serially
or in parallel, and uses the built-in analysis framework to save data snapshots
to HDF5 files. The `plot_shell.jl` script can be used to produce plots from the
saved data. The simulation should take about 20 cpu-minutes to run.

The problem is non-dimensionalized using the shell thickness and freefall time, so
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
    \$ mpiexec -n 4 julia shell_convection.jl
    \$ mpiexec -n 4 julia plot_shell.jl snapshots/*.h5
"""

using Dedalus
using Logging

logger = Logging.current_logger()

# Parameters
Ri, Ro = 14, 15
Nphi, Ntheta, Nr = 192, 96, 6
Rayleigh = 3500
Prandtl = 1
dealias = 3/2
stop_sim_time = 2000
timestepper = SBDF2
max_timestep = 1
dtype = Float64
mesh = nothing

# Bases
coords = SphericalCoordinates("phi", "theta", "r")
dist = Distributor(coords; dtype=dtype, mesh=mesh)
shell = ShellBasis(coords; shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
sphere = outer_surface(shell)

# Fields
p = Field(dist; name="p", bases=(shell,))
b = Field(dist; name="b", bases=(shell,))
u = VectorField(dist, coords; name="u", bases=(shell,))
tau_p = Field(dist; name="tau_p")
tau_b1 = Field(dist; name="tau_b1", bases=(sphere,))
tau_b2 = Field(dist; name="tau_b2", bases=(sphere,))
tau_u1 = VectorField(dist, coords; name="tau_u1", bases=(sphere,))
tau_u2 = VectorField(dist, coords; name="tau_u2", bases=(sphere,))

# Substitutions
kappa = (Rayleigh * Prandtl)^(-1/2)
nu = (Rayleigh / Prandtl)^(-1/2)
phi, theta, r = local_grids(dist, shell)
er = VectorField(dist, coords; bases=(radial_basis(shell),))
er["g"][3] = 1
rvec = VectorField(dist, coords; bases=(radial_basis(shell),))
rvec["g"][3] = r
lift_basis = derivative_basis(shell, 1)
lift = A -> Lift(A, lift_basis, -1)
grad_u = gradient(u) + rvec * lift(tau_u1)  # First-order reduction
grad_b = gradient(b) + rvec * lift(tau_b1)  # First-order reduction

# Problem
problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2]; namespace=@locals)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
add_equation!(problem, "dt(u) - nu*div(grad_u) + grad(p) - b*er + lift(tau_u2) = - u@grad(u)")
add_equation!(problem, "b(r=Ri) = 1")
add_equation!(problem, "u(r=Ri) = 0")
add_equation!(problem, "b(r=Ro) = 0")
add_equation!(problem, "u(r=Ro) = 0")
add_equation!(problem, "integ(p) = 0") # Pressure gauge

# Solver
solver = build_solver(problem, timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3) # Random noise
b["g"] .*= (r .- Ri) .* (Ro .- r)  # Damp noise at walls
b["g"] .+= (Ri .- Ri .* Ro ./ r) ./ (Ri - Ro)  # Add linear background

# Analysis
flux = er * (-kappa * gradient(b) + u * b)
snapshots = add_file_handler(solver.evaluator, "snapshots"; sim_dt=10, max_writes=10)
add_task!(snapshots, b(r=(Ri+Ro)/2); scales=dealias, name="bmid")
add_task!(snapshots, flux(r=Ro); scales=dealias, name="flux_r_outer")
add_task!(snapshots, flux(r=Ri); scales=dealias, name="flux_r_inner")
add_task!(snapshots, flux(phi=0); scales=dealias, name="flux_phi_start")
add_task!(snapshots, flux(phi=3*pi/2); scales=dealias, name="flux_phi_end")

# CFL
cfl = CFL(solver; initial_dt=max_timestep, cadence=10, safety=2, threshold=0.1,
          max_change=1.5, min_change=0.5, max_dt=max_timestep)
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

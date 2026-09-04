"""
Dedalus script simulating internally-heated Boussinesq convection in the ball.
This script demonstrates solving an initial value problem in the ball. It can be
ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_ball.jl` script can be used to produce
plots from the saved data. The simulation should take roughly 30 cpu-minutes to run.

The strength of gravity is proportional to radius, as for a constant density ball.
The problem is non-dimensionalized using the ball radius and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)^(-1/2)
    nu = (Rayleigh / Prandtl)^(-1/2)

We use stress-free boundary conditions, and maintain a constant flux on the outer
boundary. The convection is driven by the internal heating term with a conductive
equilibrium of T(r) = 1 - r^2.

For incompressible hydro in the ball, we need one tau term each for the velocity
and temperature. Here we choose to lift them to the original (k=0) basis.

The simulation will run to t=20, about the time for the first convective plumes
to hit the top boundary. After running this initial simulation, you can run the
simulation for an additional 20 time units with the command line option '--restart'.

To run, restart, and plot using e.g. 4 processes:
    \$ mpiexec -n 4 julia internally_heated_convection.jl
    \$ mpiexec -n 4 julia internally_heated_convection.jl --restart
    \$ mpiexec -n 4 julia plot_ball.jl slices/*.h5
"""

using Dedalus
using Logging

logger = Logging.current_logger()

# Allow restarting via command line
restart = length(ARGS) > 0 && ARGS[1] == "--restart"

# Parameters
Nphi, Ntheta, Nr = 128, 64, 96
Rayleigh = 1e6
Prandtl = 1
dealias = 3/2
stop_sim_time = 20 + 20 * restart
timestepper = SBDF2
max_timestep = 0.05
dtype = Float64
mesh = nothing

# Bases
coords = SphericalCoordinates("phi", "theta", "r")
dist = Distributor(coords; dtype=dtype, mesh=mesh)
ball = BallBasis(coords; shape=(Nphi, Ntheta, Nr), radius=1, dealias=dealias, dtype=dtype)
sphere = surface(ball)

# Fields
u = VectorField(dist, coords; name="u", bases=(ball,))
p = Field(dist; name="p", bases=(ball,))
T = Field(dist; name="T", bases=(ball,))
tau_p = Field(dist; name="tau_p")
tau_u = VectorField(dist, coords; name="tau_u", bases=(sphere,))
tau_T = Field(dist; name="tau_T", bases=(sphere,))

# Substitutions
phi, theta, r = local_grids(dist, ball)
r_vec = VectorField(dist, coords; bases=(radial_basis(ball),))
r_vec["g"][3] = r
T_source = 6
kappa = (Rayleigh * Prandtl)^(-1/2)
nu = (Rayleigh / Prandtl)^(-1/2)
lift = A -> Lift(A, ball, -1)
strain_rate = gradient(u) + transpose_components(gradient(u))
shear_stress = angular_component(radial_component(strain_rate(r=1); index=1))

# Problem
problem = IVP([p, u, T, tau_p, tau_u, tau_T]; namespace=@locals)
add_equation!(problem, "div(u) + tau_p = 0")
add_equation!(problem, "dt(u) - nu*lap(u) + grad(p) - r_vec*T + lift(tau_u) = - cross(curl(u),u)")
add_equation!(problem, "dt(T) - kappa*lap(T) + lift(tau_T) = - u@grad(T) + kappa*T_source")
add_equation!(problem, "shear_stress = 0")  # Stress free
add_equation!(problem, "radial(u(r=1)) = 0")  # No penetration
add_equation!(problem, "radial(grad(T)(r=1)) = -2")
add_equation!(problem, "integ(p) = 0")  # Pressure gauge

# Solver
solver = build_solver(problem, timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if !restart
    fill_random!(T, "g"; seed=42, distribution="normal", scale=0.01) # Random noise
    low_pass_filter!(T; scales=0.5)
    T["g"] .+= @. 1 - r^2 # Add equilibrium state
    file_handler_mode = "overwrite"
    initial_timestep = max_timestep
else
    write, initial_timestep = load_state!(solver, "checkpoints/checkpoints_s20.h5")
    initial_timestep = 2e-2
    file_handler_mode = "append"
end

# Analysis
slices = add_file_handler(solver.evaluator, "slices"; sim_dt=0.1, max_writes=10, mode=file_handler_mode)
add_task!(slices, T(phi=0); scales=dealias, name="T(phi=0)")
add_task!(slices, T(phi=pi); scales=dealias, name="T(phi=pi)")
add_task!(slices, T(phi=3/2*pi); scales=dealias, name="T(phi=3/2*pi)")
add_task!(slices, T(r=1); scales=dealias, name="T(r=1)")
checkpoints = add_file_handler(solver.evaluator, "checkpoints"; sim_dt=1, max_writes=1, mode=file_handler_mode)
add_tasks!(checkpoints, solver.state)

# CFL
cfl = CFL(solver; initial_dt=restart ? initial_timestep : max_timestep,
          cadence=10, safety=0.5, threshold=0.1, max_dt=max_timestep)
add_velocity!(cfl, u)

# Flow properties
flow = GlobalFlowProperty(solver; cadence=10)
add_property!(flow, DotProduct(u, u); name="u2")

# Main loop
try
    @info "Starting main loop"
    while solver.proceed
        timestep = compute_timestep(cfl)
        step!(solver, timestep)
        if (solver.iteration - 1) % 10 == 0
            max_u = sqrt(flow_max(flow, "u2"))
            @info "Iteration=$(solver.iteration), Time=$(solver.sim_time), dt=$(timestep), max(u)=$(max_u)"
        end
    end
catch e
    @error "Exception raised, triggering end of main loop."
    rethrow(e)
finally
    log_stats(solver)
end

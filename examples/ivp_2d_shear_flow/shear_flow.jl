"""
Dedalus script simulating a 2D periodic incompressible shear flow with a passive
tracer field for visualization. This script demonstrates solving a 2D periodic
initial value problem. It can be ran serially or in parallel, and uses the
built-in analysis framework to save data snapshots to HDF5 files. The
`plot_snapshots.jl` script can be used to produce plots from the saved data.
The simulation should take about 10 cpu-minutes to run.

The initial flow is in the x-direction and depends only on z. The problem is
non-dimensionalized using the shear-layer spacing and velocity jump, so the
resulting viscosity and tracer diffusivity are related to the Reynolds and
Schmidt numbers as:

    nu = 1 / Reynolds
    D = nu / Schmidt

To run and plot using e.g. 4 processes:
    \$ mpiexec -n 4 julia shear_flow.jl
    \$ mpiexec -n 4 julia plot_snapshots.jl snapshots/*.h5
"""

using Dedalus
using Logging

logger = Logging.current_logger()

# Parameters
Lx, Lz = 1, 2
Nx, Nz = 128, 256
Reynolds = 5e4
Schmidt = 1
dealias = 3/2
stop_sim_time = 20
timestepper = RK222
max_timestep = 1e-2
dtype = Float64

# Bases
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=dtype)
xbasis = RealFourier(coords["x"], Nx; bounds=(0, Lx), dealias=dealias)
zbasis = RealFourier(coords["z"], Nz; bounds=(-Lz/2, Lz/2), dealias=dealias)

# Fields
p = Field(dist; name="p", bases=(xbasis, zbasis))
s = Field(dist; name="s", bases=(xbasis, zbasis))
u = VectorField(dist, coords; name="u", bases=(xbasis, zbasis))
tau_p = Field(dist; name="tau_p")

# Substitutions
nu = 1 / Reynolds
D = nu / Schmidt
x, z = local_grids(dist, xbasis, zbasis)
ex, ez = unit_vector_fields(coords, dist)

# Problem
problem = IVP([u, s, p, tau_p]; namespace=@locals)
add_equation!(problem, "dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
add_equation!(problem, "dt(s) - D*lap(s) = - u@grad(s)")
add_equation!(problem, "div(u) + tau_p = 0")
add_equation!(problem, "integ(p) = 0") # Pressure gauge

# Solver
solver = build_solver(problem, timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
# Background shear
u["g"][1] = @. 1/2 + 1/2 * (tanh((z - 0.5) / 0.1) - tanh((z + 0.5) / 0.1))
# Match tracer to shear
s["g"] = u["g"][1]
# Add small vertical velocity perturbations localized to the shear layers
u["g"][2] .+= @. 0.1 * sin(2*pi*x/Lx) * exp(-(z - 0.5)^2 / 0.01)
u["g"][2] .+= @. 0.1 * sin(2*pi*x/Lx) * exp(-(z + 0.5)^2 / 0.01)

# Analysis
snapshots = add_file_handler(solver.evaluator, "snapshots"; sim_dt=0.1, max_writes=10)
add_task!(snapshots, s; name="tracer")
add_task!(snapshots, p; name="pressure")
add_task!(snapshots, -divergence(skew(u)); name="vorticity")

# CFL
cfl = CFL(solver; initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
          max_change=1.5, min_change=0.5, max_dt=max_timestep)
add_velocity!(cfl, u)

# Flow properties
flow = GlobalFlowProperty(solver; cadence=10)
add_property!(flow, (u * ez)^2; name="w2")

# Main loop
try
    @info "Starting main loop"
    while solver.proceed
        timestep = compute_timestep(cfl)
        step!(solver, timestep)
        if (solver.iteration - 1) % 10 == 0
            max_w = sqrt(flow_max(flow, "w2"))
            @info "Iteration=$(solver.iteration), Time=$(solver.sim_time), dt=$(timestep), max(w)=$(max_w)"
        end
    end
catch e
    @error "Exception raised, triggering end of main loop."
    rethrow(e)
finally
    log_stats(solver)
end

"""
Dedalus script simulating librational instability in a disk by solving the
incompressible Navier-Stokes equations linearized around a background librating
flow. This script demonstrates solving an initial value problem in the disk.
It can be ran serially or in parallel, and uses the built-in analysis framework
to save data snapshots to HDF5 files. The `plot_disk.jl` and `plot_scalars.jl`
scripts can be used to produce plots from the saved data. The simulation should
take roughly 20 cpu-minutes to run.

The problem is non-dimensionalized using the disk radius and librational frequency,
so the resulting viscosity is related to the Ekman number as:

    nu = Ekman

For incompressible hydro in the disk, we need one tau term for the velocity.
Here we lift to the original (k=0) basis.

To run and plot using e.g. 4 processes:
    \$ mpiexec -n 4 julia libration.jl
    \$ mpiexec -n 4 julia plot_disk.jl snapshots/*.h5
    \$ julia plot_scalars.jl scalars/*.h5
"""

using Dedalus
using SpecialFunctions: besselj
using Logging

logger = Logging.current_logger()

# Parameters
Nphi, Nr = 32, 128
Ekman = 1 / 2 / 20^2
Ro = 40
dealias = 3/2
stop_sim_time = 50
timestepper = SBDF2
timestep = 1e-3
dtype = Float64

# Bases
coords = PolarCoordinates("phi", "r")
dist = Distributor(coords; dtype=dtype)
disk = DiskBasis(coords; shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
disk_edge = edge(disk)

# Fields
u = VectorField(dist, coords; name="u", bases=(disk,))
p = Field(dist; name="p", bases=(disk,))
tau_u = VectorField(dist, coords; name="tau_u", bases=(disk_edge,))
tau_p = Field(dist; name="tau_p")

# Substitutions
phi, r = local_grids(dist, disk)
nu = Ekman
lift = A -> Lift(A, disk, -1)

# Background librating flow
u0_real = VectorField(dist, coords; bases=(disk,))
u0_imag = VectorField(dist, coords; bases=(disk,))
u0_real["g"][1] = @. Ro * real(besselj(1, (1 - 1im) * r / sqrt(2 * Ekman)) / besselj(1, (1 - 1im) / sqrt(2 * Ekman)))
u0_imag["g"][1] = @. Ro * imag(besselj(1, (1 - 1im) * r / sqrt(2 * Ekman)) / besselj(1, (1 - 1im) / sqrt(2 * Ekman)))
t = Field(dist)
u0 = cos(t) * u0_real - sin(t) * u0_imag

# Problem
problem = IVP([p, u, tau_u, tau_p]; time=t, namespace=@locals)
add_equation!(problem, "div(u) + tau_p = 0")
add_equation!(problem, "dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = - u@grad(u0) - u0@grad(u)")
add_equation!(problem, "u(r=1) = 0")
add_equation!(problem, "integ(p) = 0")

# Solver
solver = build_solver(problem, timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
fill_random!(u, "g"; seed=42, distribution="standard_normal")  # Random noise
low_pass_filter!(u; scales=0.25)  # Keep only lower fourth of the modes

# Analysis
snapshots = add_file_handler(solver.evaluator, "snapshots"; sim_dt=0.1, max_writes=20)
add_task!(snapshots, -divergence(skew(u)); scales=(4, 1), name="vorticity")
scalars = add_file_handler(solver.evaluator, "scalars"; sim_dt=0.01)
add_task!(scalars, integrate(0.5 * u * u); name="KE")

# Flow properties
flow = GlobalFlowProperty(solver; cadence=100)
add_property!(flow, u * u; name="u2")

# Main loop
try
    @info "Starting main loop"
    while solver.proceed
        step!(solver, timestep)
        if (solver.iteration - 1) % 100 == 0
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

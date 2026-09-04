"""
Dedalus script simulating the viscous shallow water equations on a sphere. This
script demonstrates solving an initial value problem on the sphere. It can be
ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_sphere.jl` script can be used to produce
plots from the saved data. The simulation should take about 5 cpu-minutes to run.

The script implements the test case of a barotropically unstable mid-latitude
jet from Galewsky et al. 2004 (https://doi.org/10.3402/tellusa.v56i5.14436).
The initial height field balanced the imposed jet is solved with an LBVP.
A perturbation is then added and the solution is evolved as an IVP.

To run and plot using e.g. 4 processes:
    \$ mpiexec -n 4 julia shallow_water.jl
    \$ mpiexec -n 4 julia plot_sphere.jl snapshots/*.h5
"""

using Dedalus
using Logging

logger = Logging.current_logger()

# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600

# Parameters
Nphi = 256
Ntheta = 128
dealias = 3/2
R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e5 * meter^2 / second / 32^2  # Hyperdiffusion matched at ell=32
g = 9.80616 * meter / second^2
H = 1e4 * meter
timestep = 600 * second
stop_sim_time = 360 * hour
dtype = Float64

# Bases
coords = S2Coordinates("phi", "theta")
dist = Distributor(coords; dtype=dtype)
basis = SphereBasis(coords, (Nphi, Ntheta); radius=R, dealias=dealias, dtype=dtype)

# Fields
u = VectorField(dist, coords; name="u", bases=(basis,))
h = Field(dist; name="h", bases=(basis,))

# Substitutions
zcross = A -> MulCosine(skew(A))

# Initial conditions: zonal jet
phi, theta = local_grids(dist, basis)
lat = @. pi / 2 - theta + 0 * phi
umax = 80 * meter / second
lat0 = pi / 7
lat1 = pi / 2 - lat0
en = exp(-4 / (lat1 - lat0)^2)
jet = @. (lat0 <= lat) & (lat <= lat1)
u_jet = @. umax / en * exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))
u["g"][1][jet] = u_jet

# Initial conditions: balanced height
c = Field(dist; name="c")
problem = LBVP([h, c]; namespace=@locals)
add_equation!(problem, "g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
add_equation!(problem, "ave(h) = 0")
solver = build_solver(problem)
solve!(solver)

# Initial conditions: perturbation
lat2 = pi / 4
hpert = 120 * meter
alpha = 1 / 3
beta = 1 / 15
h["g"] .+= @. hpert * cos(lat) * exp(-(phi / alpha)^2) * exp(-((lat2 - lat) / beta)^2)

# Problem
problem = IVP([u, h]; namespace=@locals)
add_equation!(problem, "dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
add_equation!(problem, "dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")

# Solver
solver = build_solver(problem, RK222)
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = add_file_handler(solver.evaluator, "snapshots"; sim_dt=1*hour, max_writes=10)
add_task!(snapshots, h; name="height")
add_task!(snapshots, -divergence(skew(u)); name="vorticity")

# Main loop
try
    @info "Starting main loop"
    while solver.proceed
        step!(solver, timestep)
        if (solver.iteration - 1) % 10 == 0
            @info "Iteration=$(solver.iteration), Time=$(solver.sim_time), dt=$(timestep)"
        end
    end
catch e
    @error "Exception raised, triggering end of main loop."
    rethrow(e)
finally
    log_stats(solver)
end

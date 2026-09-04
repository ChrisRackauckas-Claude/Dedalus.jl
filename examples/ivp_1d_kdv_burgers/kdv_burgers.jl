"""
Dedalus script simulating the 1D Korteweg-de Vries / Burgers equation.
This script demonstrates solving a 1D initial value problem and produces
a space-time plot of the solution. It should take just a few seconds to
run (serial only).

We use a Fourier basis to solve the IVP:
    dt(u) + u*dx(u) = a*dx(dx(u)) + b*dx(dx(dx(u)))

To run and plot:
    \$ julia kdv_burgers.jl
"""

using Dedalus
using Logging
# NOTE: For plotting, install and use CairoMakie or Plots.jl
# using CairoMakie

logger = Logging.current_logger()

# Parameters
Lx = 10
Nx = 1024
a = 1e-4
b = 2e-4
dealias = 3/2
stop_sim_time = 10
timestepper = SBDF2
timestep = 2e-3
dtype = Float64

# Bases
xcoord = Coordinate("x")
dist = Distributor(xcoord; dtype=dtype)
xbasis = RealFourier(xcoord, Nx; bounds=(0, Lx), dealias=dealias)

# Fields
u = Field(dist; name="u", bases=(xbasis,))

# Substitutions
dx = A -> Differentiate(A, xcoord)

# Problem
problem = IVP([u]; namespace=@locals)
add_equation!(problem, "dt(u) - a*dx(dx(u)) - b*dx(dx(dx(u))) = - u*dx(u)")

# Initial conditions
x = local_grid(dist, xbasis)
n = 20
u["g"] = @. log(1 + cosh(n)^2 / cosh(n * (x - 0.2 * Lx))^2) / (2 * n)

# Solver
solver = build_solver(problem, timestepper)
solver.stop_sim_time = stop_sim_time

# Main loop
u_list = [copy(u["g", 1])]  # Julia 1-based: index 1 instead of Python's 1
t_list = [solver.sim_time]
while solver.proceed
    step!(solver, timestep)
    if solver.iteration % 100 == 0
        @info "Iteration=$(solver.iteration), Time=$(solver.sim_time), dt=$(timestep)"
    end
    if solver.iteration % 25 == 0
        push!(u_list, copy(u["g", 1]))
        push!(t_list, solver.sim_time)
    end
end

# Plot
# Uncomment below if CairoMakie is available:
# fig = Figure(; size=(600, 400))
# ax = Axis(fig[1, 1]; xlabel="x", ylabel="t", title="KdV-Burgers, (a,b)=($a,$b)")
# heatmap!(ax, vec(x), t_list, reduce(hcat, u_list)'; colormap=:RdBu, colorrange=(-0.8, 0.8))
# xlims!(ax, 0, Lx)
# ylims!(ax, 0, stop_sim_time)
# save("kdv_burgers.png", fig; px_per_unit=2)

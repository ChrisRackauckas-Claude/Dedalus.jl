"""
Dedalus script solving the 2D Poisson equation with mixed boundary conditions.
This script demonstrates solving a 2D Cartesian linear boundary value problem
and produces a plot of the solution. It should take just a few seconds to run.

We use a Fourier(x) * Chebyshev(y) discretization to solve the LBVP:
    dx(dx(u)) + dy(dy(u)) = f
    u(y=0) = g
    dy(u)(y=Ly) = h

For a scalar Laplacian on a finite interval, we need two tau terms. Here we
choose to lift them to the natural output (second derivative) basis.

To run and plot:
    \$ julia poisson.jl
"""

using Dedalus
using Random
# NOTE: For plotting, install and use CairoMakie or Plots.jl
# using CairoMakie

# Parameters
Lx, Ly = 2*pi, pi
Nx, Ny = 256, 128
dtype = Float64

# Bases
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; dtype=dtype)
xbasis = RealFourier(coords["x"], Nx; bounds=(0, Lx))
ybasis = ChebyshevT(coords["y"], Ny; bounds=(0, Ly))

# Fields
u = Field(dist; name="u", bases=(xbasis, ybasis))
tau_1 = Field(dist; name="tau_1", bases=(xbasis,))
tau_2 = Field(dist; name="tau_2", bases=(xbasis,))

# Forcing
x, y = local_grids(dist, xbasis, ybasis)
f = Field(dist; bases=(xbasis, ybasis))
g = Field(dist; bases=(xbasis,))
h = Field(dist; bases=(xbasis,))
fill_random!(f, "g"; seed=40)
low_pass_filter!(f; shape=(64, 32))
g["g"] = sin.(8 .* x) .* 0.025
h["g"] .= 0

# Substitutions
dy = A -> Differentiate(A, coords["y"])
lift_basis = derivative_basis(ybasis, 2)
lift = (A, n) -> Lift(A, lift_basis, n)

# Problem
problem = LBVP([u, tau_1, tau_2]; namespace=@locals)
add_equation!(problem, "lap(u) + lift(tau_1,-1) + lift(tau_2,-2) = f")
add_equation!(problem, "u(y=0) = g")
add_equation!(problem, "dy(u)(y=Ly) = h")

# Solver
solver = build_solver(problem)
solve!(solver)

# Gather global data
x = global_grid(xbasis, dist; scale=1)
y = global_grid(ybasis, dist; scale=1)
ug = allgather_data(u, "g")

# Plot
# Uncomment below if CairoMakie is available:
# if comm_rank(dist) == 0
#     fig = Figure(; size=(600, 400))
#     ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="Randomly forced Poisson equation", aspect=DataAspect())
#     heatmap!(ax, vec(x), vec(y), ug')
#     save("poisson.png", fig; px_per_unit=2)
# end

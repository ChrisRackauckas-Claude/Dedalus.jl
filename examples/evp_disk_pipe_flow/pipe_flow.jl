"""
Dedalus script solving the linear stability eigenvalue problem for pipe flow.
This script demonstrates solving an eigenvalue problem in the periodic cylinder
using the disk basis and a parametrized axial wavenumber. It should take just
a few seconds to run (serial only).

The radius of the pipe is R = 1, and the problem is non-dimensionalized using
the radius and laminar velocity, such that the background flow is w0 = 1 - r^2.

No-slip boundary conditions are implemented on the velocity perturbations.
For incompressible hydro with one boundary, we need one tau term each for the
scalar axial velocity and vector horizontal (in-disk) velocity. Here we choose
to lift the tau terms to the original (k=0) basis.

The eigenvalues are compared to the results of Vasil et al. (2016) [1] in Table 3.

To run, print, and plot the slowest decaying mode:
    \$ julia pipe_flow.jl

References:
    [1]: G. M. Vasil, K. J. Burns, D. Lecoanet, S. Olver, B. P. Brown, J. S. Oishi,
         "Tensor calculus in polar coordinates using Jacobi polynomials," Journal
         of Computational Physics (2016).
"""

using Dedalus
using Logging
# NOTE: For plotting, install CairoMakie or Plots.jl
# using CairoMakie

logger = Logging.current_logger()

# Parameters
Re = 1e4
kz = 1
m = 5
Nphi = 2 * m + 2
Nr = 64
dtype = ComplexF64

# Bases
coords = PolarCoordinates("phi", "r")
dist = Distributor(coords; dtype=dtype)
disk = DiskBasis(coords; shape=(Nphi, Nr), radius=1, dtype=dtype)
phi, r = local_grids(dist, disk)

# Fields
s = Field(dist; name="s")
u = VectorField(dist, coords; name="u", bases=(disk,))
w = Field(dist; name="w", bases=(disk,))
p = Field(dist; name="p", bases=(disk,))
tau_u = VectorField(dist, coords; name="tau_u", bases=(edge(disk),))
tau_w = Field(dist; name="tau_w", bases=(edge(disk),))
tau_p = Field(dist; name="tau_p")

# Substitutions
dt = A -> s * A
dz = A -> 1im * kz * A
lift_basis = derivative_basis(disk, 2)
lift = A -> Lift(A, lift_basis, -1)

# Background
w0 = Field(dist; name="w0", bases=(radial_basis(disk),))
w0["g"] = @. 1 - r^2

# Problem
problem = EVP([u, w, p, tau_u, tau_w, tau_p]; eigenvalue=s, namespace=@locals)
add_equation!(problem, "div(u) + dz(w) = 0")
add_equation!(problem, "dt(u) + w0*dz(u) + grad(p) - (1/Re)*(lap(u)+dz(dz(u))) + lift(tau_u) = 0")
add_equation!(problem, "dt(w) + w0*dz(w) + u@grad(w0) + dz(p) - (1/Re)*(lap(w)+dz(dz(w))) + lift(tau_w) = 0")
add_equation!(problem, "u(r=1) = 0")
add_equation!(problem, "w(r=1) = 0")

# Solver
solver = build_solver(problem)
sp = subproblems_by_group(solver, (m, nothing))
solve_dense!(solver, sp)
evals = solver.eigenvalues[isfinite.(solver.eigenvalues)]
evals = evals[sortperm(-real.(evals))]
println("Slowest decaying mode: lambda = $(evals[1])")
set_state!(solver, argmin(abs.(solver.eigenvalues .- evals[1])), subsystems(sp)[1])

# Plot eigenfunction
# Uncomment below if CairoMakie is available:
# scales = (32, 4)
# omega = divergence(skew(u))
# omega_eval = evaluate(omega)
# change_scales!(omega_eval, scales)
# change_scales!(u, scales)
# change_scales!(w, scales)
# change_scales!(p, scales)
# phi, r = local_grids(dist, disk; scales=scales)
# x_cart, y_cart = cartesian(coords, phi, r)
#
# fig = Figure(; size=(600, 600))
# cmap = :RdBu
# ax1 = Axis(fig[1, 1]; title=L"u_\phi", aspect=DataAspect())
# heatmap!(ax1, x_cart, y_cart, real.(u["g"][1]); colormap=cmap)
# ax2 = Axis(fig[1, 2]; title=L"u_r", aspect=DataAspect())
# heatmap!(ax2, x_cart, y_cart, real.(u["g"][2]); colormap=cmap)
# ax3 = Axis(fig[2, 1]; title=L"w", aspect=DataAspect())
# heatmap!(ax3, x_cart, y_cart, real.(w["g"]); colormap=cmap)
# ax4 = Axis(fig[2, 2]; title=L"p", aspect=DataAspect())
# heatmap!(ax4, x_cart, y_cart, real.(p["g"]); colormap=cmap)
# save("pipe_eigenfunctions.png", fig; px_per_unit=2)

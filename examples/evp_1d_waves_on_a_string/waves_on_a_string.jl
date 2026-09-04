"""
Dedalus script computing the eigenmodes of waves on a clamped string.
This script demonstrates solving a 1D eigenvalue problem and produces
plots of the first few eigenmodes and the relative error of the eigenvalues.
It should take just a few seconds to run (serial only).

We use a Legendre basis to solve the EVP:
    s*u + dx(dx(u)) = 0
    u(x=0) = 0
    u(x=Lx) = 0
where s is the eigenvalue.

For the second derivative on a closed interval, we need two tau terms.
Here we choose to use a first-order formulation, putting one tau term
on an auxiliary first-order variable and another in the PDE, and lifting
both to the first derivative basis.

To run and plot:
    \$ julia waves_on_a_string.jl
"""

using Dedalus
using Logging
# NOTE: For plotting, install CairoMakie or Plots.jl
# using CairoMakie

logger = Logging.current_logger()

# Parameters
Lx = 1
Nx = 128
dtype = ComplexF64

# Bases
xcoord = Coordinate("x")
dist = Distributor(xcoord; dtype=dtype)
xbasis = Legendre(xcoord, Nx; bounds=(0, Lx))

# Fields
u = Field(dist; name="u", bases=(xbasis,))
tau_1 = Field(dist; name="tau_1")
tau_2 = Field(dist; name="tau_2")
s = Field(dist; name="s")

# Substitutions
dx = A -> Differentiate(A, xcoord)
lift_basis = derivative_basis(xbasis, 1)
lift = A -> Lift(A, lift_basis, -1)
ux = dx(u) + lift(tau_1)  # First-order reduction
uxx = dx(ux) + lift(tau_2)

# Problem
problem = EVP([u, tau_1, tau_2]; eigenvalue=s, namespace=@locals)
add_equation!(problem, "s*u + uxx = 0")
add_equation!(problem, "u(x=0) = 0")
add_equation!(problem, "u(x=Lx) = 0")

# Solve
solver = build_solver(problem)
solve_dense!(solver, solver.subproblems[1])
evals = sort(solver.eigenvalues; by=x -> (real(x), imag(x)))
n = 1:length(evals)
true_evals = (n .* pi ./ Lx).^2
relative_error = abs.(evals .- true_evals) ./ true_evals

# Plot eigenvalue error
# Uncomment below if CairoMakie is available:
# fig = Figure(; size=(600, 400))
# ax = Axis(fig[1, 1]; xlabel="eigenvalue number", ylabel="relative eigenvalue error", yscale=log10)
# scatter!(ax, n, relative_error)
# save("eigenvalue_error.png", fig; px_per_unit=2)

# Plot eigenvectors
# fig2 = Figure(; size=(600, 400))
# ax2 = Axis(fig2[1, 1]; xlabel="x", ylabel="mode structure")
# x = local_grid(dist, xbasis)
# sorted_indices = sortperm(solver.eigenvalues)
# for (ni, idx) in enumerate(sorted_indices[1:5])
#     set_state!(solver, idx, solver.subsystems[1])
#     ug = real.(u["g"] ./ u["g"][2])  # Julia 1-based: index 2 instead of Python's 1
#     lines!(ax2, vec(x), vec(ug ./ maximum(abs.(ug))); label="n=$ni")
# end
# xlims!(ax2, 0, 1)
# axislegend(ax2; position=:rb)
# save("eigenvectors.png", fig2; px_per_unit=2)

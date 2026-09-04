"""
Dedalus script solving for the eigenvalues of the Mathieu equation. This script
demonstrates solving a periodic eigenvalue problem with nonconstant coefficients
and produces a plot of the Mathieu eigenvalues 'a' as a function of the
parameter 'q'. It should take just a few seconds to run (serial only).

We use a Fourier basis to solve the EVP:
    dx(dx(y)) + (a - 2*q*cos(2*x))*y = 0
where 'a' is the eigenvalue. Periodicity is enforced by using the Fourier basis.

To run and plot:
    \$ julia mathieu_evp.jl
"""

using Dedalus
using Logging
# NOTE: For plotting, install CairoMakie or Plots.jl
# using CairoMakie

logger = Logging.current_logger()

# Parameters
N = 32
q_list = range(0, 30; length=100)

# Basis
coord = Coordinate("x")
dist = Distributor(coord; dtype=ComplexF64)
basis = ComplexFourier(coord, N; bounds=(0, 2*pi))

# Fields
y = Field(dist; bases=(basis,))
a = Field(dist)

# Substitutions
x = local_grid(dist, basis)
q = Field(dist)
cos_2x = Field(dist; bases=(basis,))
cos_2x["g"] = cos.(2 .* x)
dx = A -> Differentiate(A, coord)

# Problem
problem = EVP([y]; eigenvalue=a, namespace=@locals)
add_equation!(problem, "dx(dx(y)) + (a - 2*q*cos_2x)*y = 0")

# Solver
solver = build_solver(problem)
evals = []
for qi in q_list
    q["g"] .= qi
    solve_dense!(solver, solver.subproblems[1]; rebuild_matrices=true)
    sorted_evals = sort(real.(solver.eigenvalues))
    push!(evals, sorted_evals[1:10])
end
evals = reduce(hcat, evals)'  # (n_q, 10) matrix

# Plot
# Uncomment below if CairoMakie is available:
# fig = Figure(; size=(600, 400))
# ax = Axis(fig[1, 1]; xlabel="q", ylabel="eigenvalues", title="Mathieu eigenvalues")
# for i in 1:2:10
#     lines!(ax, collect(q_list), evals[:, i]; color=:blue)
# end
# for i in 2:2:10
#     lines!(ax, collect(q_list), evals[:, i]; color=:orange)
# end
# xlims!(ax, minimum(q_list), maximum(q_list))
# ylims!(ax, -10, 30)
# save("mathieu_eigenvalues.png", fig; px_per_unit=2)

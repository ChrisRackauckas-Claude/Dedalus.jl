"""
Dedalus script solving the Lane-Emden equation. This script demonstrates
solving a spherically symmetric nonlinear boundary value problem inside the
ball. It should converge within roughly a dozen Newton iterations, and produces a
plot of the solution. It should take just a few seconds to run (serial only).

In astrophysics, the Lane-Emden equation is a dimensionless form of Poisson's
equation for the gravitational potential of a Newtonian self-gravitating,
spherically symmetric, polytropic fluid [1].

It is usually written as:
    lap(f) + f^n = 0
    f(r=0) = 1
    f(r=R) = 0
where n is the polytropic index, and the equation is solved over the interval
r=[0,R], where R is the n-dependent first zero of f(r).

Following [2], we rescale r by 1/R, giving:
    lap(f) + (R^2)*(f^n) = 0
    f(r=0) = 1
    f(r=1) = 0
This is a nonlinear eigenvalue problem over the unit ball, with the additional
boundary condition fixing the eigenvalue R.

We can eliminate R by rescaling f by R^(2/(n-1)), giving:
    lap(f) + f^n = 0
    f(r=1) = 0
and R can then be recovered from f(r=0) = R^(2/(n-1)).

For a scalar Laplacian in the ball, we need a single tau term. Here we choose
to lift it to the original (k=0) basis.

To run and plot:
    \$ julia lane_emden.jl

References:
    [1]: http://en.wikipedia.org/wiki/Lane-Emden_equation
    [2]: J. P. Boyd, "Chebyshev spectral methods and the Lane-Emden problem,"
         Numerical Mathematics Theory (2011).
"""

using Dedalus
using Logging
# NOTE: For plotting, install CairoMakie or Plots.jl
# using CairoMakie

logger = Logging.current_logger()

# Parameters
Nr = 64
n = 3.0
ncc_cutoff = 1e-3
tolerance = 1e-10
dealias = 2
dtype = Float64

# Bases
coords = SphericalCoordinates("phi", "theta", "r")
dist = Distributor(coords; dtype=dtype)
ball = BallBasis(coords; shape=(1, 1, Nr), radius=1, dtype=dtype, dealias=dealias)

# Fields
f = Field(dist; name="f", bases=(ball,))
tau = Field(dist; name="tau", bases=(surface(ball),))

# Substitutions
lift = A -> Lift(A, ball, -1)

# Problem
problem = NLBVP([f, tau]; namespace=@locals)
add_equation!(problem, "lap(f) + lift(tau) = - f**n")
add_equation!(problem, "f(r=1) = 0")

# Initial guess
phi, theta, r = local_grids(dist, ball)
R0 = 5
f["g"] = @. R0^(2/(n-1)) * (1 - r^2)^2

# Solver
solver = build_solver(problem; ncc_cutoff=ncc_cutoff)
pert_norm = Inf
steps = [copy(vec(f["g", 1]))]  # Julia 1-based indexing
while pert_norm > tolerance
    newton_iteration!(solver)
    pert_norm = sum(allreduce_data_norm(pert, "c", 2) for pert in solver.perturbations)
    @info "Perturbation norm: $(pert_norm)"
    f0 = allgather_data(f(r=0) |> evaluate, "g")[1, 1, 1]  # Julia 1-based indexing
    Ri = f0^((n-1)/2)
    @info "R iterate: $Ri"
    push!(steps, copy(vec(f["g", 1])))
end

# Compare to reference solutions from Boyd
R_ref = Dict(
    0.0 => sqrt(6),
    0.5 => 2.752698054065,
    1.0 => pi,
    1.5 => 3.65375373621912608,
    2.0 => 4.3528745959461246769735700,
    2.5 => 5.355275459010779,
    3.0 => 6.896848619376960375454528,
    3.25 => 8.018937527,
    3.5 => 9.535805344244850444,
    4.0 => 14.971546348838095097611066,
    4.5 => 31.836463244694285264,
)
@info repeat("-", 20)
@info "Iterations: $(solver.iteration)"
@info "Final R iteration: $Ri"
if haskey(R_ref, n)
    @info "Error vs reference: $(Ri - R_ref[n])"
end

# Plot solution
# Uncomment below if CairoMakie is available:
# fig = Figure(; size=(600, 400))
# ax = Axis(fig[1, 1]; xlabel="r", ylabel="f", title="Lane-Emden, n=$n")
# alphas = range(0.2, 1.0; length=length(steps))
# colors = vcat(fill(:blue, length(steps) - 1), [:orange])
# for (i, step) in enumerate(steps)
#     lines!(ax, vec(r), step; color=(colors[i], alphas[i]), label="step $i")
# end
# axislegend(ax)
# xlims!(ax, 0, 1)
# ylims!(ax, 0, nothing)
# save("lane_emden.png", fig; px_per_unit=2)

"""
Dedalus script for calculating the maximum linear growth rates in no-slip
Rayleigh-Benard convection over a range of horizontal wavenumbers. This script
demonstrates solving a 1D eigenvalue problem in a Cartesian domain. It can
be ran serially or in parallel, and produces a plot of the highest growth rate
found for each horizontal wavenumber. It should take a few seconds to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)^(-1/2)
    nu = (Rayleigh / Prandtl)^(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    \$ mpiexec -n 4 julia rayleigh_benard_evp.jl
"""

using Dedalus
using Logging
# NOTE: For plotting, install CairoMakie or Plots.jl
# using CairoMakie

logger = Logging.current_logger()


function max_growth_rate(Rayleigh, Prandtl, kx, Nz; NEV=10, target=0)
    """Compute maximum linear growth rate."""

    # Parameters
    Lz = 1
    # Build Fourier basis for x with prescribed kx as the fundamental mode
    Nx = 2
    Lx = 2 * pi / kx

    # Bases
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=ComplexF64)
    xbasis = ComplexFourier(coords["x"], Nx; bounds=(0, Lx))
    zbasis = ChebyshevT(coords["z"], Nz; bounds=(0, Lz))

    # Fields
    omega = Field(dist; name="omega")
    p = Field(dist; name="p", bases=(xbasis, zbasis))
    b = Field(dist; name="b", bases=(xbasis, zbasis))
    u = VectorField(dist, coords; name="u", bases=(xbasis, zbasis))
    tau_p = Field(dist; name="tau_p")
    tau_b1 = Field(dist; name="tau_b1", bases=(xbasis,))
    tau_b2 = Field(dist; name="tau_b2", bases=(xbasis,))
    tau_u1 = VectorField(dist, coords; name="tau_u1", bases=(xbasis,))
    tau_u2 = VectorField(dist, coords; name="tau_u2", bases=(xbasis,))

    # Substitutions
    kappa = (Rayleigh * Prandtl)^(-1/2)
    nu = (Rayleigh / Prandtl)^(-1/2)
    x, z = local_grids(dist, xbasis, zbasis)
    ex, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    lift = A -> Lift(A, lift_basis, -1)
    grad_u = gradient(u) + ez * lift(tau_u1) # First-order reduction
    grad_b = gradient(b) + ez * lift(tau_b1) # First-order reduction
    dt = A -> -1im * omega * A

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = EVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2]; eigenvalue=omega, namespace=@locals)
    add_equation!(problem, "trace(grad_u) + tau_p = 0")
    add_equation!(problem, "dt(b) - kappa*div(grad_b) + lift(tau_b2) - ez@u = 0")
    add_equation!(problem, "dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = 0")
    add_equation!(problem, "b(z=0) = 0")
    add_equation!(problem, "u(z=0) = 0")
    add_equation!(problem, "b(z=Lz) = 0")
    add_equation!(problem, "u(z=Lz) = 0")
    add_equation!(problem, "integ(p) = 0") # Pressure gauge

    # Solver
    solver = build_solver(problem; entry_cutoff=0)
    solve_sparse!(solver, solver.subproblems[2], NEV; target=target)  # Julia 1-based: [2] instead of [1]
    return maximum(imag.(solver.eigenvalues))
end


# Main
# Parameters
Nz = 64
Rayleigh = 1710
Prandtl = 1
kx_global = range(3.0, 3.25; length=50)
NEV = 10

# Compute growth rate over local wavenumbers (serial version)
t1 = time()
growth_global = [max_growth_rate(Rayleigh, Prandtl, kx, Nz; NEV=NEV) for kx in kx_global]
t2 = time()
@info "Elapsed solve time: $(t2 - t1)"

# Plot growth rates
# Uncomment below if CairoMakie is available:
# fig = Figure(; size=(600, 400))
# ax = Axis(fig[1, 1];
#     xlabel=L"k_x",
#     ylabel=L"\mathrm{Im}(\omega)",
#     title="Rayleigh-Benard growth rates (Ra=$(Rayleigh), Pr=$(Prandtl))")
# scatter!(ax, collect(kx_global), growth_global)
# save("growth_rates.pdf", fig)

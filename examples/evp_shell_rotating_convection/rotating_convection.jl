"""
Dedalus script solving the linear stability eigenvalue problem for rotating
Rayleigh-Benard convection in a shell. This script demonstrates solving an
eigenvalue problem with non-constant coefficients that depend on both radius
and colatitude. It should take about a minute to run (serial only).

The aspect ratio of the shell is R_inner / R_outer = 0.35, and the problem is
non-dimensionalized using the outer radius and the viscous time. The script
calculates the eigenmodes for an Ekman number of 1e-5, where the critical
mode has an azimuthal wavenumber of m=13. At the critical Rayleigh number,
the imaginary part of the eigenvalue is zero.

Both stress-free (default) and no-slip boundary conditions are implemented.
For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and temperature. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

The eigenvalues are not fully converged at the given resolution and shift slightly
if the resolution is increased. For the given resolutions, the eigenvalues agree
with Table 1 of [1] to several digits of precision.

To run and print the calculated eigenvalues:
    \$ julia rotating_convection.jl

References:
    [1]: P. Marti, M. A. Calkins, K. Julien, "A computationally
         efficient spectral method for modeling coredynamics,"
         Geochemistry, Geophysics, Geosystems (2016).
"""

using Dedalus
using Logging

logger = Logging.current_logger()

# Parameters
Nphi = 28  # Critical mode has m=13
Ntheta = 64
Nr = 64
Ri = 0.35
Ro = 1
Prandtl = 1
Ekman = 1e-5
stress_free = true
dtype = ComplexF64

# Critical Rayleigh numbers
if stress_free
    Rayleigh = 2.1029e7
else
    Rayleigh = 2.0732e7
end

# Bases
coords = SphericalCoordinates("phi", "theta", "r")
dist = Distributor(coords; dtype=dtype)
shell = ShellBasis(coords; shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dtype=dtype)
sphere = outer_surface(shell)
phi, theta, r = local_grids(dist, shell)

# Fields
om = Field(dist; name="om")
u = VectorField(dist, coords; name="u", bases=(shell,))
p = Field(dist; name="p", bases=(shell,))
T = Field(dist; name="T", bases=(shell,))
tau_u1 = VectorField(dist, coords; bases=(sphere,))
tau_u2 = VectorField(dist, coords; bases=(sphere,))
tau_T1 = Field(dist; bases=(sphere,))
tau_T2 = Field(dist; bases=(sphere,))
tau_p = Field(dist)

# Substitutions
dt = A -> -1im * om * A
rvec = VectorField(dist, coords; bases=(meridional_basis(shell),))
rvec["g"][3] = r
ez = VectorField(dist, coords; bases=(meridional_basis(shell),))
ez["g"][2] = @. -sin(theta)
ez["g"][3] = @. cos(theta)
lift_basis = derivative_basis(shell, 1)
lift = A -> Lift(A, lift_basis, -1)
grad_u = gradient(u) + rvec * lift(tau_u1)  # First-order reduction
grad_T = gradient(T) + rvec * lift(tau_T1)  # First-order reduction
strain_rate = gradient(u) + transpose_components(gradient(u))

# Problem
problem = EVP([p, u, T, tau_u1, tau_u2, tau_T1, tau_T2, tau_p]; eigenvalue=om, namespace=@locals)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "dt(u) + (1/Ekman)*cross(ez, u) + grad(p) - Rayleigh*T*rvec - div(grad_u) + lift(tau_u2) = 0")
add_equation!(problem, "Prandtl*dt(T) - dot(rvec,u) - div(grad_T) + lift(tau_T2) = 0")
add_equation!(problem, "integ(p) = 0")
if stress_free
    add_equation!(problem, "radial(u(r=Ri)) = 0")
    add_equation!(problem, "radial(u(r=Ro)) = 0")
    add_equation!(problem, "angular(radial(strain_rate(r=Ri), 0), 0) = 0")
    add_equation!(problem, "angular(radial(strain_rate(r=Ro), 0), 0) = 0")
else
    add_equation!(problem, "u(r=Ri) = 0")
    add_equation!(problem, "u(r=Ro) = 0")
end
add_equation!(problem, "T(r=Ri) = 0")
add_equation!(problem, "T(r=Ro) = 0")
add_equation!(problem, "integ(p) = 0")

# Solver
solver = build_solver(problem; ncc_cutoff=1e-10)

# Select m=13
subproblem = subproblems_by_group(solver, (13, nothing, nothing))

# Find 10 eigenvalues closest to the target
if stress_free
    target = 963.765
else
    target = 731.753
end
solve_sparse!(solver, subproblem, 10; target=target)

# Report results
@info "Predicted eigenvalue: $(target + 0im)"
@info "Calculated eigenvalue: $(solver.eigenvalues[1])"
@info "Ten eigenvalues closest to target:"
@info solver.eigenvalues

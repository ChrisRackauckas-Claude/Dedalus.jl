# Examples

This section contains worked examples demonstrating how to use Dedalus.jl to solve a variety of differential equations. Each example is a self-contained Julia script converted to a documentation page via [Literate.jl](https://github.com/fredrikekre/Literate.jl). The examples are organized by problem type.

## Eigenvalue Problems (EVP)

Eigenvalue problems find the eigenvalues and eigenmodes of linear operators.

- **[Mathieu Equation](evp_1d_mathieu.md)** -- Eigenvalues of the Mathieu equation with nonconstant coefficients on a periodic domain.
- **[Rayleigh-Benard Stability](evp_1d_rayleigh_benard.md)** -- Maximum linear growth rates in no-slip Rayleigh-Benard convection over a range of horizontal wavenumbers.
- **[Waves on a String](evp_1d_waves_on_a_string.md)** -- Eigenmodes of waves on a clamped string.
- **[Pipe Flow](evp_disk_pipe_flow.md)** -- Linear stability of pipe flow using the disk basis with a parametrized axial wavenumber.
- **[Rotating Convection](evp_shell_rotating_convection.md)** -- Linear stability of rotating Rayleigh-Benard convection in a spherical shell with non-constant coefficients.

## Linear Boundary Value Problems (LBVP)

Linear boundary value problems solve linear equations with specified boundary conditions.

- **[2D Poisson Equation](lbvp_2d_poisson.md)** -- 2D Poisson equation with mixed boundary conditions in a Cartesian domain.

## Initial Value Problems (IVP)

Initial value problems evolve equations forward in time from given initial conditions.

- **[KdV-Burgers](ivp_1d_kdv_burgers.md)** -- 1D Korteweg-de Vries / Burgers equation producing a space-time plot of the solution.
- **[Rayleigh-Benard Convection](ivp_2d_rayleigh_benard.md)** -- 2D horizontally-periodic Rayleigh-Benard convection.
- **[Shear Flow](ivp_2d_shear_flow.md)** -- 2D periodic incompressible shear flow with a passive tracer.
- **[Ensemble RBC](ivp_2d_ensemble_rbc.md)** -- Ensemble of 2D Rayleigh-Benard convection simulations with different random initial conditions.
- **[Disk Libration](ivp_disk_libration.md)** -- Librational instability in a disk via linearized incompressible Navier-Stokes equations.
- **[Centrifugal Convection](ivp_annulus_centrifugal_convection.md)** -- 2D centrifugal convection in an annulus.
- **[Shallow Water](ivp_sphere_shallow_water.md)** -- Viscous shallow water equations on a sphere.
- **[Shell Convection](ivp_shell_convection.md)** -- Boussinesq convection in a spherical shell.
- **[Internally Heated Convection](ivp_ball_internally_heated_convection.md)** -- Internally-heated Boussinesq convection in the ball.

## Nonlinear Boundary Value Problems (NLBVP)

Nonlinear boundary value problems solve nonlinear equations iteratively via Newton's method.

- **[Lane-Emden](nlbvp_ball_lane_emden.md)** -- Lane-Emden equation in the ball, a spherically symmetric nonlinear boundary value problem solved via Newton iteration.

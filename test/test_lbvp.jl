"""Tests for linear boundary value problems (LBVPs) -- Cartesian only."""

using Test
using Dedalus

@testset "LBVP" begin

    dtype_range = [Float64, ComplexF64]

    @testset "algebraic T=$T" for T in dtype_range
        coord = Coordinate("x")
        dist = Distributor(coord, dtype=T)
        u = Field(dist, name="u")
        v = Field(dist, name="v")
        F = Field(dist, name="F")
        v["g"] = -1
        F["g"] = -3
        problem = LBVP([u], namespace=Dict("u" => u, "v" => v, "F" => F))
        add_equation!(problem, "v*u = F")
        solver = build_solver(problem)
        solve!(solver)
        u_true = 3
        @test isapprox(u["g"], u_true, atol=1e-12)
    end

    @testset "Poisson Fourier N=$N T=$T matrix_coupling=$mc" for
            N in [32],
            T in dtype_range,
            mc in [false, true]
        coord = Coordinate("x")
        dist = Distributor(coord, dtype=T)
        basis = Fourier(coord, size=N, bounds=(0, 2 * pi), dtype=T)
        x = local_grid(dist, basis)
        # Fields
        u = Field(dist, name="u", bases=basis)
        g = Field(dist, name="c")
        u_true = sin.(x)
        f = Field(dist, bases=basis)
        f["g"] = -sin.(x)
        # Problem
        dx = A -> Differentiate(A, coord)
        integ = A -> Integrate(A, coord)
        problem = LBVP([u, g], namespace=Dict("u" => u, "g" => g, "f" => f,
                                               "dx" => dx, "integ" => integ))
        add_equation!(problem, "dx(dx(u)) + g = f")
        add_equation!(problem, "integ(u) = 0")
        solver = build_solver(problem, matrix_coupling=[mc])
        solve!(solver)
        @test isapprox(u["g"], u_true, atol=1e-10)
    end

    @testset "Poisson Jacobi N=$N (a,b)=$ab T=$T" for
            N in [32],
            ab in [(-0.5, -0.5), (0.0, 0.0)],
            T in dtype_range
        a, b = ab
        coord = Coordinate("x")
        dist = Distributor(coord, dtype=T)
        basis = Jacobi(coord, size=N, bounds=(0, 2 * pi), a=a, b=b)
        x = local_grid(dist, basis)
        # Fields
        u = Field(dist, name="u", bases=basis)
        tau1 = Field(dist, name="tau1")
        tau2 = Field(dist, name="tau2")
        u_true = sin.(x)
        f = Field(dist, bases=basis)
        f["g"] = -sin.(x)
        # Problem
        dx = A -> Differentiate(A, coord)
        lift_fn = (A, n) -> Lift(A, derivative_basis(basis, 2), n)
        problem = LBVP([u, tau1, tau2], namespace=Dict(
            "u" => u, "tau1" => tau1, "tau2" => tau2, "f" => f,
            "dx" => dx, "lift" => lift_fn))
        add_equation!(problem, "dx(dx(u)) + lift(tau1,-1) + lift(tau2,-2) = f")
        add_equation!(problem, "u(x=\"left\") = 0")
        add_equation!(problem, "u(x=\"right\") = 0")
        solver = build_solver(problem)
        solve!(solver)
        @test isapprox(u["g"], u_true, atol=1e-10)
    end

    # ---- 2D Cartesian LBVP tests ----

    @testset "Poisson 2D Fourier-Chebyshev N=$N T=$T" for
            N in [16],
            T in dtype_range
        c = CartesianCoordinates("x", "y")
        d = Distributor(c, dtype=T)
        if T == ComplexF64
            xb = ComplexFourier(c.coords[1], size=N, bounds=(0, 2 * pi))
        else
            xb = RealFourier(c.coords[1], size=N, bounds=(0, 2 * pi))
        end
        yb = Chebyshev(c.coords[2], size=N, bounds=(0, 1))
        x, y = local_grids(d, xb, yb)
        # Fields
        u = Field(d, name="u", bases=(xb, yb))
        tau1 = Field(d, name="tau1", bases=(xb,))
        tau2 = Field(d, name="tau2", bases=(xb,))
        # Source: simple polynomial in y, sinusoidal in x
        u_true = @. sin(x) * y * (1 - y)
        f = Field(d, bases=(xb, yb))
        f["g"] = @. -sin(x) * y * (1 - y) + sin(x) * (-2)
        # Problem
        lift_basis = derivative_basis(yb, 2)
        lift_fn = (A, n) -> Lift(A, lift_basis, n)
        problem = LBVP([u, tau1, tau2], namespace=Dict(
            "u" => u, "tau1" => tau1, "tau2" => tau2, "f" => f,
            "lap" => lap, "lift" => lift_fn))
        add_equation!(problem, "lap(u) + lift(tau1,-1) + lift(tau2,-2) = f")
        add_equation!(problem, "u(y=0) = 0")
        add_equation!(problem, "u(y=1) = 0")
        solver = build_solver(problem)
        solve!(solver)
        @test isapprox(u["g"], u_true, atol=1e-10)
    end

    @testset "Poisson 2D Chebyshev-Chebyshev N=$N T=$T" for
            N in [16],
            T in dtype_range
        c = CartesianCoordinates("x", "y")
        d = Distributor(c, dtype=T)
        xb = Chebyshev(c.coords[1], size=N, bounds=(0, 1))
        yb = Chebyshev(c.coords[2], size=N, bounds=(0, 1))
        x, y = local_grids(d, xb, yb)
        # Fields
        u = Field(d, name="u", bases=(xb, yb))
        tau_x1 = Field(d, name="tau_x1", bases=(yb,))
        tau_x2 = Field(d, name="tau_x2", bases=(yb,))
        tau_y1 = Field(d, name="tau_y1", bases=(xb,))
        tau_y2 = Field(d, name="tau_y2", bases=(xb,))
        # Source: u = x^2 * y^2
        u_true = @. x^2 * y^2
        f = Field(d, bases=(xb, yb))
        f["g"] = @. 2 * y^2 + 2 * x^2
        # Problem
        lift_x_basis = derivative_basis(xb, 2)
        lift_y_basis = derivative_basis(yb, 2)
        problem = LBVP([u, tau_x1, tau_x2, tau_y1, tau_y2], namespace=Dict(
            "u" => u, "f" => f, "lap" => lap,
            "tau_x1" => tau_x1, "tau_x2" => tau_x2,
            "tau_y1" => tau_y1, "tau_y2" => tau_y2,
            "lift_x" => (A, n) -> Lift(A, lift_x_basis, n),
            "lift_y" => (A, n) -> Lift(A, lift_y_basis, n)))
        add_equation!(problem, "lap(u) + lift_x(tau_x1,-1) + lift_x(tau_x2,-2) + lift_y(tau_y1,-1) + lift_y(tau_y2,-2) = f")
        add_equation!(problem, "u(x=0) = 0")
        add_equation!(problem, "u(x=1) = u_true(x=1)")
        add_equation!(problem, "u(y=0) = 0")
        add_equation!(problem, "u(y=1) = u_true(y=1)")
        solver = build_solver(problem)
        solve!(solver)
        @test isapprox(u["g"], u_true, atol=1e-10)
    end

    # ---- 3D Cartesian LBVP tests ----

    @testset "Poisson 3D FFC N=$N T=$T" for
            N in [8],
            T in dtype_range
        c = CartesianCoordinates("x", "y", "z")
        d = Distributor(c, dtype=T)
        if T == ComplexF64
            xb = ComplexFourier(c.coords[1], size=N, bounds=(0, 2 * pi))
            yb = ComplexFourier(c.coords[2], size=N, bounds=(0, 2 * pi))
        else
            xb = RealFourier(c.coords[1], size=N, bounds=(0, 2 * pi))
            yb = RealFourier(c.coords[2], size=N, bounds=(0, 2 * pi))
        end
        zb = Chebyshev(c.coords[3], size=N, bounds=(0, 1))
        x, y, z = local_grids(d, xb, yb, zb)
        # Fields
        u = Field(d, name="u", bases=(xb, yb, zb))
        tau1 = Field(d, name="tau1", bases=(xb, yb))
        tau2 = Field(d, name="tau2", bases=(xb, yb))
        u_true = @. sin(x) * sin(y) * z * (1 - z)
        f = Field(d, bases=(xb, yb, zb))
        f["g"] = @. -2 * sin(x) * sin(y) * z * (1 - z) + sin(x) * sin(y) * (-2)
        # Problem
        lift_basis = derivative_basis(zb, 2)
        lift_fn = (A, n) -> Lift(A, lift_basis, n)
        problem = LBVP([u, tau1, tau2], namespace=Dict(
            "u" => u, "tau1" => tau1, "tau2" => tau2, "f" => f,
            "lap" => lap, "lift" => lift_fn))
        add_equation!(problem, "lap(u) + lift(tau1,-1) + lift(tau2,-2) = f")
        add_equation!(problem, "u(z=0) = 0")
        add_equation!(problem, "u(z=1) = 0")
        solver = build_solver(problem)
        solve!(solver)
        @test isapprox(u["g"], u_true, atol=1e-10)
    end

end

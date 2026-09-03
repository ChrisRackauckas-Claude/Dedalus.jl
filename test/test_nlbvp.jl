"""Tests for nonlinear boundary value problems (NLBVPs) -- Cartesian only."""

using Test
using Dedalus

@testset "NLBVP" begin

    @testset "sin Jacobi N=$N (a,b)=$ab dealias=$dealias T=$T" for
            N in [12],
            ab in [(-0.5, -0.5), (0.0, 0.0)],
            dealias in [1, 1.5],
            T in [Float64, ComplexF64]
        a, b = ab
        ncc_cutoff = 1e-6
        tolerance = 1e-6
        # Bases
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, bounds=(0, 1), a=a, b=b, dealias=dealias)
        x = local_grid(d, basis, scale=1)
        # Fields
        u = Field(d, bases=basis)
        tau = Field(d)
        # Problem
        dx = A -> Differentiate(A, c)
        lift_fn = A -> Lift(A, derivative_basis(basis, 1), -1)
        problem = NLBVP([u, tau], namespace=Dict(
            "u" => u, "tau" => tau,
            "dx" => dx, "lift" => lift_fn))
        add_equation!(problem, "dx(u)**2 + u**2 + lift(tau) = 1")
        add_equation!(problem, "u(x=0) = 1")
        solver = build_solver(problem, ncc_cutoff=ncc_cutoff)
        u["g"] = @. 1 - x / 2
        error_val = Inf
        while error_val > tolerance
            newton_iteration!(solver)
            error_val = sum(allreduce_data_norm(pert, "c", 2) for pert in solver.perturbations)
            if solver.iteration > 20
                @test false
                break
            end
        end
        # Check solution
        u_true = cos.(x)
        change_scales!(u, 1)
        @test isapprox(u["g"], u_true, atol=1e-5)
    end

    @testset "Poisson NLBVP N=$N T=$T dealias=$dealias" for
            N in [16],
            T in [Float64, ComplexF64],
            dealias in [1, 1.5]
        # Solve Poisson equation as an NLBVP (linear problem in nonlinear framework)
        ncc_cutoff = 1e-10
        tolerance = 1e-10
        # Bases
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, bounds=(0, 1), a=0, b=0, dealias=dealias)
        x = local_grid(d, basis, scale=1)
        # Fields
        u = Field(d, bases=basis)
        tau1 = Field(d)
        tau2 = Field(d)
        # Problem: u'' = 6x, u(0)=0, u(1)=1  =>  u = x^3
        dx = A -> Differentiate(A, c)
        lift_fn = (A, n) -> Lift(A, derivative_basis(basis, 2), n)
        f = Field(d, bases=basis)
        f["g"] = @. 6 * x
        problem = NLBVP([u, tau1, tau2], namespace=Dict(
            "u" => u, "tau1" => tau1, "tau2" => tau2,
            "f" => f, "dx" => dx, "lift" => lift_fn))
        add_equation!(problem, "dx(dx(u)) + lift(tau1,-1) + lift(tau2,-2) = f")
        add_equation!(problem, "u(x=0) = 0")
        add_equation!(problem, "u(x=1) = 1")
        solver = build_solver(problem, ncc_cutoff=ncc_cutoff)
        # Initial guess
        u["g"] = @. x
        error_val = Inf
        while error_val > tolerance
            newton_iteration!(solver)
            error_val = sum(sum(abs.(pert["c"])) for pert in solver.perturbations)
            if solver.iteration > 20
                @test false
                break
            end
        end
        u_true = @. x^3
        change_scales!(u, 1)
        @test isapprox(u["g"], u_true, atol=1e-8)
    end

    @testset "nonlinear ODE N=$N T=$T dealias=$dealias" for
            N in [16],
            T in [Float64, ComplexF64],
            dealias in [1, 1.5]
        # Solve u'' + u^2 = f with known solution u = sin(x) on [0, pi/2]
        # f = -sin(x) + sin(x)^2
        ncc_cutoff = 1e-10
        tolerance = 1e-10
        # Bases
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, bounds=(0, pi / 2), a=0, b=0, dealias=dealias)
        x = local_grid(d, basis, scale=1)
        # Fields
        u = Field(d, bases=basis)
        tau1 = Field(d)
        tau2 = Field(d)
        f = Field(d, bases=basis)
        f["g"] = @. -sin(x) + sin(x)^2
        # Problem
        dx = A -> Differentiate(A, c)
        lift_fn = (A, n) -> Lift(A, derivative_basis(basis, 2), n)
        problem = NLBVP([u, tau1, tau2], namespace=Dict(
            "u" => u, "tau1" => tau1, "tau2" => tau2,
            "f" => f, "dx" => dx, "lift" => lift_fn))
        add_equation!(problem, "dx(dx(u)) + u**2 + lift(tau1,-1) + lift(tau2,-2) = f")
        add_equation!(problem, "u(x=0) = 0")
        add_equation!(problem, "u(x=$(pi/2)) = 1")
        solver = build_solver(problem, ncc_cutoff=ncc_cutoff)
        # Initial guess
        u["g"] = @. 2 * x / pi
        error_val = Inf
        while error_val > tolerance
            newton_iteration!(solver)
            error_val = sum(sum(abs.(pert["c"])) for pert in solver.perturbations)
            if solver.iteration > 30
                @test false
                break
            end
        end
        u_true = sin.(x)
        change_scales!(u, 1)
        @test isapprox(u["g"], u_true, atol=1e-6)
    end

end

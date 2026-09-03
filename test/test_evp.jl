"""Tests for eigenvalue problems (EVPs) -- Cartesian only."""

using Test
using Dedalus

@testset "EVP" begin

    @testset "Laplace Fourier basis=$btype N=$N T=$T" for
            btype in ["RealFourier", "ComplexFourier"],
            N in [32],
            T in [ComplexF64]
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        if btype == "RealFourier"
            b = RealFourier(c, size=N, bounds=(0, 2 * pi))
        else
            b = ComplexFourier(c, size=N, bounds=(0, 2 * pi))
        end
        # Fields
        u = Field(d, bases=b)
        s = Field(d)
        # Problem
        dx = A -> Differentiate(A, c)
        problem = EVP([u], s, namespace=Dict("u" => u, "s" => s, "dx" => dx))
        add_equation!(problem, "s*u + dx(dx(u)) = 0")
        # Solver
        solver = build_solver(problem, matrix_coupling=[true])
        solve_dense!(solver, solver.subproblems[1])
        # Check solution
        k = wavenumbers(b)
        if btype == "RealFourier"
            k = k[2:end]  # Drop one k=0 for msin
        end
        @test isapprox(sort(real.(solver.eigenvalues)), sort(k .^ 2), atol=1e-10)
    end

    @testset "Laplace Jacobi N=$N (a,b)=$ab T=$T sparse=$sparse" for
            N in [32],
            ab in [(-0.5, -0.5), (0.0, 0.0)],
            T in [ComplexF64],
            sparse in [true, false]
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a=a, b=b, bounds=(0, pi))
        # Fields
        u = Field(d, bases=basis)
        s = Field(d)
        t1 = Field(d)
        t2 = Field(d)
        # Problem
        dx = A -> Differentiate(A, c)
        lift_fn = (A, n) -> Lift(A, derivative_basis(basis, 2), n)
        problem = EVP([u, t1, t2], s, namespace=Dict(
            "u" => u, "s" => s, "t1" => t1, "t2" => t2,
            "dx" => dx, "lift" => lift_fn, "pi" => pi))
        add_equation!(problem, "s*u + dx(dx(u)) + lift(t1,-1) + lift(t2,-2) = 0")
        add_equation!(problem, "u(x=0) = 0")
        add_equation!(problem, "u(x=pi) = 0")
        # Solver
        solver = build_solver(problem)
        Nmodes = 4
        if sparse
            solve_sparse!(solver, solver.subproblems[1], N=Nmodes, target=1.1)
        else
            solve_dense!(solver, solver.subproblems[1])
        end
        # Check eigenvalues
        k = 1 .+ collect(0:Nmodes-1)
        @test isapprox(sort(real.(solver.eigenvalues))[1:Nmodes], k .^ 2, atol=1e-10)
    end

    @testset "Laplace Jacobi first order N=$N (a,b)=$ab T=$T" for
            N in [32],
            ab in [(-0.5, -0.5), (0.0, 0.0)],
            T in [ComplexF64]
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a=a, b=b, bounds=(0, pi))
        x = local_grid(d, basis, scale=1)
        # Fields
        u = Field(d, bases=basis)
        s = Field(d)
        t1 = Field(d)
        t2 = Field(d)
        # Problem
        dx = A -> Differentiate(A, c)
        lift_fn = (A, n) -> Lift(A, derivative_basis(basis, 1), n)
        ux = dx(u) + lift_fn(t1, -1)
        problem = EVP([u, t1, t2], s, namespace=Dict(
            "u" => u, "s" => s, "t1" => t1, "t2" => t2,
            "dx" => dx, "lift" => lift_fn, "ux" => ux, "pi" => pi))
        add_equation!(problem, "s*u + dx(ux) + lift(t2,-1) = 0")
        add_equation!(problem, "u(x=0) = 0")
        add_equation!(problem, "u(x=pi) = 0")
        # Solver
        solver = build_solver(problem)
        solve_dense!(solver, solver.subproblems[1])
        i_sort = sortperm(real.(solver.eigenvalues))
        solver.eigenvalues = solver.eigenvalues[i_sort]
        solver.eigenvectors = solver.eigenvectors[:, i_sort]
        # Check first eigenfunction
        set_state!(solver, 1, solver.subproblems[1].subsystems[1])
        eigenfunction = u["g"] ./ u["g"][1]
        sol = sin.(x) ./ sin(x[1])
        @test isapprox(eigenfunction, sol, atol=1e-10)
    end

end

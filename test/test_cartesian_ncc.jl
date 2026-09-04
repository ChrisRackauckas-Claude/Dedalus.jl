"""Tests for Fourier and Jacobi non-constant coefficients (NCCs)."""

using Test
using Dedalus

@testset "Cartesian NCC" begin

    @testset "eval Jacobi NCC N=$N (a0,b0)=$ab k_ncc=$k_ncc k_arg=$k_arg dealias=$dealias T=$T" for
            N in [16],
            ab in [(-0.5, -0.5), (0.0, 0.0)],
            k_ncc in [0, 1],
            k_arg in [0, 1],
            dealias in [1.5],
            T in [Float64, ComplexF64]
        a0, b0 = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        b = Jacobi(c, size=N, a=a0, b=b0, bounds=(0, 1), dealias=dealias)
        b_ncc = clone_with(b, a=a0 + k_ncc, b=b0 + k_ncc)
        b_arg = clone_with(b, a=a0 + k_arg, b=b0 + k_arg)
        f = Field(d, bases=b_ncc)
        g = Field(d, bases=b_arg)
        fill_random!(f, "g")
        fill_random!(g, "g")
        vars = [g]
        w0 = f * g
        w1 = reinitialize(w0, ncc=true, ncc_vars=vars)
        problem = LBVP(vars)
        add_equation!(problem, (w1, 0))
        solver = build_solver(problem)
        store_ncc_matrices!(w1, vars, solver.subproblems)
        w0 = evaluate(w0)
        w1 = evaluate_as_ncc(w1)
        w2 = evaluate(w1 - w0)
        # Remove last 2*k_arg coeffs which have dealias-before-conversion errors
        if k_arg > 0
            @test isapprox(w2["c"][1:end-2*k_arg], zeros(length(w2["c"][1:end-2*k_arg])), atol=1e-10)
        else
            @test isapprox(w2["c"], zeros(length(w2["c"])), atol=1e-10)
        end
    end

    @testset "eval Fourier NCC N=$N dealias=$dealias T=$T" for
            N in [24],
            dealias in [1.5],
            T in [Float64, ComplexF64]
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        b = Fourier(c, size=N, bounds=(0, 1), dealias=dealias, dtype=T)
        f = Field(d, bases=b)
        g = Field(d, bases=b)
        fill_random!(f, "g")
        fill_random!(g, "g")
        vars = [g]
        w0 = f * g
        w1 = reinitialize(w0, ncc=true, ncc_vars=vars)
        problem = LBVP(vars)
        add_equation!(problem, (w1, 0))
        solver = build_solver(problem)
        store_ncc_matrices!(w1, vars, solver.subproblems)
        w0 = evaluate(w0)
        w1 = evaluate_as_ncc(w1)
        change_scales!(w0, 1)
        change_scales!(w1, 1)
        @test isapprox(w0["g"], w1["g"], atol=1e-10)
    end

    @testset "eval Fourier-Jacobi NCC N=$N (a0,b0)=$ab f_rank=$f_rank g_rank=$g_rank dealias=$dealias T=$T" for
            N in [16],
            ab in [(-0.5, -0.5)],
            f_rank in [0, 1],
            g_rank in [0, 1],
            dealias in [1.5],
            T in [Float64, ComplexF64]
        a0, b0 = ab
        c = CartesianCoordinates("x", "y")
        d = Distributor(c, dtype=T)
        xb = Fourier(c["x"], size=N, bounds=(0, 1), dealias=dealias, dtype=T)
        yb = Jacobi(c["y"], size=N, bounds=(0, 1), a=a0, b=b0, dealias=dealias)
        s = Field(d, bases=(xb, yb))
        f = TensorField(d, ntuple(_ -> c, f_rank), bases=(xb, yb))
        g = TensorField(d, ntuple(_ -> c, g_rank), bases=(xb, yb))
        fill_random!(s, "g")
        fill_random!(f, "g")
        fill_random!(g, "g")
        vars = [g]
        w0 = f * g
        w1 = reinitialize(w0, ncc=true, ncc_vars=vars)
        problem = LBVP(vars)
        add_equation!(problem, (s * g, 0))
        solver = build_solver(problem)
        store_ncc_matrices!(w1, vars, solver.subproblems)
        w0 = evaluate(w0)
        w1 = evaluate_as_ncc(w1)
        change_scales!(w0, 1)
        change_scales!(w1, 1)
        @test isapprox(w0["g"], w1["g"], atol=1e-10)
    end

    @testset "solve Jacobi NCC N=$N (a0,b0)=$ab k_ncc=$k_ncc k_arg=$k_arg dealias=$dealias T=$T" for
            N in [16],
            ab in [(-0.5, -0.5), (0.0, 0.0)],
            k_ncc in [0, 1],
            k_arg in [0, 1],
            dealias in [1.5],
            T in [Float64, ComplexF64]
        a0, b0 = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        b = Jacobi(c, size=N, a=a0, b=b0, bounds=(0, 1), dealias=dealias)
        b_ncc = clone_with(b, a=a0 + k_ncc, b=b0 + k_ncc)
        b_arg = clone_with(b, a=a0 + k_arg, b=b0 + k_arg)
        f = Field(d, bases=b_ncc)
        g = Field(d, bases=b_arg)
        u = Field(d, bases=b_arg)
        fill_random!(f, "g")
        fill_random!(g, "g")
        problem = LBVP([u])
        add_equation!(problem, (f * u, f * g))
        solver = build_solver(problem)
        solve!(solver)
        @test isapprox(u["c"], g["c"], atol=1e-10)
    end

    @testset "solve Fourier NCC N=$N dealias=$dealias T=$T" for
            N in [16],
            dealias in [1.5],
            T in [Float64, ComplexF64]
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        b = Fourier(c, size=N, bounds=(0, 1), dealias=dealias, dtype=T)
        f = Field(d, bases=b)
        g = Field(d, bases=b)
        u = Field(d, bases=b)
        fill_random!(f, "g")
        fill_random!(g, "g")
        if T == ComplexF64
            low_pass_filter!(f, scales=0.5)
            low_pass_filter!(g, scales=0.5)
        end
        problem = LBVP([u])
        add_equation!(problem, (f * u, f * g))
        solver = build_solver(problem)
        solve!(solver)
        change_scales!(u, 1)
        change_scales!(g, 1)
        @test isapprox(u["g"], g["g"], atol=1e-10)
    end

end

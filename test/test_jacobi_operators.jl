"""Tests for Jacobi convert, differentiate, interpolate, integrate, average, lift."""

using Test
using Dedalus

@testset "Jacobi Operators" begin

    N_range = [8, 9]
    ab_range = [(-0.5, -0.5), (0.0, 0.0)]
    k_range = [0, 1]
    dealias_range = [1]
    dtype_range = [Float64, ComplexF64]

    @testset "convert constant N=$N (a,b)=$ab k=$k dealias=$dealias T=$T layout=$layout" for
            N in N_range,
            ab in ab_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["g", "c"]
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a0=a, b0=b, a=a + k, b=b + k, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, basis, scale=1)
        f = Field(d)
        f["g"] = 1
        change_layout!(f, layout)
        g = evaluate(Convert(f, basis))
        @test isapprox(g["g"], f["g"], atol=1e-12)
    end

    @testset "convert N=$N (a,b)=$ab k=$k dk=$dk dealias=$dealias T=$T layout=$layout" for
            N in N_range,
            ab in ab_range,
            k in k_range,
            dk in [0, 1, 2],
            dealias in dealias_range,
            T in dtype_range,
            layout in ["g", "c"]
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a0=a, b0=b, a=a + k, b=b + k, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, basis, scale=1)
        f = Field(d, bases=basis)
        fill_random!(f, layout="g")
        low_pass_filter!(f, scales=0.5)
        change_layout!(f, layout)
        g = evaluate(Convert(f, derivative_basis(basis, dk)))
        @test isapprox(g["g"], f["g"], atol=1e-10)
    end

    @testset "convert implicit N=$N (a,b)=$ab k=$k dk=$dk dealias=$dealias T=$T" for
            N in N_range,
            ab in ab_range,
            k in k_range,
            dk in [0, 1, 2],
            dealias in dealias_range,
            T in dtype_range
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a0=a, b0=b, a=a + k, b=b + k, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, basis, scale=1)
        f = Field(d, bases=derivative_basis(basis, dk))
        fill_random!(f, layout="g")
        low_pass_filter!(f, scales=0.5)
        g = Field(d, bases=basis)
        problem = LBVP([g], namespace=Dict("g" => g, "f" => f))
        add_equation!(problem, "g = f")
        solver = build_solver(problem)
        solve!(solver)
        @test isapprox(g["g"], f["g"], atol=1e-10)
    end

    @testset "differentiate N=$N (a,b)=$ab k=$k dealias=$dealias T=$T" for
            N in N_range,
            ab in ab_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a0=a, b0=b, a=a + k, b=b + k, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, basis, scale=1)
        f = Field(d, bases=basis)
        f["g"] = @. x^5
        g = evaluate(Differentiate(f, c))
        @test isapprox(g["g"], @.(5 * x^4), atol=1e-10)
    end

    @testset "interpolate N=$N (a,b)=$ab k=$k dealias=$dealias T=$T" for
            N in N_range,
            ab in ab_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a0=a, b0=b, a=a + k, b=b + k, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, basis, scale=1)
        f = Field(d, bases=basis)
        f["g"] = @. x^5
        for p in [0.0, 1.0, rand()]
            fp = evaluate(Interpolate(f, c, p))
            @test isapprox(fp["g"], p^5, atol=1e-10)
        end
    end

    @testset "integrate N=$N (a,b)=$ab k=$k dealias=$dealias T=$T" for
            N in N_range,
            ab in ab_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a0=a, b0=b, a=a + k, b=b + k, bounds=(0, 3), dealias=dealias)
        x = local_grid(d, basis, scale=1)
        f = Field(d, bases=basis)
        f["g"] = @. 6 * x^5
        g = evaluate(Integrate(f, c))
        @test isapprox(g["g"], 3^6, atol=1e-8)
    end

    @testset "average N=$N (a,b)=$ab k=$k dealias=$dealias T=$T" for
            N in N_range,
            ab in ab_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a0=a, b0=b, a=a + k, b=b + k, bounds=(0, 3), dealias=dealias)
        x = local_grid(d, basis, scale=1)
        f = Field(d, bases=basis)
        f["g"] = @. 6 * x^5
        g = evaluate(Average(f, c))
        @test isapprox(g["g"], 3^6 / 3, atol=1e-8)
    end

    @testset "lift N=$N (a,b)=$ab k=$k n=$n dealias=$dealias T=$T" for
            N in N_range,
            ab in ab_range,
            k in k_range,
            n in [-1, -2],
            dealias in dealias_range,
            T in dtype_range
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a0=a, b0=b, a=a + k, b=b + k, bounds=(0, 3), dealias=dealias)
        x = local_grid(d, basis, scale=1)
        lift_basis = derivative_basis(basis, k)
        f = Field(d, bases=lift_basis)
        f["c"][end + n + 1] = 2  # Julia 1-based indexing for negative index
        tau = Field(d)
        tau["g"] = 2
        g = evaluate(Lift(tau, lift_basis, n))
        @test isapprox(g["g"], f["g"], atol=1e-12)
    end

end

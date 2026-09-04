"""Tests for Cardinal basis operators: ConvertConstant, Interpolate, Integrate, Average."""

using Test
using Dedalus

@testset "Cardinal Operators" begin

    N_range = [5, 10]
    dtype_range = [Float64, ComplexF64]

    @testset "convert constant N=$N T=$T layout=$layout" for
            N in N_range,
            T in dtype_range,
            layout in ["g", "c"]
        c = Coordinate("n")
        dist = Distributor(c, dtype=T)
        b = CardinalBasis(c, size=N)
        n = local_grid(dist, b, scale=1)
        f = Field(dist)
        f["g"] = 3
        change_layout!(f, layout)
        g = evaluate(Convert(f, b))
        @test isapprox(g["g"], 3 * ones(N), atol=1e-12)
    end

    @testset "interpolate N=$N T=$T index=$index" for
            N in N_range,
            T in dtype_range,
            index in [0, 2, -1]
        c = Coordinate("n")
        dist = Distributor(c, dtype=T)
        b = CardinalBasis(c, size=N)
        n = local_grid(dist, b, scale=1)
        f = Field(dist, bases=b)
        fill_random!(f, "g")
        g = evaluate(Interpolate(f, c, index))
        # Convert Python index to Julia indexing
        jl_index = index >= 0 ? index + 1 : N + index + 1
        @test isapprox(g["g"], f["g"][jl_index], atol=1e-12)
    end

    @testset "integrate N=$N T=$T" for
            N in N_range,
            T in dtype_range
        c = Coordinate("n")
        dist = Distributor(c, dtype=T)
        b = CardinalBasis(c, size=N)
        n = local_grid(dist, b, scale=1)
        f = Field(dist, bases=b)
        fill_random!(f, "g")
        g = evaluate(Integrate(f, c))
        @test isapprox(g["g"], sum(f["g"]), atol=1e-12)
    end

    @testset "integrate constant N=$N T=$T" for
            N in N_range,
            T in dtype_range
        c = Coordinate("n")
        dist = Distributor(c, dtype=T)
        b = CardinalBasis(c, size=N)
        n = local_grid(dist, b, scale=1)
        f = Field(dist, bases=b)
        f["g"] .= 3
        g = evaluate(Integrate(f, c))
        @test isapprox(g["g"], 3 * N, atol=1e-12)
    end

    @testset "average N=$N T=$T" for
            N in N_range,
            T in dtype_range
        c = Coordinate("n")
        dist = Distributor(c, dtype=T)
        b = CardinalBasis(c, size=N)
        n = local_grid(dist, b, scale=1)
        f = Field(dist, bases=b)
        fill_random!(f, "g")
        g = evaluate(Average(f, c))
        @test isapprox(g["g"], sum(f["g"]) / N, atol=1e-12)
    end

    @testset "average constant N=$N T=$T" for
            N in N_range,
            T in dtype_range
        c = Coordinate("n")
        dist = Distributor(c, dtype=T)
        b = CardinalBasis(c, size=N)
        n = local_grid(dist, b, scale=1)
        f = Field(dist, bases=b)
        f["g"] .= 7
        g = evaluate(Average(f, c))
        @test isapprox(g["g"], 7, atol=1e-12)
    end

end

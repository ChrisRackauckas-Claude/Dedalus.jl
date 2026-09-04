"""Tests for Fourier convert, differentiate, interpolate, integrate, average."""

using Test
using Dedalus

@testset "Fourier Operators" begin

    N_range = [10]
    bounds_range = [(0.5, 1.666)]
    dealias_range = [1]
    dtype_range = [Float64, ComplexF64]

    @testset "convert constant N=$N bounds=$bounds dealias=$dealias T=$T layout=$layout" for
            N in N_range,
            bounds in bounds_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["g", "c"]
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        b = Fourier(c, size=N, bounds=bounds, dealias=dealias, dtype=T)
        x = local_grid(d, b, scale=1)
        f = Field(d)
        f["g"] = 1
        change_layout!(f, layout)
        g = evaluate(Convert(f, b))
        @test isapprox(g["g"], f["g"], atol=1e-12)
    end

    @testset "differentiate N=$N bounds=$bounds dealias=$dealias T=$T" for
            N in N_range,
            bounds in bounds_range,
            dealias in dealias_range,
            T in dtype_range
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        b = Fourier(c, size=N, bounds=bounds, dealias=dealias, dtype=T)
        x = local_grid(d, b, scale=1)
        f = Field(d, bases=b)
        k = 4 * pi / (bounds[2] - bounds[1])
        f["g"] = @. 1 + sin(k * x + 0.1)
        g = evaluate(Differentiate(f, c))
        @test isapprox(g["g"], @.(k * cos(k * x + 0.1)), atol=1e-10)
    end

    @testset "interpolate N=$N bounds=$bounds dealias=$dealias T=$T" for
            N in N_range,
            bounds in bounds_range,
            dealias in dealias_range,
            T in dtype_range
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        b = Fourier(c, size=N, bounds=bounds, dealias=dealias, dtype=T)
        x = local_grid(d, b, scale=1)
        f = Field(d, bases=b)
        k = 4 * pi / (bounds[2] - bounds[1])
        f["g"] = @. 1 + sin(k * x + 0.1)
        for p in [bounds[1], bounds[2], bounds[1] + (bounds[2] - bounds[1]) * rand()]
            g = evaluate(Interpolate(f, c, p))
            @test isapprox(g["g"], 1 + sin(k * p + 0.1), atol=1e-10)
        end
    end

    @testset "integrate N=$N bounds=$bounds dealias=$dealias T=$T" for
            N in N_range,
            bounds in bounds_range,
            dealias in dealias_range,
            T in dtype_range
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        b = Fourier(c, size=N, bounds=bounds, dealias=dealias, dtype=T)
        x = local_grid(d, b, scale=1)
        f = Field(d, bases=b)
        k = 4 * pi / (bounds[2] - bounds[1])
        f["g"] = @. 1 + sin(k * x + 0.1)
        g = evaluate(Integrate(f, c))
        @test isapprox(g["g"], bounds[2] - bounds[1], atol=1e-10)
    end

    @testset "average N=$N bounds=$bounds dealias=$dealias T=$T" for
            N in N_range,
            bounds in bounds_range,
            dealias in dealias_range,
            T in dtype_range
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        b = Fourier(c, size=N, bounds=bounds, dealias=dealias, dtype=T)
        x = local_grid(d, b, scale=1)
        f = Field(d, bases=b)
        k = 4 * pi / (bounds[2] - bounds[1])
        f["g"] = @. 1 + sin(k * x + 0.1)
        g = evaluate(Average(f, c))
        @test isapprox(g["g"], 1, atol=1e-10)
    end

end

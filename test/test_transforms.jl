"""Tests for forward/backward transforms on Jacobi and Fourier bases."""

using Test
using Dedalus

@testset "Transforms" begin

    # ---- 1D Fourier transforms ----

    @testset "CF scalar roundtrip N=$N dealias=$dealias" for
            N in [8],
            dealias in [0.5, 1.0, 1.5]
        c = Coordinate("x")
        d = Distributor([c])
        xb = ComplexFourier(c, size=N, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, xb, scale=dealias)
        u = Field(dist=d, bases=(xb,), dtype=ComplexF64)
        preset_scales!(u, dealias)
        ug = @. exp(2 * pi * im * x)
        u["g"] = ug
        u["c"]  # force forward transform
        @test isapprox(u["g"], ug, atol=1e-12)
    end

    @testset "RF scalar roundtrip N=$N dealias=$dealias" for
            N in [8],
            dealias in [0.5, 1.0, 1.5]
        c = Coordinate("x")
        d = Distributor([c])
        xb = RealFourier(c, size=N, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, xb, scale=dealias)
        u = Field(dist=d, bases=(xb,), dtype=Float64)
        preset_scales!(u, dealias)
        ug = @. cos(2 * pi * x + pi / 4)
        u["g"] = ug
        u["c"]
        @test isapprox(u["g"], ug, atol=1e-12)
    end

    # ---- 1D Jacobi transforms ----

    @testset "J scalar roundtrip a=$a b=$b N=$N dealias=$dealias T=$T" for
            a in [-0.5, 0.0],
            b in [-0.5, 0.0],
            N in [8],
            dealias in [0.5, 1.0, 1.5],
            T in [Float64, ComplexF64]
        c = Coordinate("x")
        d = Distributor([c])
        xb = Jacobi(c, a=a, b=b, size=N, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, xb, scale=dealias)
        u = Field(dist=d, bases=(xb,), dtype=T)
        preset_scales!(u, dealias)
        ug = @. 2 * x^2 - 1
        u["g"] = ug
        u["c"]
        @test isapprox(u["g"], ug, atol=1e-12)
    end

    # ---- 1D Fourier library backward/forward tests ----

    @testset "Real Fourier libraries backward N=$N dealias=$dealias" for
            N in [16],
            dealias in [0.5, 1.0, 1.5]
        c = Coordinate("x")
        d = Distributor([c])
        # Matrix reference
        b_mat = RealFourier(c, size=N, bounds=(0, 2 * pi), dealias=dealias, library="matrix")
        u_mat = Field(dist=d, bases=(b_mat,), dtype=Float64)
        preset_scales!(u_mat, dealias)
        u_mat["c"] = randn(N)
        # Library
        for library in ["fftpack", "scipy", "fftw", "fftw_hc"]
            b_lib = RealFourier(c, size=N, bounds=(0, 2 * pi), dealias=dealias, library=library)
            u_lib = Field(dist=d, bases=(b_lib,), dtype=Float64)
            preset_scales!(u_lib, dealias)
            u_lib["c"] = u_mat["c"]
            @test isapprox(u_mat["g"], u_lib["g"], atol=1e-12)
        end
    end

    @testset "Real Fourier libraries forward N=$N dealias=$dealias" for
            N in [16],
            dealias in [0.5, 1.0, 1.5]
        c = Coordinate("x")
        d = Distributor([c])
        b_mat = RealFourier(c, size=N, bounds=(0, 2 * pi), dealias=dealias, library="matrix")
        u_mat = Field(dist=d, bases=(b_mat,), dtype=Float64)
        preset_scales!(u_mat, dealias)
        u_mat["g"] = randn(Int(ceil(dealias * N)))
        for library in ["fftpack", "scipy", "fftw", "fftw_hc"]
            b_lib = RealFourier(c, size=N, bounds=(0, 2 * pi), dealias=dealias, library=library)
            u_lib = Field(dist=d, bases=(b_lib,), dtype=Float64)
            preset_scales!(u_lib, dealias)
            u_lib["g"] = u_mat["g"]
            @test isapprox(u_mat["c"], u_lib["c"], atol=1e-12)
        end
    end

    @testset "Chebyshev libraries backward N=$N alpha=$alpha dealias=$dealias T=$T" for
            N in [15, 16],
            alpha in [0, 1, 2],
            dealias in [0.5, 1.0, 1.5],
            T in [Float64, ComplexF64]
        c = Coordinate("x")
        d = Distributor([c])
        b_mat = Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library="matrix")
        u_mat = Field(dist=d, bases=(b_mat,), dtype=T)
        preset_scales!(u_mat, dealias)
        u_mat["c"] = randn(N)
        for library in ["scipy_dct", "fftw_dct"]
            b_lib = Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
            u_lib = Field(dist=d, bases=(b_lib,), dtype=T)
            preset_scales!(u_lib, dealias)
            u_lib["c"] = u_mat["c"]
            @test isapprox(u_mat["g"], u_lib["g"], atol=1e-12)
        end
    end

    @testset "Chebyshev libraries forward N=$N alpha=$alpha dealias=$dealias T=$T" for
            N in [15, 16],
            alpha in [0, 1, 2],
            dealias in [0.5, 1.0, 1.5],
            T in [Float64, ComplexF64]
        c = Coordinate("x")
        d = Distributor([c])
        b_mat = Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library="matrix")
        u_mat = Field(dist=d, bases=(b_mat,), dtype=T)
        preset_scales!(u_mat, dealias)
        u_mat["g"] = randn(Int(ceil(dealias * N)))
        for library in ["scipy_dct", "fftw_dct"]
            b_lib = Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
            u_lib = Field(dist=d, bases=(b_lib,), dtype=T)
            preset_scales!(u_lib, dealias)
            u_lib["g"] = u_mat["g"]
            @test isapprox(u_mat["c"], u_lib["c"], atol=1e-12)
        end
    end

    # ---- 2D Cartesian transforms ----

    @testset "CF-CF scalar roundtrip Nx=$Nx Ny=$Ny dx=$dx dy=$dy" for
            Nx in [8],
            Ny in [8],
            dx in [0.5, 1.0, 1.5],
            dy in [0.5, 1.0, 1.5]
        c = CartesianCoordinates("x", "y")
        d = Distributor((c,))
        xb = ComplexFourier(c.coords[1], size=Nx, bounds=(0, pi), dealias=dx)
        yb = ComplexFourier(c.coords[2], size=Ny, bounds=(0, pi), dealias=dy)
        x = local_grid(d, xb, scale=dx)
        y = local_grid(d, yb, scale=dy)
        f = Field(dist=d, bases=(xb, yb), dtype=ComplexF64)
        preset_scales!(f, (dx, dy))
        fg = @. exp(2im * x) * exp(2im * y + im * pi / 3) + 3 + exp(2im * y)
        f["g"] = fg
        f["c"]
        @test isapprox(f["g"], fg, atol=1e-12)
    end

    @testset "RF-RF scalar roundtrip Nx=$Nx Ny=$Ny dx=$dx dy=$dy" for
            Nx in [8],
            Ny in [8],
            dx in [0.5, 1.0, 1.5],
            dy in [0.5, 1.0, 1.5]
        c = CartesianCoordinates("x", "y")
        d = Distributor((c,))
        xb = RealFourier(c.coords[1], size=Nx, bounds=(0, pi), dealias=dx)
        yb = RealFourier(c.coords[2], size=Ny, bounds=(0, pi), dealias=dy)
        x = local_grid(d, xb, scale=dx)
        y = local_grid(d, yb, scale=dy)
        f = Field(dist=d, bases=(xb, yb), dtype=Float64)
        preset_scales!(f, (dx, dy))
        fg = @. sin(2 * x) + cos(2 * y + pi / 3) + 3 + sin(2 * y)
        f["g"] = fg
        f["c"]
        @test isapprox(f["g"], fg, atol=1e-12)
    end

    @testset "CF-J scalar roundtrip a=$a b=$b Nx=$Nx Ny=$Ny dx=$dx dy=$dy" for
            a in [-0.5, 0.0],
            b in [-0.5, 0.0],
            Nx in [8],
            Ny in [8],
            dx in [0.5, 1.0, 1.5],
            dy in [0.5, 1.0, 1.5]
        c = CartesianCoordinates("x", "y")
        d = Distributor((c,))
        xb = ComplexFourier(c.coords[1], size=Nx, bounds=(0, pi), dealias=dx)
        yb = Jacobi(c.coords[2], a=a, b=b, size=Ny, bounds=(0, 1), dealias=dy)
        x = local_grid(d, xb, scale=dx)
        y = local_grid(d, yb, scale=dy)
        f = Field(dist=d, bases=(xb, yb), dtype=ComplexF64)
        preset_scales!(f, (dx, dy))
        fg = @. sin(2 * x) * y^5
        f["g"] = fg
        f["c"]
        @test isapprox(f["g"], fg, atol=1e-12)
    end

    @testset "CF-J vector roundtrip a=$a b=$b Nx=$Nx Ny=$Ny dx=$dx dy=$dy" for
            a in [-0.5, 0.0],
            b in [-0.5, 0.0],
            Nx in [8],
            Ny in [8],
            dx in [0.5, 1.0, 1.5],
            dy in [0.5, 1.0, 1.5]
        c = CartesianCoordinates("x", "y")
        d = Distributor((c,))
        xb = ComplexFourier(c.coords[1], size=Nx, bounds=(0, pi), dealias=dx)
        yb = Jacobi(c.coords[2], a=a, b=b, size=Ny, bounds=(0, 1), dealias=dy)
        x = local_grid(d, xb, scale=dx)
        y = local_grid(d, yb, scale=dy)
        u = Field(dist=d, bases=(xb, yb), tensorsig=(c,), dtype=ComplexF64)
        preset_scales!(u, (dx, dy))
        ug = cat((@. cos(2 * x) * 2 * y^2), (@. sin(2 * x) * y + y); dims=1)
        u["g"] = ug
        u["c"]
        @test isapprox(u["g"], ug, atol=1e-12)
    end

    @testset "CF-J 1d vector roundtrip a=$a b=$b Nx=$Nx Ny=$Ny dx=$dx dy=$dy" for
            a in [-0.5, 0.0],
            b in [-0.5, 0.0],
            Nx in [8],
            Ny in [8],
            dx in [0.5, 1.0, 1.5],
            dy in [0.5, 1.0, 1.5]
        c = CartesianCoordinates("x", "y")
        d = Distributor((c,))
        xb = ComplexFourier(c.coords[1], size=Nx, bounds=(0, pi), dealias=dx)
        yb = Jacobi(c.coords[2], a=a, b=b, size=Ny, bounds=(0, 1), dealias=dy)
        x = local_grid(d, xb, scale=dx)
        y = local_grid(d, yb, scale=dy)
        v = Field(dist=d, bases=(xb,), tensorsig=(c,), dtype=ComplexF64)
        preset_scales!(v, (dx, dy))
        vg = cat((@. cos(2 * x) * 2), (@. sin(2 * x) + 1); dims=1)
        v["g"] = vg
        v["c"]
        @test isapprox(v["g"], vg, atol=1e-12)
    end

end

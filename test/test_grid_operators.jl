"""Tests for ufuncs and GeneralFunction operators (Cartesian only)."""

using Test
using Dedalus

@testset "Grid Operators" begin

    N_range = [16]
    dealias_range = [1]
    dtype_range = [Float64, ComplexF64]

    # List of standard ufuncs to test
    ufunc_list = [sin, cos, tan, exp, log, sqrt, abs,
                  sinh, cosh, tanh, asin, acos, atan,
                  asinh, atanh]

    @testset "Jacobi ufunc field N=$N (a,b)=$ab dealias=$dealias T=$T func=$func" for
            N in N_range,
            ab in [(-0.5, -0.5), (0.0, 0.0)],
            dealias in dealias_range,
            T in dtype_range,
            func in ufunc_list
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a=a, b=b, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, basis, scale=1)
        f = Field(d, bases=basis)
        if func === acosh
            f["g"] = @. 1 + x^2
        else
            f["g"] = @. x^2
        end
        result = evaluate(UnaryGridFunction(func, f))
        @test isapprox(result["g"], func.(f["g"]), atol=1e-12)
    end

    @testset "acosh Jacobi ufunc field N=$N (a,b)=$ab dealias=$dealias T=$T" for
            N in N_range,
            ab in [(-0.5, -0.5), (0.0, 0.0)],
            dealias in dealias_range,
            T in dtype_range
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a=a, b=b, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, basis, scale=1)
        f = Field(d, bases=basis)
        f["g"] = @. 1 + x^2
        result = evaluate(UnaryGridFunction(acosh, f))
        @test isapprox(result["g"], acosh.(f["g"]), atol=1e-12)
    end

    @testset "Jacobi GeneralFunction coord N=$N (a,b)=$ab dealias=$dealias T=$T" for
            N in N_range,
            ab in [(-0.5, -0.5), (0.0, 0.0)],
            dealias in dealias_range,
            T in dtype_range
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a=a, b=b, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, basis, dealias)
        f = Field(d, bases=basis)
        F(x_val) = sin.(x_val)
        F_op(args...) = GeneralFunction(d, domain(f), layout="g",
                                        tensorsig=tensorsig(f), dtype=T,
                                        func=F, args=args)
        @test isapprox(F_op(x)["g"], F(x), atol=1e-12)
    end

    @testset "Jacobi GeneralFunction field N=$N (a,b)=$ab dealias=$dealias T=$T" for
            N in N_range,
            ab in [(-0.5, -0.5), (0.0, 0.0)],
            dealias in dealias_range,
            T in dtype_range
        a, b = ab
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        basis = Jacobi(c, size=N, a=a, b=b, bounds=(0, 1), dealias=dealias)
        x = local_grid(d, basis, dealias)
        f = Field(d, bases=basis)
        preset_scales!(f, dealias)
        f["g"] = cos.(x)
        F(x_val, fld) = 2 .* fld["g"] .+ sin.(x_val)
        F_op(args...) = GeneralFunction(d, domain(f), layout="g",
                                        tensorsig=tensorsig(f), dtype=T,
                                        func=F, args=args)
        @test isapprox(F_op(x, f)["g"], F(x, f), atol=1e-12)
    end

end

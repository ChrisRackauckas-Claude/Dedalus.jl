using Test
using Dedalus

@testset "Distributor" begin
    @testset "serial 1D distributor" begin
        c = CartesianCoordinates("x")
        d = Distributor(c, Float64)
        @test d.dim == 1
        @test isempty(d.mesh)
    end

    @testset "serial multi-dimensional distributor" begin
        c = CartesianCoordinates("x", "y")
        d = Distributor(c, Float64)
        @test d.dim == 2
        @test isempty(d.mesh)
    end

    @testset "basis from equal-but-not-identical coordinate" begin
        # The basis caches key on `==` (name equality), so a basis may be bound
        # to a coordinate object that is `==` but not `===` to the distributor's.
        c1 = CartesianCoordinates("x")
        b = RealFourier(c1.coords[1], 8, (0, 2pi))
        c2 = CartesianCoordinates("x")
        d = Distributor(c2, Float64)
        @test Dedalus.first_axis(d, b) == 1
        grids = local_grids(d, b; scales = 1)
        @test length(grids) == 1
        @test vec(grids[1]) ≈ collect(range(0, 2pi, 9)[1:8])
    end

    @testset "field scalar assignment" begin
        c = CartesianCoordinates("x")
        d = Distributor(c, Float64)
        b = RealFourier(c.coords[1], 8, (0, 2pi))
        f = Dedalus.Field(d; name = "f", bases = (b,))
        f["g"] = 2
        @test all(==(2), f["g"])
        f["c"] = 0
        @test all(==(0), f["c"])
    end

    @testset "Fourier group arrays" begin
        # local/global_group_arrays pass each basis a dense per-axis index
        # array; elements_to_groups must map every element to its wavenumber.
        for (T, expected) in [
                (Float64, [0, 0, 1, 1, 2, 2, 3, 3]),
                (ComplexF64, [0, 1, 2, 3, -3, -2, -1]),
            ]
            c = Coordinate("x")
            d = Distributor(c, T)
            b = Fourier(c, length(expected), (0.0, 2pi); dtype = T)
            f = Dedalus.Field(d; bases = b, dtype = T)
            layout = Dedalus.coeff_layout(d)
            for fun in (local_group_arrays, global_group_arrays)
                @test vec(fun(layout, f.domain, (1,))[1]) == expected
            end
            dense = collect(0:(length(expected) - 1))
            @test elements_to_groups(b, (false,), dense) == expected
            @test elements_to_groups(b, (false,), [dense]) == [expected]
        end
    end

    @testset "Fourier valid_elements" begin
        # The -sin(0*x) mode (0-based element 1) is dropped for real Fourier.
        c = Coordinate("x")
        d = Distributor(c, Float64)
        b = Fourier(c, 8, (0.0, 2pi); dtype = Float64)
        f = Dedalus.Field(d; bases = b)
        valid = valid_elements(Dedalus.coeff_layout(d), f.tensorsig, f.domain, 1)
        @test vec(valid) == Bool[1, 0, 1, 1, 1, 1, 1, 1]
        @test vec(valid_modes(f)) == Bool[1, 0, 1, 1, 1, 1, 1, 1]
    end

    @testset "domain get_coord" begin
        c = CartesianCoordinates("x")
        d = Distributor(c, Float64)
        b = Fourier(c.coords[1], 8, (0.0, 2pi); dtype = Float64)
        f = Dedalus.Field(d; bases = b)
        @test get_coord(f.domain, "x").name == "x"
        @test_throws ArgumentError get_coord(f.domain, "y")
    end

    @testset "basis algebra nothing fallbacks" begin
        c = Coordinate("x")
        b = Fourier(c, 8, (0.0, 2pi); dtype = Float64)
        @test basis_mul(nothing, b) === b
        @test basis_mul(nothing, nothing) === nothing
        @test basis_matmul(nothing, b) === b
        @test Dedalus.interpolate_basis(b, 0.0) === nothing
    end

    @testset "Fourier group matrices" begin
        c = Coordinate("x")
        d = Distributor(c, Float64)
        br = Fourier(c, 8, (0.0, 2pi); dtype = Float64)
        @test wavenumbers(br) == [0, 0, 1, 1, 2, 2, 3, 3]
        @test Dedalus.differentiate_real_fourier_matrix(2, br) == [0.0 -2.0; 2.0 0.0]
        bc = Fourier(c, 7, (0.0, 2pi); dtype = ComplexF64)
        @test wavenumbers(bc) == [0, 1, 2, 3, -3, -2, -1]
        @test Dedalus.differentiate_complex_fourier_matrix(2, bc) == reshape([2im], 1, 1)
    end
end

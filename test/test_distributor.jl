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

    @testset "field layout transforms (serial)" begin
        c = CartesianCoordinates("x")
        d = Distributor(c, Float64)
        b = RealFourier(c.coords[1], 8, (0, 2pi))
        x = local_grids(d, b; scales = 1)[1]

        f = Dedalus.Field(d; name = "f", bases = (b,))
        @test size(f["g"]) == (8,)
        f["g"] = sin.(x)
        fc = f["c"]
        @test size(fc) == (8,)
        @test count(v -> abs(v) > 1.0e-12, fc) == 1
        @test f["g"] ≈ sin.(x)

        lf = Dedalus.LockedField(d; name = "lf", bases = (b,))
        Dedalus.lock_to_layouts!(lf, d.coeff_layout)
        @test_throws ArgumentError Dedalus.towards_grid_space!(lf)
        Dedalus.lock_to_layouts!(lf, d.coeff_layout, d.grid_layout)
        Dedalus.towards_grid_space!(lf)
        @test lf.layout === d.grid_layout
        Dedalus.towards_coeff_space!(lf)
        @test lf.layout === d.coeff_layout
    end
end

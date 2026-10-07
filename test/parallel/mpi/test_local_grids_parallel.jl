"""
Local grids and modes under a distributed mesh=(2,2).

Each rank's `local_grids`/`local_modes` must cover exactly its own part of the
field data, i.e. the global grid sliced by the grid layout's local elements.
Requires MPI with at least 4 processes.
"""

using Test
using MPI
using Dedalus

@testset "Local grids parallel" begin

    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)

    if nprocs < 4
        @warn "test_local_grids_parallel requires at least 4 MPI processes; skipping."
        @test_broken false
        return
    end

    @testset "cartesian z=$zname scales=$scales" for
        (zname, zb_ctor) in (
                ("RealFourier", c -> RealFourier(c, 16, (0, 2pi))),
                ("ChebyshevT", c -> ChebyshevT(c, 12, (0, 1))),
            ),
            scales in (1, 1.5)
        c = CartesianCoordinates("x", "y", "z")
        d = Distributor(c, Float64; mesh = (2, 2))
        xb = RealFourier(c.coords[1], 16, (0, 2pi))
        yb = RealFourier(c.coords[2], 8, (0, 2pi))
        zb = zb_ctor(c.coords[3])
        bases = (xb, yb, zb)
        u = Field(d, name = "u", bases = bases, dtype = Float64)
        Dedalus.change_scales!(u, scales)
        layout_scales = Dedalus.remedy_scales(d, scales)
        grid_shape = Dedalus.local_shape(Dedalus.grid_layout(d), u.domain, layout_scales)

        local_g = local_grids(d, bases...; scales = scales)
        sl = Dedalus.slices(Dedalus.grid_layout(d), u.domain, layout_scales)
        for ax in 1:3
            global_g = Dedalus.global_grid(bases[ax], d, scales)
            @test size(local_g[ax], ax) == grid_shape[ax]
            @test vec(local_g[ax]) == vec(global_g)[sl[ax]]
        end

        x, y, z = local_g
        u["g"] = @. sin(x) * cos(2y) * z
        @test u["g"] ≈ @. sin(x) * cos(2y) * z

        global_lengths = [length(Dedalus.global_grid(bases[ax], d, scales)) for ax in 1:3]
        @test count(ax -> grid_shape[ax] < global_lengths[ax], 1:3) == 2
        for ax in 1:3
            counts = MPI.Allgather(Int32(length(local_g[ax])), comm)
            gathered = MPI.Allgatherv(vec(local_g[ax]), counts, comm)
            @test sort(unique(gathered)) == sort(vec(Dedalus.global_grid(bases[ax], d, scales)))
        end

        coeff_shape = Dedalus.local_shape(Dedalus.coeff_layout(d), u.domain, Dedalus.remedy_scales(d, 1))
        for ax in 1:3
            modes = local_modes(d, bases[ax])
            @test length(modes) == coeff_shape[ax]
            @test size(modes, ax) == length(modes)
        end
    end

    @testset "annulus dealias=$dealias" for dealias in (1, 1.5)
        c = PolarCoordinates("phi", "r")
        d = Distributor(c, Float64; mesh = (4,))
        b = AnnulusBasis(c, (16, 12), Float64; radii = (0.5, 2), dealias = (dealias, dealias))
        u = Field(d, name = "u", bases = (b,), dtype = Float64)
        Dedalus.change_scales!(u, dealias)
        grid_shape = Dedalus.local_shape(Dedalus.grid_layout(d), u.domain, Dedalus.remedy_scales(d, dealias))
        phi, r = local_grids(d, b; scales = dealias)
        global_phi, global_r = Dedalus.global_grids(b, d, (dealias, dealias))
        @test size(phi, 1) == grid_shape[1]
        @test size(r, 2) == grid_shape[2]
        for (loc, glob) in ((phi, global_phi), (r, global_r))
            counts = MPI.Allgather(Int32(length(loc)), comm)
            gathered = MPI.Allgatherv(vec(loc), counts, comm)
            @test sort(unique(gathered)) == sort(vec(glob))
        end
        u["g"] = @. r^2 * cos(phi)
        @test u["g"] ≈ @. r^2 * cos(phi)
    end

    @testset "ball dealias=$dealias" for dealias in (1, 1.5)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, Float64; mesh = (2, 2))
        b = BallBasis(c, (8, 10, 8), Float64; radius = 1.5, dealias = (dealias, dealias, dealias))
        u = Field(d, name = "u", bases = (b,), dtype = Float64)
        Dedalus.change_scales!(u, dealias)
        grid_shape = Dedalus.local_shape(Dedalus.grid_layout(d), u.domain, Dedalus.remedy_scales(d, dealias))
        phi, theta, r = local_grids(d, b; scales = dealias)
        @test size(phi, 1) == grid_shape[1]
        @test size(theta, 2) == grid_shape[2]
        @test size(r, 3) == grid_shape[3]
        u["g"] = @. r^2 * cos(theta) + 0 * phi
        @test u["g"] ≈ @. r^2 * cos(theta) + 0 * phi
    end
end

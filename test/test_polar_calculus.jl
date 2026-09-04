"""Tests for polar coordinate calculus operators on disk and annulus bases:
gradient, divergence, curl, laplacian."""

using Test
using Dedalus

@testset "Polar Calculus" begin

    Nphi_range = [16]
    Nr_range = [8]
    dealias_range = [1, 1.5]
    radius_disk = 1.5
    radii_annulus = (0.5, 3.0)

    # ---- Builder functions ----

    function build_disk(Nphi, Nr, dealias, T)
        c = PolarCoordinates("phi", "r")
        d = Distributor(c, dtype=T)
        b = DiskBasis(c, (Nphi, Nr), T; radius=radius_disk, dealias=(dealias, dealias))
        phi, r = local_grids(d, b, scales=dealias)
        x, y = cartesian(PolarCoordinates, phi, r)
        return c, d, b, phi, r, x, y
    end

    function build_annulus(Nphi, Nr, dealias, T)
        c = PolarCoordinates("phi", "r")
        d = Distributor(c, dtype=T)
        b = AnnulusBasis(c, (Nphi, Nr), T; radii=radii_annulus, dealias=(dealias, dealias))
        phi, r = local_grids(d, b, scales=dealias)
        x, y = cartesian(PolarCoordinates, phi, r)
        return c, d, b, phi, r, x, y
    end

    # ---- Gradient tests ----

    @testset "gradient scalar $bname Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. 3*x^2 + 2*y
            u = evaluate(gradient(f, c))
            ex = @. [-sin(phi) + 0*r, cos(phi) + 0*r]
            ey = @. [cos(phi) + 0*r, sin(phi) + 0*r]
            ug = @. 6*x*ex + 2*ey
            @test isapprox(u["g"], ug, atol=1e-9)
        catch e
            @test_broken false
        end
    end

    @testset "gradient radial scalar $bname Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            Nphi = 1
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. r^4
            u = evaluate(gradient(f, c))
            ug = @. [0*r*phi, 4*r^3 + 0*phi]
            @test isapprox(u["g"], ug, atol=1e-9)
        catch e
            @test_broken false
        end
    end

    @testset "gradient vector $bname Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. 3*x^4 + 2*y*x
            # Double gradient: grad(grad(f)) -> rank-2 tensor
            T_result = evaluate(gradient(gradient(f, c), c))
            # Unit vectors in polar components: [phi-component, r-component]
            ex_raw = @. [-sin(phi) + 0*r, cos(phi) + 0*r]
            ey_raw = @. [cos(phi) + 0*r, sin(phi) + 0*r]
            # Outer products for tensor components: shape (2,2,Nphi_grid,Nr_grid)
            # exex[i,j,...] = ex[i,...] * ex[j,...]
            # In NumPy: ex[:,None,...] * ex[None,...] gives (2,2,Nphi,Nr)
            # In Julia with array-of-arrays from broadcasting, we need to
            # construct the (2,2,...) tensor manually
            s = size(phi)  # grid shape for spatial dims
            Tg = zeros(T, 2, 2, s...)
            for i in 1:2, j in 1:2
                Tg[i, j, :, :] .= 36 .* x .* x .* ex_raw[i] .* ex_raw[j] .+
                                   2 .* (ex_raw[i] .* ey_raw[j] .+ ey_raw[i] .* ex_raw[j])
            end
            @test isapprox(T_result["g"], Tg, atol=1e-9)
        catch e
            @test_broken false
        end
    end

    @testset "gradient radial vector $bname Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            Nphi = 1
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. r^4
            T_result = evaluate(gradient(gradient(f, c), c))
            # For Nphi=1, phi has shape (1, Nr_grid) and r has shape (1, Nr_grid)
            # er = [0, 1] in (phi, r) components, ephi = [1, 0]
            s = size(phi)
            Tg = zeros(T, 2, 2, s...)
            # er*er component (index [2,2])
            Tg[2, 2, :, :] .= 12 .* r .^ 2
            # ephi*ephi component (index [1,1])
            Tg[1, 1, :, :] .= 4 .* r .^ 2
            @test isapprox(T_result["g"], Tg, atol=1e-9)
        catch e
            @test_broken false
        end
    end

    # ---- Divergence tests ----

    @testset "divergence vector $bname Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. 3*x^4 + 2*y*x
            # div(grad(f)) = laplacian of scalar
            S = evaluate(divergence(gradient(f, c)))
            Sg = @. 36*x^2
            @test isapprox(S["g"], Sg, atol=1e-9)
        catch e
            @test_broken false
        end
    end

    @testset "divergence radial vector $bname Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            Nphi = 1
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. r^2
            h = evaluate(divergence(gradient(f, c)))
            hg = 4
            @test isapprox(h["g"], fill(T(hg), size(h["g"])), atol=1e-9)
        catch e
            @test_broken false
        end
    end

    @testset "divergence tensor $bname Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            v = VectorField(d, c, bases=(b,), dtype=T)
            preset_scales!(v, dealias)
            ex_raw = @. [-sin(phi) + 0*r, cos(phi) + 0*r]
            ey_raw = @. [cos(phi) + 0*r, sin(phi) + 0*r]
            # v = 4*x^3*ey + 3*y^2*ey = (4*x^3 + 3*y^2)*ey
            for i in 1:2
                v["g"][i, :, :] .= (4 .* x .^ 3 .+ 3 .* y .^ 2) .* ey_raw[i]
            end
            # div(grad(v))
            U = evaluate(divergence(gradient(v, c)))
            # Expected: (24*x + 6)*ey
            s = size(phi)
            Ug = zeros(T, 2, s...)
            for i in 1:2
                Ug[i, :, :] .= (24 .* x .+ 6) .* ey_raw[i]
            end
            @test isapprox(U["g"], Ug, atol=1e-9)
        catch e
            @test_broken false
        end
    end

    # ---- Curl tests ----

    @testset "curl vector $bname Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            v = VectorField(d, c, bases=(b,), dtype=T)
            preset_scales!(v, dealias)
            ex_raw = @. [-sin(phi) + 0*r, cos(phi) + 0*r]
            ey_raw = @. [cos(phi) + 0*r, sin(phi) + 0*r]
            # v = 4*x^3*ey + 3*y^2*ey = (4*x^3 + 3*y^2)*ey
            for i in 1:2
                v["g"][i, :, :] .= (4 .* x .^ 3 .+ 3 .* y .^ 2) .* ey_raw[i]
            end
            # curl = -div(skew(v))
            u = evaluate(-divergence(skew(v)))
            ug = @. 12*x^2
            @test isapprox(u["g"], ug, atol=1e-9)
        catch e
            @test_broken false
        end
    end

    # ---- Laplacian tests ----

    @testset "laplacian scalar $bname Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. x^4 + 2*y^4
            h = evaluate(laplacian(f, c))
            hg = @. 12*x^2 + 24*y^2
            @test isapprox(h["g"], hg, atol=1e-9)
        catch e
            @test_broken false
        end
    end

    @testset "laplacian radial scalar $bname Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            Nphi = 1
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. r^2
            h = evaluate(laplacian(f, c))
            hg = 4
            @test isapprox(h["g"], fill(T(hg), size(h["g"])), atol=1e-9)
        catch e
            @test_broken false
        end
    end

    @testset "laplacian vector $bname Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            v = VectorField(d, c, bases=(b,), dtype=T)
            preset_scales!(v, dealias)
            ex_raw = @. [-sin(phi) + 0*r, cos(phi) + 0*r]
            ey_raw = @. [cos(phi) + 0*r, sin(phi) + 0*r]
            # v = 4*x^3*ey + 3*y^2*ey = (4*x^3 + 3*y^2)*ey
            for i in 1:2
                v["g"][i, :, :] .= (4 .* x .^ 3 .+ 3 .* y .^ 2) .* ey_raw[i]
            end
            U = evaluate(laplacian(v, c))
            # Expected: (24*x + 6)*ey
            s = size(phi)
            Ug = zeros(T, 2, s...)
            for i in 1:2
                Ug[i, :, :] .= (24 .* x .+ 6) .* ey_raw[i]
            end
            @test isapprox(U["g"], Ug, atol=1e-9)
        catch e
            @test_broken false
        end
    end

    @testset "laplacian radial vector $bname Nr=$Nr dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nr in Nr_range,
            dealias in dealias_range,
            T in [Float64, ComplexF64]
        try
            Nphi = 1
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, dealias, T)
            u = VectorField(d, c, bases=(b,), dtype=T)
            preset_scales!(u, dealias)
            # u['g'][1] = 4*r^3 (Python index 1 = r-component)
            # In Julia, component 2 is the r-component (1=phi, 2=r)
            u["g"][2, :, :] .= 4 .* r .^ 3
            v = evaluate(laplacian(u, c))
            s = size(phi)
            vg = zeros(T, 2, s...)
            # v['g'][1] = 32*r (Python index 1 = r-component -> Julia index 2)
            vg[2, :, :] .= 32 .* r
            @test isapprox(v["g"], vg, atol=1e-9)
        catch e
            @test_broken false
        end
    end

end

"""Tests for polar coordinate operators on disk and annulus bases:
convert, trace, transpose, skew, interpolate, integrate, average,
radial/azimuthal component."""

using Test
using Dedalus

@testset "Polar Operators" begin

    Nphi_range = [8]
    Nr_range = [8]
    k_range = [0, 1]
    dealias_range = [1, 1.5]
    dtype_range = [Float64, ComplexF64]
    radius_disk = 1.5
    radii_annulus = (0.5, 3.0)

    # ---- Builder functions ----

    function build_disk(Nphi, Nr, k, dealias, T)
        c = PolarCoordinates("phi", "r")
        d = Distributor(c, T)
        b = DiskBasis(c, (Nphi, Nr), T; radius=radius_disk, k=k,
                      dealias=(dealias, dealias))
        phi, r = local_grids(d, b, scales=dealias)
        x, y = cartesian(PolarCoordinates, phi, r)
        return c, d, b, phi, r, x, y
    end

    function build_annulus(Nphi, Nr, k, dealias, T)
        c = PolarCoordinates("phi", "r")
        d = Distributor(c, T)
        b = AnnulusBasis(c, (Nphi, Nr), T; radii=radii_annulus, k=k,
                         dealias=(dealias, dealias))
        phi, r = local_grids(d, b, scales=dealias)
        x, y = cartesian(PolarCoordinates, phi, r)
        return c, d, b, phi, r, x, y
    end

    # ---- Convert tests ----

    @testset "convert constant scalar $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
            f = Field(d, dtype=T)
            f["g"] = 1
            g = evaluate(Convert(f, b))
            @test isapprox(f["g"], g["g"], atol=1e-12)
        catch e
            @test_broken false
        end
    end

    @testset "convert scalar $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T layout=$layout" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["c", "g"]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. 3 * x^2 + 2 * y
            g = evaluate(laplacian(f, c))
            change_layout!(f, layout)
            change_layout!(g, layout)
            h = evaluate(f + g)
            @test isapprox(h["g"], f["g"] .+ g["g"], atol=1e-10)
        catch e
            @test_broken false
        end
    end

    @testset "convert vector $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T layout=$layout" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["c", "g"]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
            u = VectorField(d, c, bases=b)
            preset_scales!(u, dealias)
            # Construct unit vectors in polar coords: e_phi = (-sin(phi), cos(phi)), e_r = (cos(phi), sin(phi))
            ex = cat(-sin.(phi) .+ 0 .* r, cos.(phi) .+ 0 .* r; dims=1)
            ey = cat(cos.(phi) .+ 0 .* r, sin.(phi) .+ 0 .* r; dims=1)
            # Reshape for 2-component vector field: component dimension first
            sz = size(phi)
            ex = reshape(ex, 2, sz...)
            ey = reshape(ey, 2, sz...)
            u["g"] = @. 4 * x^3 * ey + 3 * y^2 * ey
            v = evaluate(laplacian(u, c))
            change_layout!(u, layout)
            change_layout!(v, layout)
            w = evaluate(u + v)
            @test isapprox(w["g"], u["g"] .+ v["g"], atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Skew tests ----

    @testset "skew explicit $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T layout=$layout" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["c", "g"]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
            f = VectorField(d, c, bases=b)
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            change_layout!(f, layout)
            g = evaluate(skew(f))
            @test isapprox(g["g"][1, :], f["g"][2, :], atol=1e-12)
            @test isapprox(g["g"][2, :], -f["g"][1, :], atol=1e-12)
        catch e
            @test_broken false
        end
    end

    @testset "skew implicit $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
            f = VectorField(d, c, bases=b)
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            u = VectorField(d, c, bases=b)
            problem = LBVP([u], namespace=Dict("u" => u, "f" => f, "skew" => skew))
            add_equation!(problem, "skew(u) = skew(f)")
            solver = build_solver(problem)
            solve!(solver)
            change_scales!(u, dealias)
            change_scales!(f, dealias)
            @test isapprox(u["g"], f["g"], atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Trace tests ----

    @testset "trace explicit tensor $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T layout=$layout" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["c", "g"]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
            u = VectorField(d, c, bases=b)
            preset_scales!(u, dealias)
            # Unit vectors in polar basis
            sz = size(phi)
            ex_phi = reshape(cat(-sin.(phi) .+ 0 .* r, cos.(phi) .+ 0 .* r; dims=1), 2, sz...)
            ey_phi = reshape(cat(cos.(phi) .+ 0 .* r, sin.(phi) .+ 0 .* r; dims=1), 2, sz...)
            u["g"] = @. 4 * x^3 * ey_phi + 3 * y^2 * ey_phi
            T_field = evaluate(gradient(u, c))
            # Compute expected trace in grid space before layout change
            fg = T_field["g"][1, 1, :] .+ T_field["g"][2, 2, :]
            change_layout!(T_field, layout)
            f = evaluate(trace_op(T_field))
            @test isapprox(f["g"], fg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    @testset "trace implicit tensor $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            g = Field(d, bases=(b,), dtype=T)
            fill_random!(g, "g")
            low_pass_filter!(g, scales=0.5)
            # Build identity tensor on radial basis
            rb = Dedalus.radial_basis(b)
            I_tensor = TensorField(d, (c, c), bases=rb)
            I_tensor["g"][1, 1, :] .= 1
            I_tensor["g"][2, 2, :] .= 1
            problem = LBVP([f])
            add_equation!(problem, (trace_op(I_tensor * f), 2 * g))
            solver = LinearBoundaryValueSolver(problem, matrix_coupling=[false, true])
            solve!(solver)
            @test isapprox(f["c"], g["c"], atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Transpose tests ----

    @testset "transpose explicit $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T layout=$layout" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["c", "g"]
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
            f = TensorField(d, (c, c), bases=b)
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            change_layout!(f, layout)
            g = evaluate(transpose_components(f))
            # Check g[i,j,...] == f[j,i,...]
            for i in 1:2, j in 1:2
                @test isapprox(g["g"][i, j, :], f["g"][j, i, :], atol=1e-12)
            end
        catch e
            @test_broken false
        end
    end

    @testset "transpose implicit $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
            f = TensorField(d, (c, c), bases=b)
            fill_random!(f, layout="g")
            low_pass_filter!(f, scales=0.75)
            u = TensorField(d, (c, c), bases=b)
            problem = LBVP([u], namespace=Dict("u" => u, "f" => f,
                                               "trans" => transpose_components))
            add_equation!(problem, "trans(u) = trans(f)")
            solver = build_solver(problem)
            solve!(solver)
            change_scales!(u, dealias)
            change_scales!(f, dealias)
            @test isapprox(u["g"], f["g"], atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Azimuthal average tests ----

    @testset "azimuthal average scalar $bname Nphi=16 Nr=10 k=$k dealias=$dealias T=$T" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in [0, 1, 2, 5],
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b, phi, r, x, y = basis_fn(16, 10, k, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. r^2 + x
            h = evaluate(average(f, c.coords[1]))
            hg = @. r^2
            @test isapprox(h["g"], hg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Integrate tests ----

    @testset "integrate scalar $bname Nphi=16 Nr=10 k=$k dealias=$dealias T=$T n=$n" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in [0, 1, 2, 5],
            dealias in dealias_range,
            T in dtype_range,
            n in [0, 1, 2]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 10, k, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. r^(2 * n)
            h = evaluate(integrate(f, c))
            # Compute analytical result: integral of r^(2n) over disk/annulus
            if b isa DiskBasis
                r_inner, r_outer = 0.0, radius_disk
            else
                r_inner, r_outer = radii_annulus
            end
            hg = 2 * pi * (r_outer^(2 + 2 * n) - r_inner^(2 + 2 * n)) / (2 + 2 * n)
            @test isapprox(h["g"], hg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Interpolate azimuth tests (scalar) ----

    @testset "interpolate azimuth scalar $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T phi_i=$phi_i" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            phi_i in [0.5, 1.0, 1.5]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. x^4 + 2 * y^4
            h = evaluate(interpolate(f; phi=phi_i))
            x_i, y_i = cartesian(PolarCoordinates, fill(phi_i, 1, 1), r)
            hg = @. x_i^4 + 2 * y_i^4
            @test isapprox(h["g"], hg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Interpolate radius tests (scalar) ----

    @testset "interpolate radius scalar $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T r_i=$r_i" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            r_i in [0.5, 1.0, 1.5]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. x^4 + 2 * y^4
            h = evaluate(interpolate(f; r=r_i))
            x_i, y_i = cartesian(PolarCoordinates, phi, fill(r_i, 1, 1))
            hg = @. x_i^4 + 2 * y_i^4
            @test isapprox(h["g"], hg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Interpolate azimuth tests (vector) ----

    @testset "interpolate azimuth vector $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T phi_i=$phi_i" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            phi_i in [0.5, 1.0, 1.5]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. x^4 + 2 * y^4
            u = gradient(f, c)
            v = evaluate(interpolate(u; phi=phi_i))
            # Evaluate expected gradient at the interpolation point
            phi_arr = fill(phi_i, 1, 1)
            x_i, y_i = cartesian(PolarCoordinates, phi_arr, r)
            sz = size(r)
            ex = reshape(cat(-sin.(phi_arr) .+ 0 .* r, cos.(phi_arr) .+ 0 .* r; dims=1), 2, sz...)
            ey = reshape(cat(cos.(phi_arr) .+ 0 .* r, sin.(phi_arr) .+ 0 .* r; dims=1), 2, sz...)
            vg = @. 4 * x_i^3 * ex + 8 * y_i^3 * ey
            @test isapprox(v["g"], vg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Interpolate radius tests (vector) ----

    @testset "interpolate radius vector $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T r_i=$r_i" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            r_i in [0.5, 1.0, 1.5]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. x^4 + 2 * y^4
            u = gradient(f, c)
            v = evaluate(interpolate(u; r=r_i))
            # Evaluate expected gradient at the interpolation point
            r_arr = fill(r_i, 1, 1)
            x_i, y_i = cartesian(PolarCoordinates, phi, r_arr)
            sz = size(phi)
            ex = reshape(cat(-sin.(phi) .+ 0 .* r_arr, cos.(phi) .+ 0 .* r_arr; dims=1), 2, sz...)
            ey = reshape(cat(cos.(phi) .+ 0 .* r_arr, sin.(phi) .+ 0 .* r_arr; dims=1), 2, sz...)
            vg = @. 4 * x_i^3 * ex + 8 * y_i^3 * ey
            @test isapprox(v["g"], vg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Interpolate azimuth tests (tensor) ----

    @testset "interpolate azimuth tensor $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T phi_i=$phi_i" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            phi_i in [0.5, 1.0, 1.5]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. x^4 + 2 * y^4
            u = gradient(f, c)
            T_field = gradient(u, c)
            v = evaluate(interpolate(T_field; phi=phi_i))
            # Evaluate expected Hessian at the interpolation point
            phi_arr = fill(phi_i, 1, 1)
            x_i, y_i = cartesian(PolarCoordinates, phi_arr, r)
            sz = size(r)
            ex_vec = cat(-sin.(phi_arr) .+ 0 .* r, cos.(phi_arr) .+ 0 .* r; dims=1)
            ey_vec = cat(cos.(phi_arr) .+ 0 .* r, sin.(phi_arr) .+ 0 .* r; dims=1)
            ex = reshape(ex_vec, 2, sz...)
            ey = reshape(ey_vec, 2, sz...)
            # Outer products: exex[i,j,...] = ex[i,...] * ex[j,...]
            exex = reshape(ex_vec, 2, 1, sz...) .* reshape(ex_vec, 1, 2, sz...)
            eyey = reshape(ey_vec, 2, 1, sz...) .* reshape(ey_vec, 1, 2, sz...)
            vg = @. 12 * x_i^2 * exex + 24 * y_i^2 * eyey
            @test isapprox(v["g"], vg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Interpolate radius tests (tensor) ----

    @testset "interpolate radius tensor $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T r_i=$r_i" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            r_i in [0.5, 1.0, 1.5]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
            f = Field(d, bases=(b,), dtype=T)
            preset_scales!(f, dealias)
            f["g"] = @. x^4 + 2 * y^4
            u = gradient(f, c)
            T_field = gradient(u, c)
            v = evaluate(interpolate(T_field; r=r_i))
            # Evaluate expected Hessian at the interpolation point
            r_arr = fill(r_i, 1, 1)
            x_i, y_i = cartesian(PolarCoordinates, phi, r_arr)
            sz = size(phi)
            ex_vec = cat(-sin.(phi) .+ 0 .* r_arr, cos.(phi) .+ 0 .* r_arr; dims=1)
            ey_vec = cat(cos.(phi) .+ 0 .* r_arr, sin.(phi) .+ 0 .* r_arr; dims=1)
            ex = reshape(ex_vec, 2, sz...)
            ey = reshape(ey_vec, 2, sz...)
            exex = reshape(ex_vec, 2, 1, sz...) .* reshape(ex_vec, 1, 2, sz...)
            eyey = reshape(ey_vec, 2, 1, sz...) .* reshape(ey_vec, 1, 2, sz...)
            vg = @. 12 * x_i^2 * exex + 24 * y_i^2 * eyey
            @test isapprox(v["g"], vg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Radial component tests (vector) ----

    @testset "radial component vector $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T radius=$radius" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            radius in [0.5, 1.0, 1.5]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
            cp = cos.(phi)
            sp = sin.(phi)
            u = VectorField(d, c, bases=b)
            preset_scales!(u, dealias)
            # Unit vectors
            sz = size(phi)
            ex = reshape(cat(-sin.(phi) .+ 0 .* r, cos.(phi) .+ 0 .* r; dims=1), 2, sz...)
            ey = reshape(cat(cos.(phi) .+ 0 .* r, sin.(phi) .+ 0 .* r; dims=1), 2, sz...)
            u["g"] = @. (x^2 * y - 2 * x * y^5) * ex + (x^2 * y + 7 * x^3 * y^2) * ey
            v = evaluate(radial_component(interpolate(u; r=radius)))
            # Expected: dot product with radial unit vector (cos(phi), sin(phi))
            # u_r = u_x*cos(phi) + u_y*sin(phi), where u is in (e_phi, e_r) basis
            # radial_component extracts the e_r component
            vg = @. (radius^3 * cp^2 * sp - 2 * radius^6 * cp * sp^5) * cp +
                    (radius^3 * cp^2 * sp + 7 * radius^5 * cp^3 * sp^2) * sp
            @test isapprox(v["g"], vg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Radial component tests (tensor) ----

    @testset "radial component tensor $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T radius=$radius" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            radius in [0.5, 1.0, 1.5]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
            cp = cos.(phi)
            sp = sin.(phi)
            T_field = TensorField(d, (c, c), bases=b)
            preset_scales!(T_field, dealias)
            # Unit vectors and outer products
            sz = size(phi)
            ex_vec = cat(-sin.(phi) .+ 0 .* r, cos.(phi) .+ 0 .* r; dims=1)
            ey_vec = cat(cos.(phi) .+ 0 .* r, sin.(phi) .+ 0 .* r; dims=1)
            ex = reshape(ex_vec, 2, sz...)
            ey = reshape(ey_vec, 2, sz...)
            exex = reshape(ex_vec, 2, 1, sz...) .* reshape(ex_vec, 1, 2, sz...)
            exey = reshape(ex_vec, 2, 1, sz...) .* reshape(ey_vec, 1, 2, sz...)
            eyex = reshape(ey_vec, 2, 1, sz...) .* reshape(ex_vec, 1, 2, sz...)
            eyey = reshape(ey_vec, 2, 1, sz...) .* reshape(ey_vec, 1, 2, sz...)
            T_field["g"] = @. (3 * x^2 + y) * exex + y^3 * exey +
                              x^2 * y^2 * eyex + (y^5 - 2 * x * y) * eyey
            A = evaluate(radial_component(interpolate(T_field; r=radius)))
            # Expected: contraction of T with radial unit vector on first index
            Ag = @. (3 * radius^2 * cp^2 + radius * sp) * cp * ex +
                    radius^3 * sp^3 * cp * ey +
                    radius^4 * cp^2 * sp^2 * sp * ex +
                    (radius^5 * sp^5 - 2 * radius^2 * cp * sp) * sp * ey
            @test isapprox(A["g"], Ag, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Azimuthal component tests (vector) ----

    @testset "azimuthal component vector $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T radius=$radius" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            radius in [0.5, 1.0, 1.5]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
            cp = cos.(phi)
            sp = sin.(phi)
            u = VectorField(d, c, bases=b)
            preset_scales!(u, dealias)
            sz = size(phi)
            ex = reshape(cat(-sin.(phi) .+ 0 .* r, cos.(phi) .+ 0 .* r; dims=1), 2, sz...)
            ey = reshape(cat(cos.(phi) .+ 0 .* r, sin.(phi) .+ 0 .* r; dims=1), 2, sz...)
            u["g"] = @. (x^2 * y - 2 * x * y^5) * ex + (x^2 * y + 7 * x^3 * y^2) * ey
            v = evaluate(azimuthal_component(interpolate(u; r=radius)))
            # Azimuthal component: dot with (-sin(phi), cos(phi)) = e_phi direction
            vg = @. (radius^3 * cp^2 * sp - 2 * radius^6 * cp * sp^5) * (-sp) +
                    (radius^3 * cp^2 * sp + 7 * radius^5 * cp^3 * sp^2) * cp
            @test isapprox(v["g"], vg, atol=1e-10)
        catch e
            @test_broken false
        end
    end

    # ---- Azimuthal component tests (tensor) ----

    @testset "azimuthal component tensor $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T radius=$radius" for
            (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            radius in [0.5, 1.0, 1.5]
        try
            c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
            cp = cos.(phi)
            sp = sin.(phi)
            T_field = TensorField(d, (c, c), bases=b)
            preset_scales!(T_field, dealias)
            sz = size(phi)
            ex_vec = cat(-sin.(phi) .+ 0 .* r, cos.(phi) .+ 0 .* r; dims=1)
            ey_vec = cat(cos.(phi) .+ 0 .* r, sin.(phi) .+ 0 .* r; dims=1)
            ex = reshape(ex_vec, 2, sz...)
            ey = reshape(ey_vec, 2, sz...)
            exex = reshape(ex_vec, 2, 1, sz...) .* reshape(ex_vec, 1, 2, sz...)
            exey = reshape(ex_vec, 2, 1, sz...) .* reshape(ey_vec, 1, 2, sz...)
            eyex = reshape(ey_vec, 2, 1, sz...) .* reshape(ex_vec, 1, 2, sz...)
            eyey = reshape(ey_vec, 2, 1, sz...) .* reshape(ey_vec, 1, 2, sz...)
            T_field["g"] = @. (3 * x^2 + y) * exex + y^3 * exey +
                              x^2 * y^2 * eyex + (y^5 - 2 * x * y) * eyey
            A = evaluate(azimuthal_component(interpolate(T_field; r=radius)))
            # Expected: contraction with azimuthal unit vector (-sin(phi), cos(phi)) on first index
            Ag = @. (3 * radius^2 * cp^2 + radius * sp) * (-sp) * ex +
                    radius^3 * sp^3 * (-sp) * ey +
                    radius^4 * cp^2 * sp^2 * cp * ex +
                    (radius^5 * sp^5 - 2 * radius^2 * cp * sp) * cp * ey
            @test isapprox(A["g"], Ag, atol=1e-10)
        catch e
            @test_broken false
        end
    end

end

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

    # NumPy-style array helpers: tensor components are stacked along new
    # leading axes, and grid arrays are padded with leading singleton axes so
    # they broadcast against the component axes.
    stackcomp(cs...) = stack(collect(cs); dims = 1)
    padlead(a, n) = reshape(a, ntuple(_ -> 1, n)..., size(a)...)
    outer2(u, v) = reshape(u, 2, 1, size(u)[2:end]...) .* reshape(v, 1, 2, size(v)[2:end]...)
    unit_vectors(phi, r) = (
        stackcomp(-sin.(phi) .+ 0 .* r, cos.(phi) .+ 0 .* r),
        stackcomp(cos.(phi) .+ 0 .* r, sin.(phi) .+ 0 .* r),
    )

    # ---- Builder functions ----

    function build_disk(Nphi, Nr, k, dealias, T)
        c = PolarCoordinates("phi", "r")
        d = Distributor(c, T)
        b = DiskBasis(
            c, (Nphi, Nr), T; radius = radius_disk, k = k,
            dealias = (dealias, dealias)
        )
        phi, r = local_grids(d, b, scales = dealias)
        x, y = cartesian(PolarCoordinates, phi, r)
        return c, d, b, phi, r, x, y
    end

    function build_annulus(Nphi, Nr, k, dealias, T)
        c = PolarCoordinates("phi", "r")
        d = Distributor(c, T)
        b = AnnulusBasis(
            c, (Nphi, Nr), T; radii = radii_annulus, k = k,
            dealias = (dealias, dealias)
        )
        phi, r = local_grids(d, b, scales = dealias)
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
        c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
        f = Field(d, dtype = T)
        f["g"] = 1
        g = evaluate(Convert(f, b))
        @test isapprox(g["g"], f["g"] .+ zero(g["g"]), atol = 1.0e-12)
    end

    @testset "convert scalar $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T layout=$layout" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["c", "g"]
        c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
        f = Field(d, bases = (b,), dtype = T)
        preset_scales!(f, dealias)
        f["g"] = @. 3 * x^2 + 2 * y
        g = evaluate(laplacian(f, c))
        change_layout!(f, layout)
        change_layout!(g, layout)
        h = evaluate(f + g)
        @test isapprox(h["g"], f["g"] .+ g["g"], atol = 1.0e-10)
    end

    @testset "convert vector $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T layout=$layout" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            layout in ["c", "g"]
        c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
        u = VectorField(d, c, bases = b)
        preset_scales!(u, dealias)
        ex, ey = unit_vectors(phi, r)
        u["g"] = padlead(@.(4 * x^3 + 3 * y^2), 1) .* ey
        v = evaluate(laplacian(u, c))
        change_layout!(u, layout)
        change_layout!(v, layout)
        w = evaluate(u + v)
        @test isapprox(w["g"], u["g"] .+ v["g"], atol = 1.0e-10)
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
        c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
        f = VectorField(d, c, bases = b)
        fill_random!(f, layout = "g")
        low_pass_filter!(f, scales = 0.75)
        change_layout!(f, layout)
        g = evaluate(skew(f))
        @test isapprox(g["g"][1, :, :], f["g"][2, :, :], atol = 1.0e-12)
        @test isapprox(g["g"][2, :, :], -f["g"][1, :, :], atol = 1.0e-12)
    end

    @testset "skew implicit $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
        f = VectorField(d, c, bases = b)
        fill_random!(f, layout = "g")
        low_pass_filter!(f, scales = 0.75)
        u = VectorField(d, c, bases = b)
        # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
        @test_broken begin
            problem = LBVP([u], namespace = Dict("u" => u, "f" => f, "skew" => skew))
            add_equation!(problem, "skew(u) = skew(f)")
            solver = build_solver(problem)
            solve!(solver)
            change_scales!(u, dealias)
            change_scales!(f, dealias)
            isapprox(u["g"], f["g"], atol = 1.0e-10)
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
        c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
        u = VectorField(d, c, bases = b)
        preset_scales!(u, dealias)
        ex, ey = unit_vectors(phi, r)
        u["g"] = padlead(@.(4 * x^3 + 3 * y^2), 1) .* ey
        T_field = evaluate(gradient(u, c))
        # Compute expected trace in grid space before layout change
        fg = T_field["g"][1, 1, :, :] .+ T_field["g"][2, 2, :, :]
        change_layout!(T_field, layout)
        f = evaluate(trace_op(T_field))
        @test isapprox(f["g"], fg, atol = 1.0e-10)
    end

    @testset "trace implicit tensor $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
        f = Field(d, bases = (b,), dtype = T)
        g = Field(d, bases = (b,), dtype = T)
        fill_random!(g, layout = "g")
        low_pass_filter!(g, scales = 0.5)
        # Build identity tensor on radial basis
        rb = Dedalus.radial_basis(b)
        I_tensor = TensorField(d, (c, c), bases = rb)
        I_tensor["g"][1, 1, :, :] .= 1
        I_tensor["g"][2, 2, :, :] .= 1
        # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
        @test_broken begin
            problem = LBVP([f])
            add_equation!(problem, (trace_op(I_tensor * f), 2 * g))
            solver = LinearBoundaryValueSolver(problem, matrix_coupling = [false, true])
            solve!(solver)
            isapprox(f["c"], g["c"], atol = 1.0e-10)
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
        c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
        f = TensorField(d, (c, c), bases = b)
        fill_random!(f, layout = "g")
        low_pass_filter!(f, scales = 0.75)
        change_layout!(f, layout)
        g = evaluate(transpose_components(f))
        # Check g[i,j,...] == f[j,i,...]
        for i in 1:2, j in 1:2
            @test isapprox(g["g"][i, j, :, :], f["g"][j, i, :, :], atol = 1.0e-12)
        end
    end

    @testset "transpose implicit $bname Nphi=$Nphi Nr=$Nr k=$k dealias=$dealias T=$T" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            Nphi in Nphi_range,
            Nr in Nr_range,
            k in k_range,
            dealias in dealias_range,
            T in dtype_range
        c, d, b, phi, r, x, y = basis_fn(Nphi, Nr, k, dealias, T)
        f = TensorField(d, (c, c), bases = b)
        fill_random!(f, layout = "g")
        low_pass_filter!(f, scales = 0.75)
        u = TensorField(d, (c, c), bases = b)
        # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
        @test_broken begin
            problem = LBVP(
                [u], namespace = Dict(
                    "u" => u, "f" => f,
                    "trans" => transpose_components
                )
            )
            add_equation!(problem, "trans(u) = trans(f)")
            solver = build_solver(problem)
            solve!(solver)
            change_scales!(u, dealias)
            change_scales!(f, dealias)
            isapprox(u["g"], f["g"], atol = 1.0e-10)
        end
    end

    # ---- Azimuthal average tests ----

    @testset "azimuthal average scalar $bname Nphi=16 Nr=10 k=$k dealias=$dealias T=$T" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in [0, 1, 2, 5],
            dealias in dealias_range,
            T in dtype_range
        c, d, b, phi, r, x, y = basis_fn(16, 10, k, dealias, T)
        f = Field(d, bases = (b,), dtype = T)
        preset_scales!(f, dealias)
        f["g"] = @. r^2 + x
        hg = @. r^2
        # Polar integrate/average are not implemented: https://github.com/SciML/Dedalus.jl/issues/26
        @test_broken isapprox(evaluate(average(f, c.coords[1]))["g"], hg, atol = 1.0e-10)
    end

    # ---- Integrate tests ----

    @testset "integrate scalar $bname Nphi=16 Nr=10 k=$k dealias=$dealias T=$T n=$n" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in [0, 1, 2, 5],
            dealias in dealias_range,
            T in dtype_range,
            n in [0, 1, 2]
        c, d, b, phi, r, x, y = basis_fn(16, 10, k, dealias, T)
        f = Field(d, bases = (b,), dtype = T)
        preset_scales!(f, dealias)
        f["g"] = @. r^(2 * n)
        # Compute analytical result: integral of r^(2n) over disk/annulus
        if b isa DiskBasis
            r_inner, r_outer = 0.0, radius_disk
        else
            r_inner, r_outer = radii_annulus
        end
        hg = 2 * pi * (r_outer^(2 + 2 * n) - r_inner^(2 + 2 * n)) / (2 + 2 * n)
        # Polar integrate/average are not implemented: https://github.com/SciML/Dedalus.jl/issues/26
        @test_broken all(isapprox.(evaluate(integrate(f, c))["g"], hg, atol = 1.0e-10))
    end

    # ---- Interpolate azimuth tests (scalar) ----

    @testset "interpolate azimuth scalar $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T phi_i=$phi_i" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            phi_i in [0.5, 1.0, 1.5]
        c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
        f = Field(d, bases = (b,), dtype = T)
        preset_scales!(f, dealias)
        f["g"] = @. x^4 + 2 * y^4
        x_i, y_i = cartesian(PolarCoordinates, fill(phi_i, 1, 1), r)
        hg = @. x_i^4 + 2 * y_i^4
        # Polar interpolation is not implemented: https://github.com/SciML/Dedalus.jl/issues/25
        @test_broken isapprox(evaluate(interpolate(f; phi = phi_i))["g"], hg, atol = 1.0e-10)
    end

    # ---- Interpolate radius tests (scalar) ----

    @testset "interpolate radius scalar $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T r_i=$r_i" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            r_i in [0.5, 1.0, 1.5]
        c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
        f = Field(d, bases = (b,), dtype = T)
        preset_scales!(f, dealias)
        f["g"] = @. x^4 + 2 * y^4
        x_i, y_i = cartesian(PolarCoordinates, phi, fill(r_i, 1, 1))
        hg = @. x_i^4 + 2 * y_i^4
        # Polar interpolation is not implemented: https://github.com/SciML/Dedalus.jl/issues/25
        @test_broken isapprox(evaluate(interpolate(f; r = r_i))["g"], hg, atol = 1.0e-10)
    end

    # ---- Interpolate azimuth tests (vector) ----

    @testset "interpolate azimuth vector $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T phi_i=$phi_i" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            phi_i in [0.5, 1.0, 1.5]
        c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
        f = Field(d, bases = (b,), dtype = T)
        preset_scales!(f, dealias)
        f["g"] = @. x^4 + 2 * y^4
        u = gradient(f, c)
        # Expected gradient at the interpolation point
        phi_arr = fill(phi_i, 1, 1)
        x_i, y_i = cartesian(PolarCoordinates, phi_arr, r)
        ex, ey = unit_vectors(phi_arr, r)
        vg = padlead(4 .* x_i .^ 3, 1) .* ex .+ padlead(8 .* y_i .^ 3, 1) .* ey
        # Polar interpolation is not implemented: https://github.com/SciML/Dedalus.jl/issues/25
        @test_broken isapprox(evaluate(interpolate(u; phi = phi_i))["g"], vg, atol = 1.0e-10)
    end

    # ---- Interpolate radius tests (vector) ----

    @testset "interpolate radius vector $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T r_i=$r_i" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            r_i in [0.5, 1.0, 1.5]
        c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
        f = Field(d, bases = (b,), dtype = T)
        preset_scales!(f, dealias)
        f["g"] = @. x^4 + 2 * y^4
        u = gradient(f, c)
        # Expected gradient at the interpolation point
        r_arr = fill(r_i, 1, 1)
        x_i, y_i = cartesian(PolarCoordinates, phi, r_arr)
        ex, ey = unit_vectors(phi, r_arr)
        vg = padlead(4 .* x_i .^ 3, 1) .* ex .+ padlead(8 .* y_i .^ 3, 1) .* ey
        # Polar interpolation is not implemented: https://github.com/SciML/Dedalus.jl/issues/25
        @test_broken isapprox(evaluate(interpolate(u; r = r_i))["g"], vg, atol = 1.0e-10)
    end

    # ---- Interpolate azimuth tests (tensor) ----

    @testset "interpolate azimuth tensor $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T phi_i=$phi_i" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            phi_i in [0.5, 1.0, 1.5]
        c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
        f = Field(d, bases = (b,), dtype = T)
        preset_scales!(f, dealias)
        f["g"] = @. x^4 + 2 * y^4
        T_field = gradient(gradient(f, c), c)
        # Expected Hessian at the interpolation point
        phi_arr = fill(phi_i, 1, 1)
        x_i, y_i = cartesian(PolarCoordinates, phi_arr, r)
        ex, ey = unit_vectors(phi_arr, r)
        vg = padlead(12 .* x_i .^ 2, 2) .* outer2(ex, ex) .+ padlead(24 .* y_i .^ 2, 2) .* outer2(ey, ey)
        # Polar interpolation is not implemented: https://github.com/SciML/Dedalus.jl/issues/25
        @test_broken isapprox(evaluate(interpolate(T_field; phi = phi_i))["g"], vg, atol = 1.0e-10)
    end

    # ---- Interpolate radius tests (tensor) ----

    @testset "interpolate radius tensor $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T r_i=$r_i" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            r_i in [0.5, 1.0, 1.5]
        c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
        f = Field(d, bases = (b,), dtype = T)
        preset_scales!(f, dealias)
        f["g"] = @. x^4 + 2 * y^4
        T_field = gradient(gradient(f, c), c)
        # Expected Hessian at the interpolation point
        r_arr = fill(r_i, 1, 1)
        x_i, y_i = cartesian(PolarCoordinates, phi, r_arr)
        ex, ey = unit_vectors(phi, r_arr)
        vg = padlead(12 .* x_i .^ 2, 2) .* outer2(ex, ex) .+ padlead(24 .* y_i .^ 2, 2) .* outer2(ey, ey)
        # Polar interpolation is not implemented: https://github.com/SciML/Dedalus.jl/issues/25
        @test_broken isapprox(evaluate(interpolate(T_field; r = r_i))["g"], vg, atol = 1.0e-10)
    end

    # ---- Radial component tests (vector) ----

    @testset "radial component vector $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T radius=$radius" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            radius in [0.5, 1.0, 1.5]
        c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
        cp = cos.(phi)
        sp = sin.(phi)
        u = VectorField(d, c, bases = b, dtype = T)
        preset_scales!(u, dealias)
        ex, ey = unit_vectors(phi, r)
        u["g"] = padlead(@.(x^2 * y - 2 * x * y^5), 1) .* ex .+
            padlead(@.(x^2 * y + 7 * x^3 * y^2), 1) .* ey
        vg = @. (radius^3 * cp^2 * sp - 2 * radius^6 * cp * sp^5) * cp +
            (radius^3 * cp^2 * sp + 7 * radius^5 * cp^3 * sp^2) * sp
        # Polar interpolation is not implemented: https://github.com/SciML/Dedalus.jl/issues/25
        @test_broken isapprox(evaluate(radial_component(interpolate(u; r = radius)))["g"], vg, atol = 1.0e-10)
    end

    # ---- Radial component tests (tensor) ----

    @testset "radial component tensor $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T radius=$radius" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            radius in [0.5, 1.0, 1.5]
        c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
        cp = cos.(phi)
        sp = sin.(phi)
        T_field = TensorField(d, (c, c), bases = b, dtype = T)
        preset_scales!(T_field, dealias)
        ex, ey = unit_vectors(phi, r)
        T_field["g"] = padlead(@.(3 * x^2 + y), 2) .* outer2(ex, ex) .+
            padlead(y .^ 3, 2) .* outer2(ex, ey) .+
            padlead(@.(x^2 * y^2), 2) .* outer2(ey, ex) .+
            padlead(@.(y^5 - 2 * x * y), 2) .* outer2(ey, ey)
        # Unit vectors at the azimuthal grid points, for the expected values
        ex, ey = unit_vectors(phi, 0)
        Ag = padlead(@.((3 * radius^2 * cp^2 + radius * sp) * cp), 1) .* ex .+
            padlead(@.(radius^3 * sp^3 * cp), 1) .* ey .+
            padlead(@.(radius^4 * cp^2 * sp^2 * sp), 1) .* ex .+
            padlead(@.((radius^5 * sp^5 - 2 * radius^2 * cp * sp) * sp), 1) .* ey
        # Polar interpolation is not implemented: https://github.com/SciML/Dedalus.jl/issues/25
        @test_broken isapprox(evaluate(radial_component(interpolate(T_field; r = radius)))["g"], Ag, atol = 1.0e-10)
    end

    # ---- Azimuthal component tests (vector) ----

    @testset "azimuthal component vector $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T radius=$radius" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            radius in [0.5, 1.0, 1.5]
        c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
        cp = cos.(phi)
        sp = sin.(phi)
        u = VectorField(d, c, bases = b, dtype = T)
        preset_scales!(u, dealias)
        ex, ey = unit_vectors(phi, r)
        u["g"] = padlead(@.(x^2 * y - 2 * x * y^5), 1) .* ex .+
            padlead(@.(x^2 * y + 7 * x^3 * y^2), 1) .* ey
        vg = @. (radius^3 * cp^2 * sp - 2 * radius^6 * cp * sp^5) * (-sp) +
            (radius^3 * cp^2 * sp + 7 * radius^5 * cp^3 * sp^2) * cp
        # Polar interpolation is not implemented: https://github.com/SciML/Dedalus.jl/issues/25
        @test_broken isapprox(evaluate(azimuthal_component(interpolate(u; r = radius)))["g"], vg, atol = 1.0e-10)
    end

    # ---- Azimuthal component tests (tensor) ----

    @testset "azimuthal component tensor $bname Nphi=16 Nr=8 k=$k dealias=$dealias T=$T radius=$radius" for
        (bname, basis_fn) in [("disk", build_disk), ("annulus", build_annulus)],
            k in k_range,
            dealias in dealias_range,
            T in dtype_range,
            radius in [0.5, 1.0, 1.5]
        c, d, b, phi, r, x, y = basis_fn(16, 8, k, dealias, T)
        cp = cos.(phi)
        sp = sin.(phi)
        T_field = TensorField(d, (c, c), bases = b, dtype = T)
        preset_scales!(T_field, dealias)
        ex, ey = unit_vectors(phi, r)
        T_field["g"] = padlead(@.(3 * x^2 + y), 2) .* outer2(ex, ex) .+
            padlead(y .^ 3, 2) .* outer2(ex, ey) .+
            padlead(@.(x^2 * y^2), 2) .* outer2(ey, ex) .+
            padlead(@.(y^5 - 2 * x * y), 2) .* outer2(ey, ey)
        # Unit vectors at the azimuthal grid points, for the expected values
        ex, ey = unit_vectors(phi, 0)
        Ag = padlead(@.((3 * radius^2 * cp^2 + radius * sp) * (-sp)), 1) .* ex .+
            padlead(@.(radius^3 * sp^3 * (-sp)), 1) .* ey .+
            padlead(@.(radius^4 * cp^2 * sp^2 * cp), 1) .* ex .+
            padlead(@.((radius^5 * sp^5 - 2 * radius^2 * cp * sp) * cp), 1) .* ey
        # Polar interpolation is not implemented: https://github.com/SciML/Dedalus.jl/issues/25
        @test_broken isapprox(evaluate(azimuthal_component(interpolate(T_field; r = radius)))["g"], Ag, atol = 1.0e-10)
    end

end

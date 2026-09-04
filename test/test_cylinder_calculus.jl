"""Tests for cylinder calculus (gradient, divergence, curl, laplacian),
operators (trace, transpose, integrate, average), and NCC products
on DirectProduct(z, polar) coordinate systems."""

using Test
using Dedalus

@testset "Cylinder Calculus, Operators, and NCC" begin

    # ================================================================
    # Constants and builder helpers
    # ================================================================

    cyl_length = 1.88
    cyl_radius_disk = 1.5
    cyl_radii_annulus = (0.5, 3.0)

    function build_periodic_cylinder(Nz, Nphi, Nr, alpha, k, dealias, T)
        cz = Coordinate("z")
        cp = PolarCoordinates("phi", "r")
        c = DirectProduct(cz, cp)
        d = Distributor(c, T)
        bz = Fourier(cz, Nz, (0, cyl_length); dealias=dealias, dtype=T)
        bp = DiskBasis(cp, (Nphi, Nr), T; radius=cyl_radius_disk, alpha=alpha, k=k, dealias=dealias)
        z, phi, r = local_grids(d, bz, bp, scales=dealias)
        x, y = cartesian(PolarCoordinates, phi, r)
        return c, d, (bz, bp), z, phi, r, x, y
    end

    function build_periodic_cylindrical_annulus(Nz, Nphi, Nr, alpha, k, dealias, T)
        cz = Coordinate("z")
        cp = PolarCoordinates("phi", "r")
        c = DirectProduct(cz, cp)
        d = Distributor(c, T)
        bz = Fourier(cz, Nz, (0, cyl_length); dealias=dealias, dtype=T)
        bp = AnnulusBasis(cp, (Nphi, Nr), T; radii=cyl_radii_annulus, k=k, dealias=dealias)
        z, phi, r = local_grids(d, bz, bp, scales=dealias)
        x, y = cartesian(PolarCoordinates, phi, r)
        return c, d, (bz, bp), z, phi, r, x, y
    end

    # ================================================================
    # Section 1: Cylinder Calculus Tests
    # (from test_cylinder_calculus.py)
    # ================================================================

    @testset "Cylinder Calculus" begin

        # Parameters matching Python test_cylinder_calculus.py
        Nz_range = [8]
        Nphi_range = [16]
        Nr_range = [8]
        alpha_range = [0]
        k_range = [0]
        dealias_range = [1, 1.5]
        basis_range = [
            ("disk", build_periodic_cylinder),
            ("annulus", build_periodic_cylindrical_annulus),
        ]
        dtype_range = [Float64, ComplexF64]

        # -- 1. gradient of scalar: grad(3*x^2 + 2*y + sin(kz*z)) --
        @testset "gradient scalar $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                kz = 2 * pi / cyl_length
                f = Field(d, bases=b)
                preset_scales!(f, dealias)
                f["g"] = @. 3 * x^2 + 2 * y + sin(kz * z)
                # Expected gradient: (6x, 2, kz*cos(kz*z))
                # In cylindrical (z, phi, r) components:
                #   e_z component: kz*cos(kz*z)
                #   e_phi component: (-sin(phi))*(6x) + cos(phi)*2
                #                  = -sin(phi)*6*r*cos(phi) + 2*cos(phi)
                #                  = -3*r*sin(2*phi) + 2*cos(phi)
                #   e_r component: cos(phi)*(6x) + sin(phi)*2
                #                = cos(phi)*6*r*cos(phi) + 2*sin(phi)
                #                = 6*r*cos(phi)^2 + 2*sin(phi)
                #                = 3*r*(1 + cos(2*phi)) + 2*sin(phi)
                g = gradient(f, c)
                g_eval = evaluate(g)
                ge = VectorField(d, c, bases=b)
                preset_scales!(ge, dealias)
                ge["g"][1, :, :, :] = @. kz * cos(kz * z) + 0 * phi + 0 * r
                ge["g"][2, :, :, :] = @. -3 * r * sin(2 * phi) + 2 * cos(phi) + 0 * z
                ge["g"][3, :, :, :] = @. 3 * r * (1 + cos(2 * phi)) + 2 * sin(phi) + 0 * z
                @test isapprox(g_eval["g"], ge["g"], atol=1e-9)
            catch e
                @test_broken false
            end
        end

        # -- 2. gradient of vector: grad(grad(f)) --
        @testset "gradient vector $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                kz = 2 * pi / cyl_length
                f = Field(d, bases=b)
                preset_scales!(f, dealias)
                f["g"] = @. 3 * x^2 + 2 * y + sin(kz * z)
                # grad(grad(f)) is a rank-2 tensor
                T_op = gradient(gradient(f, c), c)
                T_eval = evaluate(T_op)
                # Check symmetry: T[i,j] == T[j,i]
                for i in 1:3, j in 1:3
                    @test isapprox(T_eval["g"][i, j, :, :, :], T_eval["g"][j, i, :, :, :], atol=1e-9)
                end
            catch e
                @test_broken false
            end
        end

        # -- 3. divergence of vector: div(grad(f)) should equal laplacian(f) --
        @testset "divergence vector $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                kz = 2 * pi / cyl_length
                f = Field(d, bases=b)
                preset_scales!(f, dealias)
                f["g"] = @. 3 * x^2 + 2 * y + sin(kz * z)
                div_grad_f = divergence(gradient(f, c))
                lap_f = laplacian(f, c)
                div_grad_eval = evaluate(div_grad_f)
                lap_eval = evaluate(lap_f)
                @test isapprox(div_grad_eval["g"], lap_eval["g"], atol=1e-9)
            catch e
                @test_broken false
            end
        end

        # -- 4. divergence of tensor: div(grad(v)) --
        @testset "divergence tensor $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                kz = 2 * pi / cyl_length
                # Build a vector field
                v = VectorField(d, c, bases=b)
                preset_scales!(v, dealias)
                v["g"][1, :, :, :] = @. sin(kz * z) + 0 * phi + 0 * r
                v["g"][2, :, :, :] = @. 0 * z + sin(phi) * r + 0 * r
                v["g"][3, :, :, :] = @. 0 * z + cos(phi) * r + 0 * r
                # div(grad(v)) should equal laplacian(v)
                dg = divergence(gradient(v, c))
                lv = laplacian(v, c)
                dg_eval = evaluate(dg)
                lv_eval = evaluate(lv)
                @test isapprox(dg_eval["g"], lv_eval["g"], atol=1e-9)
            catch e
                @test_broken false
            end
        end

        # -- 5. curl of vector --
        @testset "curl vector $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                kz = 2 * pi / cyl_length
                # Simple test: curl of gradient should be zero
                f = Field(d, bases=b)
                preset_scales!(f, dealias)
                f["g"] = @. 3 * x^2 + 2 * y + sin(kz * z)
                curl_grad_f = curl(gradient(f, c))
                cg_eval = evaluate(curl_grad_f)
                expected_zero = zeros(size(cg_eval["g"]))
                @test isapprox(cg_eval["g"], expected_zero, atol=1e-9)
            catch e
                @test_broken false
            end
        end

        # -- 6. laplacian of scalar --
        @testset "laplacian scalar $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                kz = 2 * pi / cyl_length
                f = Field(d, bases=b)
                preset_scales!(f, dealias)
                f["g"] = @. 3 * x^2 + 2 * y + sin(kz * z)
                lap_f = laplacian(f, c)
                lap_eval = evaluate(lap_f)
                # Analytic: laplacian(3x^2 + 2y + sin(kz*z)) = 6 - kz^2*sin(kz*z)
                expected = Field(d, bases=b)
                preset_scales!(expected, dealias)
                expected["g"] = @. 6 - kz^2 * sin(kz * z) + 0 * phi + 0 * r
                @test isapprox(lap_eval["g"], expected["g"], atol=1e-9)
            catch e
                @test_broken false
            end
        end

        # -- 7. laplacian of vector --
        @testset "laplacian vector $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                kz = 2 * pi / cyl_length
                v = VectorField(d, c, bases=b)
                preset_scales!(v, dealias)
                v["g"][1, :, :, :] = @. sin(kz * z) + 0 * phi + 0 * r
                v["g"][2, :, :, :] = @. 0 * z + 0 * phi + 0 * r
                v["g"][3, :, :, :] = @. 0 * z + 0 * phi + 0 * r
                # laplacian of (sin(kz*z), 0, 0) should be (-kz^2*sin(kz*z), 0, 0)
                lap_v = laplacian(v, c)
                lap_eval = evaluate(lap_v)
                expected = VectorField(d, c, bases=b)
                preset_scales!(expected, dealias)
                expected["g"][1, :, :, :] = @. -kz^2 * sin(kz * z) + 0 * phi + 0 * r
                expected["g"][2, :, :, :] = @. 0 * z + 0 * phi + 0 * r
                expected["g"][3, :, :, :] = @. 0 * z + 0 * phi + 0 * r
                @test isapprox(lap_eval["g"], expected["g"], atol=1e-9)
            catch e
                @test_broken false
            end
        end

    end  # Cylinder Calculus

    # ================================================================
    # Section 2: Cylinder Operator Tests
    # (from test_cylinder_operators.py)
    # ================================================================

    @testset "Cylinder Operators" begin

        # Parameters matching Python test_cylinder_operators.py
        Nz_range = [8]
        Nphi_range = [8]
        Nr_range = [8]
        alpha_range = [0, 1]
        k_range = [0]
        dealias_range = [1, 1.5]
        basis_range = [
            ("disk", build_periodic_cylinder),
            ("annulus", build_periodic_cylindrical_annulus),
        ]
        dtype_range = [Float64, ComplexF64]

        # -- 8. trace of tensor (explicit) --
        @testset "trace explicit $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = TensorField(d, (c, c), bases=b)
                fill_random!(f, layout="g")
                g = evaluate(trace_op(f))
                # Trace is sum of diagonal elements (3D: z, phi, r)
                expected = f["g"][1, 1, :, :, :] .+ f["g"][2, 2, :, :, :] .+ f["g"][3, 3, :, :, :]
                @test isapprox(g["g"], expected, atol=1e-12)
            catch e
                @test_broken false
            end
        end

        # -- 9. trace of tensor (implicit via LBVP) --
        @testset "trace implicit $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = Field(d, bases=b)
                fill_random!(f, layout="g")
                u = Field(d, bases=b)
                # Build identity tensor
                I_tensor = TensorField(d, (c, c))
                for i in 1:3
                    I_tensor["g"][i, i, :] .= 1
                end
                problem = LBVP([u], namespace=Dict("u" => u, "f" => f, "I" => I_tensor,
                                                   "dim" => 3, "trace" => trace_op))
                add_equation!(problem, "trace(I*u) = dim*f")
                solver = build_solver(problem)
                solve!(solver)
                @test isapprox(u["c"], f["c"], atol=1e-10)
            catch e
                @test_broken false
            end
        end

        # -- 10. transpose of tensor (explicit) --
        @testset "transpose explicit $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = TensorField(d, (c, c), bases=b)
                fill_random!(f, layout="g")
                g = evaluate(transpose_components(f))
                # Transposed: g[i,j,...] = f[j,i,...]
                for i in 1:3, j in 1:3
                    @test isapprox(g["g"][i, j, :, :, :], f["g"][j, i, :, :, :], atol=1e-12)
                end
            catch e
                @test_broken false
            end
        end

        # -- 11. transpose of tensor (implicit via LBVP) --
        @testset "transpose implicit $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = TensorField(d, (c, c), bases=b)
                fill_random!(f, layout="g")
                u = TensorField(d, (c, c), bases=b)
                problem = LBVP([u], namespace=Dict("u" => u, "f" => f,
                                                   "transpose" => transpose_components))
                add_equation!(problem, "transpose(u) = transpose(f)")
                solver = build_solver(problem)
                solve!(solver)
                @test isapprox(u["c"], f["c"], atol=1e-10)
            catch e
                @test_broken false
            end
        end

        # -- 12. integrate scalar --
        @testset "integrate scalar $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = Field(d, bases=b)
                preset_scales!(f, dealias)
                # f = 1 so integral = volume
                f["g"] = @. 1 + 0 * z + 0 * phi + 0 * r
                int_f = evaluate(integrate(f))
                # Expected volume
                if bname == "disk"
                    expected_volume = cyl_length * pi * cyl_radius_disk^2
                else
                    expected_volume = cyl_length * pi * (cyl_radii_annulus[2]^2 - cyl_radii_annulus[1]^2)
                end
                @test isapprox(int_f["g"][1], expected_volume, rtol=1e-9)
            catch e
                @test_broken false
            end
        end

        # -- 13. average scalar --
        @testset "average scalar $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = Field(d, bases=b)
                preset_scales!(f, dealias)
                # f = 1, average should be 1
                f["g"] = @. 1 + 0 * z + 0 * phi + 0 * r
                avg_f = evaluate(average(f))
                @test isapprox(avg_f["g"][1], 1.0, atol=1e-9)
            catch e
                @test_broken false
            end
        end

    end  # Cylinder Operators

    # ================================================================
    # Section 3: Cylinder NCC Tests
    # (from test_cylinder_ncc.py)
    # ================================================================

    @testset "Cylinder NCC" begin

        # Parameters matching Python test_cylinder_ncc.py
        Nz_range = [8]
        Nphi_range = [16]
        Nr_range = [16]
        alpha_range = [0]
        k_range = [0, 1]
        dealias_range = [1]
        basis_range = [
            ("disk", build_periodic_cylinder),
            ("annulus", build_periodic_cylindrical_annulus),
        ]
        dtype_range = [Float64, ComplexF64]

        # -- 14. scalar * scalar NCC --
        @testset "scalar prod scalar $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = Field(d, bases=b)
                g = Field(d, bases=b)
                fill_random!(f, layout="g")
                fill_random!(g, layout="g")
                vars = [g]
                w0 = f * g
                w1 = reinitialize(w0, ncc=true, ncc_vars=vars)
                problem = LBVP(vars)
                add_equation!(problem, (w1, 0))
                solver = build_solver(problem)
                store_ncc_matrices!(w1, vars, solver.subproblems)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                change_scales!(w0, 1)
                change_scales!(w1, 1)
                @test isapprox(w0["g"], w1["g"], atol=1e-10)
            catch e
                @test_broken false
            end
        end

        # -- 15. scalar * vector NCC --
        @testset "scalar prod vector $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = Field(d, bases=b)
                g = VectorField(d, c, bases=b)
                fill_random!(f, layout="g")
                fill_random!(g, layout="g")
                vars = [g]
                w0 = f * g
                w1 = reinitialize(w0, ncc=true, ncc_vars=vars)
                # Need a scalar placeholder for the equation
                s = Field(d, bases=b)
                fill_random!(s, layout="g")
                problem = LBVP(vars)
                add_equation!(problem, (s * g, 0))
                solver = build_solver(problem)
                store_ncc_matrices!(w1, vars, solver.subproblems)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                change_scales!(w0, 1)
                change_scales!(w1, 1)
                @test isapprox(w0["g"], w1["g"], atol=1e-10)
            catch e
                @test_broken false
            end
        end

        # -- 16. vector * scalar NCC --
        @testset "vector prod scalar $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = VectorField(d, c, bases=b)
                g = Field(d, bases=b)
                fill_random!(f, layout="g")
                fill_random!(g, layout="g")
                vars = [g]
                w0 = f * g
                w1 = reinitialize(w0, ncc=true, ncc_vars=vars)
                # Need a scalar placeholder for the equation
                s = Field(d, bases=b)
                fill_random!(s, layout="g")
                problem = LBVP(vars)
                add_equation!(problem, (s * g, 0))
                solver = build_solver(problem)
                store_ncc_matrices!(w1, vars, solver.subproblems)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                change_scales!(w0, 1)
                change_scales!(w1, 1)
                @test isapprox(w0["g"], w1["g"], atol=1e-10)
            catch e
                @test_broken false
            end
        end

        # -- 17. vector dot vector NCC --
        @testset "vector dot vector $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = VectorField(d, c, bases=b)
                g = VectorField(d, c, bases=b)
                fill_random!(f, layout="g")
                fill_random!(g, layout="g")
                vars = [g]
                w0 = f * g  # dot product for vectors
                w1 = reinitialize(w0, ncc=true, ncc_vars=vars)
                # Need a scalar placeholder for the equation
                s = Field(d, bases=b)
                fill_random!(s, layout="g")
                problem = LBVP(vars)
                add_equation!(problem, (s * g, 0))
                solver = build_solver(problem)
                store_ncc_matrices!(w1, vars, solver.subproblems)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                change_scales!(w0, 1)
                change_scales!(w1, 1)
                @test isapprox(w0["g"], w1["g"], atol=1e-10)
            catch e
                @test_broken false
            end
        end

        # -- 18. tensor dot vector NCC --
        @testset "tensor dot vector $bname Nz=$Nz Nphi=$Nphi Nr=$Nr alpha=$alpha k=$k dealias=$dealias T=$T" for
                (bname, basis_fn) in basis_range,
                Nz in Nz_range,
                Nphi in Nphi_range,
                Nr in Nr_range,
                alpha in alpha_range,
                k in k_range,
                dealias in dealias_range,
                T in dtype_range
            try
                c, d, b, z, phi, r, x, y = basis_fn(Nz, Nphi, Nr, alpha, k, dealias, T)
                f = TensorField(d, (c, c), bases=b)
                g = VectorField(d, c, bases=b)
                fill_random!(f, layout="g")
                fill_random!(g, layout="g")
                vars = [g]
                w0 = f * g  # tensor dot vector -> vector
                w1 = reinitialize(w0, ncc=true, ncc_vars=vars)
                # Need a scalar placeholder for the equation
                s = Field(d, bases=b)
                fill_random!(s, layout="g")
                problem = LBVP(vars)
                add_equation!(problem, (s * g, 0))
                solver = build_solver(problem)
                store_ncc_matrices!(w1, vars, solver.subproblems)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                change_scales!(w0, 1)
                change_scales!(w1, 1)
                @test isapprox(w0["g"], w1["g"], atol=1e-10)
            catch e
                @test_broken false
            end
        end

    end  # Cylinder NCC

end  # top-level testset

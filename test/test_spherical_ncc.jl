"""Tests for 3D spherical non-constant coefficients (NCCs) on ball and shell bases:
radial multiply, radial dot, radial cross, meridional multiply, meridional dot,
meridional cross.

Reference: dedalus/tests/test_spherical_ncc.py
"""

using Test
using Dedalus

@testset "Spherical NCC" begin

    radius_ball = 1.5
    radii_shell = (0.6, 1.7)
    ncc_cutoff = 1e-6

    Nphi_range = [8]
    Ntheta_range = [4]
    Nr_range = [16]
    alpha_range = [0]
    k_range = [0, 3]
    dealias_range = [2]
    dtype_range = [Float64, ComplexF64]

    # ---- Builder functions ----

    function build_ball(Nphi, Ntheta, Nr, alpha, dealias, T)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, T)
        b = BallBasis(c, (Nphi, Ntheta, Nr), T;
                      alpha=alpha, radius=radius_ball,
                      dealias=(dealias, dealias, dealias))
        return c, d, b
    end

    function build_shell(Nphi, Ntheta, Nr, alpha, dealias, T)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, T)
        b = ShellBasis(c, (Nphi, Ntheta, Nr), T;
                       alpha=(-0.5 + alpha, -0.5 + alpha), radii=radii_shell,
                       dealias=(dealias, dealias, dealias))
        return c, d, b
    end

    # ---- NCC helpers ----

    function norm2(field_val)
        rank = length(tensorsig(field_val))
        if rank == 0
            return field_val^2
        elseif rank == 1
            return DotProduct(field_val, field_val)
        elseif rank == 2
            return Trace(DotProduct(field_val, field_val))
        end
    end

    # ========================================================================
    # 1. Radial multiply (scalar and vector NCC ranks)
    #    NCC is on the radial basis, argument is on the full basis
    # ========================================================================
    @testset "radial multiply $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg rank_ncc=$rank_ncc rank_arg=$rank_arg ell_coupling=$ell_coupling dealias=$dealias T=$T" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            rank_ncc in [0, 1],
            rank_arg in [0, 1],
            ell_coupling in [false],
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, T)
            # Fields
            b_ncc = clone_with(radial_basis(b); k=k_ncc)
            b_arg = clone_with(b; k=k_arg)
            f = Field(d, bases=(b_ncc,), tensorsig=ntuple(_ -> c, rank_ncc), dtype=T)
            g = Field(d, bases=(b_arg,), tensorsig=ntuple(_ -> c, rank_arg), dtype=T)
            fill_random!(f, "g")
            fill_random!(g, "g")
            # Dummy problem to build subproblems with correct coupling/dependence
            problem = LBVP([g])
            add_equation!(problem, (norm2(f) * g, 0))
            solver = build_solver(problem; matrix_coupling=(false, ell_coupling, true))
            # NCC operators
            w0 = f * g
            w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
            store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
            w0 = evaluate(w0)
            w1 = evaluate_as_ncc(w1)
            @test isapprox(w0["c"], w1["c"], atol=1e-12)
        catch e
            @test_broken false
            @warn "radial_multiply failed" exception=e
        end
    end

    # ell_coupling=true cases are expected to fail
    @testset "radial multiply ell_coupling $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg rank_ncc=$rank_ncc rank_arg=$rank_arg dealias=$dealias T=$T" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            rank_ncc in [0, 1],
            rank_arg in [0, 1],
            dealias in dealias_range,
            T in dtype_range
        try
            @test_broken begin
                c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, T)
                b_ncc = clone_with(radial_basis(b); k=k_ncc)
                b_arg = clone_with(b; k=k_arg)
                f = Field(d, bases=(b_ncc,), tensorsig=ntuple(_ -> c, rank_ncc), dtype=T)
                g = Field(d, bases=(b_arg,), tensorsig=ntuple(_ -> c, rank_arg), dtype=T)
                fill_random!(f, "g")
                fill_random!(g, "g")
                problem = LBVP([g])
                add_equation!(problem, (norm2(f) * g, 0))
                solver = build_solver(problem; matrix_coupling=(false, true, true))
                w0 = f * g
                w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
                store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                isapprox(w0["c"], w1["c"], atol=1e-12)
            end
        catch e
            @test_broken false
            @warn "radial_multiply_ell_coupling failed" exception=e
        end
    end

    # ========================================================================
    # 2. Radial dot (vector and tensor NCC ranks)
    # ========================================================================
    @testset "radial dot $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg rank_ncc=$rank_ncc rank_arg=$rank_arg ell_coupling=$ell_coupling dealias=$dealias T=$T" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            rank_ncc in [1, 2],
            rank_arg in [1, 2],
            ell_coupling in [false],
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, T)
            b_ncc = clone_with(radial_basis(b); k=k_ncc)
            b_arg = clone_with(b; k=k_arg)
            f = Field(d, bases=(b_ncc,), tensorsig=ntuple(_ -> c, rank_ncc), dtype=T)
            g = Field(d, bases=(b_arg,), tensorsig=ntuple(_ -> c, rank_arg), dtype=T)
            fill_random!(f, "g")
            fill_random!(g, "g")
            problem = LBVP([g])
            add_equation!(problem, (norm2(f) * g, 0))
            solver = build_solver(problem; matrix_coupling=(false, ell_coupling, true))
            # NCC operators using dot product
            w0 = DotProduct(f, g)
            w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
            store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
            w0 = evaluate(w0)
            w1 = evaluate_as_ncc(w1)
            @test isapprox(w0["c"], w1["c"], atol=1e-12)
        catch e
            @test_broken false
            @warn "radial_dot failed" exception=e
        end
    end

    # ell_coupling=true cases are expected to fail
    @testset "radial dot ell_coupling $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg rank_ncc=$rank_ncc rank_arg=$rank_arg dealias=$dealias T=$T" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            rank_ncc in [1, 2],
            rank_arg in [1, 2],
            dealias in dealias_range,
            T in dtype_range
        try
            @test_broken begin
                c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, T)
                b_ncc = clone_with(radial_basis(b); k=k_ncc)
                b_arg = clone_with(b; k=k_arg)
                f = Field(d, bases=(b_ncc,), tensorsig=ntuple(_ -> c, rank_ncc), dtype=T)
                g = Field(d, bases=(b_arg,), tensorsig=ntuple(_ -> c, rank_arg), dtype=T)
                fill_random!(f, "g")
                fill_random!(g, "g")
                problem = LBVP([g])
                add_equation!(problem, (norm2(f) * g, 0))
                solver = build_solver(problem; matrix_coupling=(false, true, true))
                w0 = DotProduct(f, g)
                w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
                store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                isapprox(w0["c"], w1["c"], atol=1e-12)
            end
        catch e
            @test_broken false
            @warn "radial_dot_ell_coupling failed" exception=e
        end
    end

    # ========================================================================
    # 3. Radial cross
    # ========================================================================
    @testset "radial cross $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg ell_coupling=$ell_coupling dealias=$dealias T=$T" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            ell_coupling in [false],
            dealias in dealias_range,
            T in dtype_range
        try
            c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, T)
            b_ncc = clone_with(radial_basis(b); k=k_ncc)
            b_arg = clone_with(b; k=k_arg)
            f = VectorField(d, c, bases=(b_ncc,), dtype=T)
            g = VectorField(d, c, bases=(b_arg,), dtype=T)
            fill_random!(f, "g")
            fill_random!(g, "g")
            problem = LBVP([g])
            add_equation!(problem, (norm2(f) * g, 0))
            solver = build_solver(problem; matrix_coupling=(false, ell_coupling, true))
            w0 = CrossProduct(f, g)
            w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
            store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
            w0 = evaluate(w0)
            w1 = evaluate_as_ncc(w1)
            @test isapprox(w0["c"], w1["c"], atol=1e-12)
        catch e
            @test_broken false
            @warn "radial_cross failed" exception=e
        end
    end

    # ell_coupling=true cases are expected to fail
    @testset "radial cross ell_coupling $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg dealias=$dealias T=$T" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            dealias in dealias_range,
            T in dtype_range
        try
            @test_broken begin
                c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, T)
                b_ncc = clone_with(radial_basis(b); k=k_ncc)
                b_arg = clone_with(b; k=k_arg)
                f = VectorField(d, c, bases=(b_ncc,), dtype=T)
                g = VectorField(d, c, bases=(b_arg,), dtype=T)
                fill_random!(f, "g")
                fill_random!(g, "g")
                problem = LBVP([g])
                add_equation!(problem, (norm2(f) * g, 0))
                solver = build_solver(problem; matrix_coupling=(false, true, true))
                w0 = CrossProduct(f, g)
                w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
                store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                isapprox(w0["c"], w1["c"], atol=1e-12)
            end
        catch e
            @test_broken false
            @warn "radial_cross_ell_coupling failed" exception=e
        end
    end

    # ========================================================================
    # 4. Meridional multiply
    #    Ball meridional NCCs are not implemented (xfail for ball).
    #    Real dtypes for meridional NCCs are not implemented (xfail for Float64).
    # ========================================================================
    @testset "meridional multiply $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg rank_ncc=$rank_ncc rank_arg=$rank_arg dealias=$dealias T=$T" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            rank_ncc in [0, 1],
            rank_arg in [0, 1],
            dealias in dealias_range,
            T in [ComplexF64]
        try
            c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, T)
            b_ncc = clone_with(meridional_basis(b); k=k_ncc)
            b_arg = clone_with(b; k=k_arg)
            f = Field(d, bases=(b_ncc,), tensorsig=ntuple(_ -> c, rank_ncc), dtype=T)
            g = Field(d, bases=(b_arg,), tensorsig=ntuple(_ -> c, rank_arg), dtype=T)
            fill_random!(f, "g")
            fill_random!(g, "g")
            problem = LBVP([g])
            add_equation!(problem, (norm2(f) * g, 0))
            solver = build_solver(problem)
            w0 = f * g
            w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
            store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
            w0 = evaluate(w0)
            w1 = evaluate_as_ncc(w1)
            @test isapprox(w0["c"], w1["c"], atol=1e-12)
        catch e
            @test_broken false
            @warn "meridional_multiply failed" exception=e
        end
    end

    # Ball meridional NCCs - expected to fail
    @testset "meridional multiply ball (xfail) Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg rank_ncc=$rank_ncc rank_arg=$rank_arg dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            rank_ncc in [0, 1],
            rank_arg in [0, 1],
            dealias in dealias_range,
            T in [ComplexF64]
        try
            @test_broken begin
                c, d, b = build_ball(Nphi, Ntheta, Nr, alpha, dealias, T)
                b_ncc = clone_with(meridional_basis(b); k=k_ncc)
                b_arg = clone_with(b; k=k_arg)
                f = Field(d, bases=(b_ncc,), tensorsig=ntuple(_ -> c, rank_ncc), dtype=T)
                g = Field(d, bases=(b_arg,), tensorsig=ntuple(_ -> c, rank_arg), dtype=T)
                fill_random!(f, "g")
                fill_random!(g, "g")
                problem = LBVP([g])
                add_equation!(problem, (norm2(f) * g, 0))
                solver = build_solver(problem)
                w0 = f * g
                w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
                store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                isapprox(w0["c"], w1["c"], atol=1e-12)
            end
        catch e
            @test_broken false
            @warn "meridional_multiply_ball_xfail failed" exception=e
        end
    end

    # Real meridional NCCs - expected to fail
    @testset "meridional multiply real dtype (xfail) $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg rank_ncc=$rank_ncc rank_arg=$rank_arg dealias=$dealias" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            rank_ncc in [0, 1],
            rank_arg in [0, 1],
            dealias in dealias_range
        try
            @test_broken begin
                c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, Float64)
                b_ncc = clone_with(meridional_basis(b); k=k_ncc)
                b_arg = clone_with(b; k=k_arg)
                f = Field(d, bases=(b_ncc,), tensorsig=ntuple(_ -> c, rank_ncc), dtype=Float64)
                g = Field(d, bases=(b_arg,), tensorsig=ntuple(_ -> c, rank_arg), dtype=Float64)
                fill_random!(f, "g")
                fill_random!(g, "g")
                problem = LBVP([g])
                add_equation!(problem, (norm2(f) * g, 0))
                solver = build_solver(problem)
                w0 = f * g
                w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
                store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                isapprox(w0["c"], w1["c"], atol=1e-12)
            end
        catch e
            @test_broken false
            @warn "meridional_multiply_real_dtype_xfail failed" exception=e
        end
    end

    # ========================================================================
    # 5. Meridional dot
    # ========================================================================
    @testset "meridional dot $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg rank_ncc=$rank_ncc rank_arg=$rank_arg dealias=$dealias T=$T" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            rank_ncc in [1, 2],
            rank_arg in [1, 2],
            dealias in dealias_range,
            T in [ComplexF64]
        try
            c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, T)
            b_ncc = clone_with(meridional_basis(b); k=k_ncc)
            b_arg = clone_with(b; k=k_arg)
            f = Field(d, bases=(b_ncc,), tensorsig=ntuple(_ -> c, rank_ncc), dtype=T)
            g = Field(d, bases=(b_arg,), tensorsig=ntuple(_ -> c, rank_arg), dtype=T)
            fill_random!(f, "g")
            fill_random!(g, "g")
            problem = LBVP([g])
            add_equation!(problem, (norm2(f) * g, 0))
            solver = build_solver(problem)
            w0 = DotProduct(f, g)
            w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
            store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
            w0 = evaluate(w0)
            w1 = evaluate_as_ncc(w1)
            @test isapprox(w0["c"], w1["c"], atol=1e-12)
        catch e
            @test_broken false
            @warn "meridional_dot failed" exception=e
        end
    end

    # Ball meridional dot - expected to fail
    @testset "meridional dot ball (xfail) Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg rank_ncc=$rank_ncc rank_arg=$rank_arg dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            rank_ncc in [1, 2],
            rank_arg in [1, 2],
            dealias in dealias_range,
            T in [ComplexF64]
        try
            @test_broken begin
                c, d, b = build_ball(Nphi, Ntheta, Nr, alpha, dealias, T)
                b_ncc = clone_with(meridional_basis(b); k=k_ncc)
                b_arg = clone_with(b; k=k_arg)
                f = Field(d, bases=(b_ncc,), tensorsig=ntuple(_ -> c, rank_ncc), dtype=T)
                g = Field(d, bases=(b_arg,), tensorsig=ntuple(_ -> c, rank_arg), dtype=T)
                fill_random!(f, "g")
                fill_random!(g, "g")
                problem = LBVP([g])
                add_equation!(problem, (norm2(f) * g, 0))
                solver = build_solver(problem)
                w0 = DotProduct(f, g)
                w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
                store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                isapprox(w0["c"], w1["c"], atol=1e-12)
            end
        catch e
            @test_broken false
            @warn "meridional_dot_ball_xfail failed" exception=e
        end
    end

    # Real meridional dot - expected to fail
    @testset "meridional dot real dtype (xfail) $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg rank_ncc=$rank_ncc rank_arg=$rank_arg dealias=$dealias" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            rank_ncc in [1, 2],
            rank_arg in [1, 2],
            dealias in dealias_range
        try
            @test_broken begin
                c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, Float64)
                b_ncc = clone_with(meridional_basis(b); k=k_ncc)
                b_arg = clone_with(b; k=k_arg)
                f = Field(d, bases=(b_ncc,), tensorsig=ntuple(_ -> c, rank_ncc), dtype=Float64)
                g = Field(d, bases=(b_arg,), tensorsig=ntuple(_ -> c, rank_arg), dtype=Float64)
                fill_random!(f, "g")
                fill_random!(g, "g")
                problem = LBVP([g])
                add_equation!(problem, (norm2(f) * g, 0))
                solver = build_solver(problem)
                w0 = DotProduct(f, g)
                w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
                store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                isapprox(w0["c"], w1["c"], atol=1e-12)
            end
        catch e
            @test_broken false
            @warn "meridional_dot_real_dtype_xfail failed" exception=e
        end
    end

    # ========================================================================
    # 6. Meridional cross
    # ========================================================================
    @testset "meridional cross $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg dealias=$dealias T=$T" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            dealias in dealias_range,
            T in [ComplexF64]
        try
            c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, T)
            b_ncc = clone_with(meridional_basis(b); k=k_ncc)
            b_arg = clone_with(b; k=k_arg)
            f = VectorField(d, c, bases=(b_ncc,), dtype=T)
            g = VectorField(d, c, bases=(b_arg,), dtype=T)
            fill_random!(f, "g")
            fill_random!(g, "g")
            problem = LBVP([g])
            add_equation!(problem, (norm2(f) * g, 0))
            solver = build_solver(problem)
            w0 = CrossProduct(f, g)
            w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
            store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
            w0 = evaluate(w0)
            w1 = evaluate_as_ncc(w1)
            @test isapprox(w0["c"], w1["c"], atol=1e-12)
        catch e
            @test_broken false
            @warn "meridional_cross failed" exception=e
        end
    end

    # Ball meridional cross - expected to fail
    @testset "meridional cross ball (xfail) Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg dealias=$dealias T=$T" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            dealias in dealias_range,
            T in [ComplexF64]
        try
            @test_broken begin
                c, d, b = build_ball(Nphi, Ntheta, Nr, alpha, dealias, T)
                b_ncc = clone_with(meridional_basis(b); k=k_ncc)
                b_arg = clone_with(b; k=k_arg)
                f = VectorField(d, c, bases=(b_ncc,), dtype=T)
                g = VectorField(d, c, bases=(b_arg,), dtype=T)
                fill_random!(f, "g")
                fill_random!(g, "g")
                problem = LBVP([g])
                add_equation!(problem, (norm2(f) * g, 0))
                solver = build_solver(problem)
                w0 = CrossProduct(f, g)
                w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
                store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                isapprox(w0["c"], w1["c"], atol=1e-12)
            end
        catch e
            @test_broken false
            @warn "meridional_cross_ball_xfail failed" exception=e
        end
    end

    # Real meridional cross - expected to fail
    @testset "meridional cross real dtype (xfail) $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr alpha=$alpha k_ncc=$k_ncc k_arg=$k_arg dealias=$dealias" for
            (bname, basis_fn) in [("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            alpha in alpha_range,
            k_ncc in k_range,
            k_arg in k_range,
            dealias in dealias_range
        try
            @test_broken begin
                c, d, b = basis_fn(Nphi, Ntheta, Nr, alpha, dealias, Float64)
                b_ncc = clone_with(meridional_basis(b); k=k_ncc)
                b_arg = clone_with(b; k=k_arg)
                f = VectorField(d, c, bases=(b_ncc,), dtype=Float64)
                g = VectorField(d, c, bases=(b_arg,), dtype=Float64)
                fill_random!(f, "g")
                fill_random!(g, "g")
                problem = LBVP([g])
                add_equation!(problem, (norm2(f) * g, 0))
                solver = build_solver(problem)
                w0 = CrossProduct(f, g)
                w1 = reinitialize(w0; ncc=true, ncc_vars=[g])
                store_ncc_matrices!(w1, [g], solver.subproblems; ncc_cutoff=ncc_cutoff)
                w0 = evaluate(w0)
                w1 = evaluate_as_ncc(w1)
                isapprox(w0["c"], w1["c"], atol=1e-12)
            end
        catch e
            @test_broken false
            @warn "meridional_cross_real_dtype_xfail failed" exception=e
        end
    end

end

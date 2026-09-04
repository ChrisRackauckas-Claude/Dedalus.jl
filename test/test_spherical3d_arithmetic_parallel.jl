"""
Tests for spherical 3D arithmetic with parallel distribution (mesh=(2,2)).

Translated from dedalus/tests_parallel/test_spherical3d_arithmetic_parallel.py.
Tests ghost broadcasting in colatitude and radius for Ball and Shell bases.
Requires MPI with at least 4 processes.
"""

using Test
using MPI
using Dedalus

@testset "Spherical 3D Arithmetic Parallel" begin

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Skip if fewer than 4 processes
    if nprocs < 4
        @warn "test_spherical3d_arithmetic_parallel requires at least 4 MPI processes; skipping."
        @test_broken false
        return
    end

    Nphi_range = [8]
    Ntheta_range = [10]
    Nr_range = [8]
    dealias_range = [1, 1.5]

    radius_ball = 1.5
    radii_shell = (1, 2)

    # ---- Builder functions (with mesh) ----

    function build_ball(Nphi, Ntheta, Nr, T, dealias; mesh=nothing)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, T; mesh=mesh)
        b = BallBasis(c, (Nphi, Ntheta, Nr), T;
                      radius=radius_ball,
                      dealias=(dealias, dealias, dealias))
        phi, theta, r = local_grids(d, b)
        x, y, z = cartesian(SphericalCoordinates, phi, theta, r)
        return c, d, b, phi, theta, r, x, y, z
    end

    function build_shell(Nphi, Ntheta, Nr, T, dealias; mesh=nothing)
        c = SphericalCoordinates("phi", "theta", "r")
        d = Distributor(c, T; mesh=mesh)
        b = ShellBasis(c, (Nphi, Ntheta, Nr), T;
                       radii=radii_shell,
                       dealias=(dealias, dealias, dealias))
        phi, theta, r = local_grids(d, b)
        x, y, z = cartesian(SphericalCoordinates, phi, theta, r)
        return c, d, b, phi, theta, r, x, y, z
    end

    # ========================================================================
    # test_sphere_constant_S2_multiplication
    # Tests ghost broadcasting in colatitude.
    # ========================================================================
    @testset "sphere constant S2 multiplication Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr T=$T dealias=$dealias" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            T in [Float64, ComplexF64],
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, T, dealias; mesh=(2, 2))
        b_S2 = S2_basis(b)
        f = Field(d, dtype=T)
        g = Field(d, bases=(b_S2,), dtype=T)
        h = Field(d, bases=(b_S2,), dtype=T)
        f["g"] = 6
        g["g"] = @. (5 * cos(theta)^2 - 1) * sin(theta) * cos(phi)
        h["g"] = @. 6 * (5 * cos(theta)^2 - 1) * sin(theta) * cos(phi)
        h_op = evaluate(f * g)
        change_scales!(h_op, 1)
        @test isapprox(h_op["g"], h["g"], atol=1e-12)
    end

    # ========================================================================
    # test_sphere_constant_radial_multiplication
    # Tests ghost broadcasting in radius for both Ball and Shell.
    # ========================================================================
    @testset "sphere constant radial multiplication $bname Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr T=$T dealias=$dealias" for
            (bname, build_fn) in [("ball", build_ball), ("shell", build_shell)],
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            T in [Float64, ComplexF64],
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = build_fn(Nphi, Ntheta, Nr, T, dealias; mesh=(2, 2))
        b_r = radial_basis(b)
        f = Field(d, dtype=T)
        g = Field(d, bases=(b_r,), dtype=T)
        h = Field(d, bases=(b_r,), dtype=T)
        f["g"] = 6
        g["g"] = @. r^2 - 0.5 * r^4
        h["g"] = @. 6 * (r^2 - 0.5 * r^4)
        h_op = evaluate(f * g)
        change_scales!(h_op, 1)
        @test isapprox(h_op["g"], h["g"], atol=1e-12)
    end

    # ========================================================================
    # test_shell_S2_radial_multiplication
    # Tests ghost broadcasting in colatitude and radius for Shell.
    # ========================================================================
    @testset "shell S2 radial multiplication Nphi=$Nphi Ntheta=$Ntheta Nr=$Nr T=$T dealias=$dealias" for
            Nphi in Nphi_range,
            Ntheta in Ntheta_range,
            Nr in Nr_range,
            T in [Float64, ComplexF64],
            dealias in dealias_range
        c, d, b, phi, theta, r, x, y, z = build_shell(Nphi, Ntheta, Nr, T, dealias; mesh=(2, 2))
        b_S2 = S2_basis(b)
        b_r = radial_basis(b)
        f = Field(d, bases=(b_S2,), dtype=T)
        g = Field(d, bases=(b_r,), dtype=T)
        h = Field(d, bases=(b,), dtype=T)
        f["g"] = @. (5 * cos(theta)^2 - 1) * sin(theta) * cos(phi)
        g["g"] = @. r^2 - 0.5 * r^3
        h["g"] = @. (r^2 - 0.5 * r^3) * (5 * cos(theta)^2 - 1) * sin(theta) * cos(phi)
        h_op = evaluate(f * g)
        change_scales!(h_op, 1)
        @test isapprox(h_op["g"], h["g"], atol=1e-12)
    end

end

"""Tests for CFL timestep computation -- Cartesian only."""

using Test
using Dedalus

@testset "CFL" begin

    @testset "full CFL Fourier-Chebyshev Nx=$Nx Nz=$Nz dealias=$dealias T=$T safety=$safety z_vel=$z_velocity_mag" for
            Nx in [32],
            Nz in [16],
            dealias in [1, 1.5],
            T in [Float64, ComplexF64],
            safety in [0.2, 0.4],
            z_velocity_mag in [0, 2]
        Lx = 2.0
        Lz = 1.0
        c = CartesianCoordinates("x", "z")
        d = Distributor(c, dtype=T)
        xb = Fourier(c.coords[1], size=Nx, bounds=(0, Lx), dealias=dealias, dtype=T)
        zb = Chebyshev(c.coords[2], size=Nz, bounds=(0, Lz), dealias=dealias)
        x, z = local_grids(d, xb, zb)
        # IVP
        u = VectorField(d, c, bases=(xb, zb))
        problem = IVP([u], namespace=Dict("u" => u, "dt" => dt))
        add_equation!(problem, "dt(u) = 0")
        solver = build_solver(problem, SBDF1)
        # CFL
        cfl = CFL(solver, initial_dt=1, safety=safety, cadence=1)
        add_velocity!(cfl, u)
        # Test
        fill_random!(u, layout="g")
        for i in 1:2
            step!(solver, 1)
        end
        dt_cfl = compute_timestep(cfl)
        cfl_op = AdvectiveCFL(u, c)
        spacing = cfl_spacing(cfl_op)
        cfl_freq = abs.(u["g"][1, :, :]) ./ spacing[1]
        cfl_freq .+= abs.(u["g"][2, :, :]) ./ spacing[2]
        cfl_freq_max = maximum(cfl_freq)
        dt_target = safety / cfl_freq_max
        @test isapprox(dt_cfl, dt_target, atol=1e-12)
    end

    @testset "CFL Fourier N=$N L=$L dealias=$dealias T=$T" for
            N in [32],
            L in [1.44],
            dealias in [1, 1.5],
            T in [Float64, ComplexF64]
        c = CartesianCoordinates("x")
        d = Distributor(c, dtype=T)
        b = Fourier(c.coords[1], size=N, bounds=(0, L), dealias=dealias, dtype=T)
        x = local_grid(d, b, scale=1)
        u = VectorField(d, c, bases=b)
        fill_random!(u, layout="g")
        cfl = AdvectiveCFL(u, c)
        cfl_freq = evaluate(cfl)["g"]
        target_freq = abs.(u["g"]) ./ cfl_spacing(cfl)[1]
        @test isapprox(cfl_freq, target_freq, atol=1e-12)
    end

    @testset "CFL Chebyshev N=$N L=$L dealias=$dealias T=$T" for
            N in [32],
            L in [1.44],
            dealias in [1, 1.5],
            T in [Float64, ComplexF64]
        c = CartesianCoordinates("x")
        d = Distributor(c, dtype=T)
        b = Chebyshev(c.coords[1], size=N, bounds=(0, L), dealias=dealias)
        x = local_grid(d, b, scale=1)
        u = VectorField(d, c, bases=b)
        fill_random!(u, layout="g")
        cfl = AdvectiveCFL(u, c)
        cfl_freq = evaluate(cfl)["g"]
        target_freq = abs.(u["g"]) ./ cfl_spacing(cfl)[1]
        @test isapprox(cfl_freq, target_freq, atol=1e-12)
    end

    @testset "CFL Fourier-Chebyshev Nx=$Nx Nz=$Nz dealias=$dealias T=$T z_vel=$z_velocity_mag" for
            Nx in [32],
            Nz in [16],
            dealias in [1, 1.5],
            T in [Float64, ComplexF64],
            z_velocity_mag in [0, 2]
        c = CartesianCoordinates("x", "z")
        d = Distributor(c, dtype=T)
        xb = Fourier(c.coords[1], size=Nx, bounds=(0, 2), dealias=dealias, dtype=T)
        zb = Chebyshev(c.coords[2], size=Nz, bounds=(0, 1), dealias=dealias)
        x, z = local_grids(d, xb, zb)
        u = VectorField(d, c, bases=(xb, zb))
        fill_random!(u, layout="g")
        cfl = AdvectiveCFL(u, c)
        cfl_freq = evaluate(cfl)["g"]
        spacing = cfl_spacing(cfl)
        target_freq = abs.(u["g"][1, :, :]) ./ spacing[1]
        target_freq .+= abs.(u["g"][2, :, :]) ./ spacing[2]
        @test isapprox(cfl_freq, target_freq, atol=1e-12)
    end

end

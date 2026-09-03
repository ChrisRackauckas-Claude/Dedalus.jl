"""Tests for initial value problems (IVPs) -- Cartesian only."""

using Test
using Dedalus

@testset "IVP" begin

    # List of timestepper names available in Dedalus
    timestepper_names = ["RK111", "RK222", "RK443", "SBDF1", "SBDF2", "SBDF3", "SBDF4",
                         "CNAB1", "CNAB2", "CNLF2", "MCNAB2"]

    @testset "heat periodic basis=ComplexFourier N=$N dealias=$dealias T=$T stepper=$stepper" for
            N in [8],
            dealias in [1],
            T in [ComplexF64],
            stepper in timestepper_names
        c = Coordinate("x")
        d = Distributor(c, dtype=T)
        b = ComplexFourier(c, size=N, bounds=(0, 2 * pi), dealias=dealias)
        x = local_grid(d, b, scale=1)
        # Fields
        u = Field(d, bases=b)
        F = Field(d, bases=b)
        F["g"] = sin.(x)
        # Problem
        dx = A -> Differentiate(A, c)
        problem = IVP([u], namespace=Dict("u" => u, "F" => F, "dx" => dx, "dt" => dt))
        add_equation!(problem, "dt(u) - dx(dx(u)) = F")
        # Solver
        solver = build_solver(problem, stepper)
        dt_val = 1e-5
        n_iter = 20
        for i in 1:n_iter
            step!(solver, dt_val)
        end
        # Check solution
        amp = 1 - exp(-solver.sim_time)
        u_true = amp .* sin.(x)
        @test isapprox(u["g"], u_true, atol=1e-8)
    end

    @testset "heat periodic explicit check" begin
        # Simple explicit test with known timestepper
        c = Coordinate("x")
        d = Distributor(c, dtype=ComplexF64)
        b = ComplexFourier(c, size=8, bounds=(0, 2 * pi), dealias=1)
        x = local_grid(d, b, scale=1)
        u = Field(d, bases=b)
        F = Field(d, bases=b)
        F["g"] = sin.(x)
        dx = A -> Differentiate(A, c)
        problem = IVP([u], namespace=Dict("u" => u, "F" => F, "dx" => dx, "dt" => dt))
        add_equation!(problem, "dt(u) - dx(dx(u)) = F")
        solver = build_solver(problem, "RK111")
        dt_val = 1e-5
        for i in 1:20
            step!(solver, dt_val)
        end
        amp = 1 - exp(-solver.sim_time)
        u_true = amp .* sin.(x)
        @test isapprox(u["g"], u_true, atol=1e-8)
    end

end

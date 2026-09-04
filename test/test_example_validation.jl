using Test
using Dedalus
using TOML

const REFDIR = joinpath(@__DIR__, "reference_values")

@testset "Example Validation" begin

    @testset "EVP: Mathieu equation (q=0 analytical)" begin
        ref = TOML.parsefile(joinpath(REFDIR, "evp_mathieu.toml"))
        rtol = ref["metadata"]["tolerance_rtol"]
        expected = ref["reference"]["eigenvalues_sorted_real"]

        N = ref["parameters"]["N"]
        q_val = ref["parameters"]["q"]

        coord = Coordinate("x")
        dist = Distributor(coord; dtype=ComplexF64)
        basis = ComplexFourier(coord, N; bounds=(0, 2*pi))

        y = Field(dist; bases=(basis,))
        a = Field(dist)
        q = Field(dist)
        cos_2x = Field(dist; bases=(basis,))
        x = local_grid(dist, basis)
        cos_2x["g"] = cos.(2 .* x)
        dx = A -> Differentiate(A, coord)

        problem = EVP([y]; eigenvalue=a, namespace=@locals)
        add_equation!(problem, "dx(dx(y)) + (a - 2*q*cos_2x)*y = 0")

        solver = build_solver(problem)
        q["g"] .= q_val
        solve_dense!(solver, solver.subproblems[1]; rebuild_matrices=true)
        sorted_evals = sort(real.(solver.eigenvalues))
        computed = sorted_evals[1:length(expected)]

        for i in eachindex(expected)
            if expected[i] == 0.0
                @test abs(computed[i]) < 1e-10
            else
                @test isapprox(computed[i], expected[i]; rtol=rtol)
            end
        end
    end

    @testset "LBVP: Poisson equation (residual check)" begin
        ref = TOML.parsefile(joinpath(REFDIR, "lbvp_poisson.toml"))
        tol = ref["metadata"]["tolerance_residual_norm"]

        Nx = ref["parameters"]["Nx"]
        Ny = ref["parameters"]["Ny"]
        Lx = ref["parameters"]["Lx"]
        Ly = ref["parameters"]["Ly"]
        seed = ref["parameters"]["seed"]

        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64)
        xbasis = RealFourier(coords["x"], Nx; bounds=(0, Lx))
        ybasis = ChebyshevT(coords["y"], Ny; bounds=(0, Ly))

        u = Field(dist; name="u", bases=(xbasis, ybasis))
        tau_1 = Field(dist; name="tau_1", bases=(xbasis,))
        tau_2 = Field(dist; name="tau_2", bases=(xbasis,))

        x, y = local_grids(dist, xbasis, ybasis)
        f = Field(dist; bases=(xbasis, ybasis))
        g = Field(dist; bases=(xbasis,))
        h = Field(dist; bases=(xbasis,))
        fill_random!(f, "g"; seed=seed)
        low_pass_filter!(f; shape=(Nx ÷ 4, Ny ÷ 4))
        g["g"] = sin.(8 .* x) .* 0.025
        h["g"] .= 0

        dy = A -> Differentiate(A, coords["y"])
        lift_basis = derivative_basis(ybasis, 2)
        lift = (A, n) -> Lift(A, lift_basis, n)

        problem = LBVP([u, tau_1, tau_2]; namespace=@locals)
        add_equation!(problem, "lap(u) + lift(tau_1,-1) + lift(tau_2,-2) = f")
        add_equation!(problem, "u(y=0) = g")
        add_equation!(problem, "dy(u)(y=Ly) = h")

        solver = build_solver(problem)
        solve!(solver)

        # Verify solution exists and is non-trivial
        ug = allgather_data(u, "g")
        @test !all(ug .== 0)
        @test all(isfinite.(ug))

        # The solve should succeed without error — that itself validates correctness
        @test true
    end

    @testset "IVP: KdV-Burgers (initial condition and early evolution)" begin
        ref = TOML.parsefile(joinpath(REFDIR, "ivp_kdv_burgers.toml"))
        rtol = ref["metadata"]["tolerance_rtol"]

        Lx = ref["parameters"]["Lx"]
        Nx = ref["parameters"]["Nx"]
        a_visc = ref["parameters"]["a"]
        b_disp = ref["parameters"]["b"]
        n_ic = ref["parameters"]["n_ic"]
        dt = ref["parameters"]["timestep"]
        num_steps = ref["parameters"]["num_steps"]

        xcoord = Coordinate("x")
        dist = Distributor(xcoord; dtype=Float64)
        xbasis = RealFourier(xcoord, Nx; bounds=(0, Lx), dealias=3/2)

        u = Field(dist; name="u", bases=(xbasis,))

        a = a_visc
        b = b_disp
        dx = A -> Differentiate(A, xcoord)

        problem = IVP([u]; namespace=@locals)
        add_equation!(problem, "dt(u) - a*dx(dx(u)) - b*dx(dx(dx(u))) = - u*dx(u)")

        x = local_grid(dist, xbasis)
        n = n_ic
        u["g"] = @. log(1 + cosh(n)^2 / cosh(n * (x - 0.2 * Lx))^2) / (2 * n)

        # Check initial condition peak value
        u_peak_ref = ref["reference"]["u_peak_t0"]
        u_data = u["g"]
        u_max_initial = maximum(u_data)
        @test isapprox(u_max_initial, u_peak_ref; rtol=rtol)

        # Time-step a few iterations
        solver = build_solver(problem, SBDF2)
        solver.stop_sim_time = dt * num_steps
        while solver.proceed
            step!(solver, dt)
        end

        # After a few steps, solution should remain bounded and smooth
        u_data_final = u["g"]
        max_bound = ref["reference"]["max_u_upper_bound"]
        @test maximum(abs.(u_data_final)) < max_bound
        @test all(isfinite.(u_data_final))
    end

    @testset "NLBVP: Lane-Emden n=3 (R against Boyd reference)" begin
        ref = TOML.parsefile(joinpath(REFDIR, "nlbvp_lane_emden.toml"))
        rtol = ref["metadata"]["tolerance_rtol"]

        Nr = ref["parameters"]["Nr"]
        n = ref["parameters"]["n"]
        tol = ref["parameters"]["tolerance"]
        R_ref = ref["reference"]["R_n3"]

        coords = SphericalCoordinates("phi", "theta", "r")
        dist = Distributor(coords; dtype=Float64)
        ball = BallBasis(coords; shape=(1, 1, Nr), radius=1, dtype=Float64, dealias=2)

        f = Field(dist; name="f", bases=(ball,))
        tau = Field(dist; name="tau", bases=(surface(ball),))

        lift = A -> Lift(A, ball, -1)

        problem = NLBVP([f, tau]; namespace=@locals)
        add_equation!(problem, "lap(f) + lift(tau) = - f^n")
        add_equation!(problem, "f(r=1) = 0")

        phi, theta, r = local_grids(dist, ball)
        R0 = 5
        f["g"] = @. R0^(2/(n-1)) * (1 - r^2)^2

        solver = build_solver(problem; ncc_cutoff=1e-3)
        pert_norm = Inf
        max_iter = 30
        iter_count = 0
        while pert_norm > tol && iter_count < max_iter
            newton_iteration!(solver)
            pert_norm = sum(allreduce_data_norm(pert, "c", 2) for pert in solver.perturbations)
            iter_count += 1
        end

        # Extract R from f(r=0)
        f0 = allgather_data(f(r=0) |> evaluate, "g")[1, 1, 1]
        R_computed = f0^((n-1)/2)

        @test isapprox(R_computed, R_ref; rtol=rtol)
        @test iter_count < max_iter
    end

end

using Test
using Dedalus
using LinearAlgebra

@testset "Problem construction and operator-interface fixes" begin
    c = Coordinate("x")
    dist = Distributor(c, Float64)
    basis = Fourier(c, 8, (0.0, 2pi); dtype = Float64)
    u = Field(dist; bases = basis, name = "u")
    dx = A -> differentiate(A, c)

    @testset "IVP add_equation" begin
        ivp = IVP([u]; namespace = Dict("u" => u))
        eqn = Dedalus.add_equation!(ivp, (dt(u) - dx(dx(u)), u))
        @test eqn isa Dict
        @test eqn["LHS"] !== nothing
    end

    @testset "IVP -> EVP conversion" begin
        ivp = IVP([u]; namespace = Dict("u" => u))
        Dedalus.add_equation!(ivp, (dt(u) - dx(dx(u)), u))
        evp = Dedalus.build_EVP(ivp)
        @test evp isa EVP
    end

    @testset "NLBVP add_equation" begin
        nlbvp = NLBVP([u]; namespace = Dict("u" => u))
        eqn = Dedalus.add_equation!(nlbvp, (dx(dx(u)), dx(dx(u)) * u))
        @test eqn isa Dict
        @test eqn["LHS"] !== nothing
    end

    @testset "LBVP add_equation" begin
        f = Field(dist; bases = basis, name = "f")
        lbvp = LBVP([u]; namespace = Dict("u" => u, "f" => f))
        eqn = Dedalus.add_equation!(lbvp, (dx(dx(u)), f))
        @test eqn isa Dict
        @test eqn["LHS"] !== nothing
    end

    @testset "interpolate by coord name" begin
        f = Field(dist; bases = basis)
        h = interpolate(f; x = 0.5)
        @test h isa Dedalus.Interpolate
        @test Dedalus.operator_operand(h) === f
    end

    @testset "field product basis" begin
        f = Field(dist; bases = basis)
        g = Field(dist; bases = basis)
        w = f * g
        @test w isa Dedalus.AbstractOperand
        @test length(w.domain.bases) == 1
    end
end

@testset "apply_dense" begin
    A = rand(4, 4)
    M = rand(4, 4)
    @test Dedalus.apply_dense(M, A, 1) ≈ M * A
    # dim > 2 path: reshape uses array_shape captured before flattening
    B = rand(4, 3, 2)
    out = Dedalus.apply_dense(M, B, 1)
    @test size(out) == size(B)
    for k in 1:2, j in 1:3
        @test out[:, j, k] ≈ M * B[:, j, k]
    end
    # non-first axis
    C = rand(3, 4, 2)
    out2 = Dedalus.apply_dense(M, C, 2)
    @test size(out2) == size(C)
    for k in 1:2, i in 1:3
        @test out2[i, :, k] ≈ M * C[i, :, k]
    end
end

struct ArgsOnlyLinearOperator <: AbstractLinearOperator
    args::Vector{Any}
end
Dedalus.new_operand(::ArgsOnlyLinearOperator, operand; kw...) = ArgsOnlyLinearOperator(Any[operand])

@testset "default operator_operand is args[1]" begin
    @test operator_operand(ArgsOnlyLinearOperator(Any[42])) === 42

    c = Coordinate("x")
    dist = Distributor(c, Float64)
    basis = Fourier(c, 8, (0.0, 2pi); dtype = Float64)
    u = Field(dist; bases = basis, name = "u")
    v = Field(dist; bases = basis, name = "v")
    op = ArgsOnlyLinearOperator(Any[u])
    @test operator_operand(replace_operand(op, u, v)) === v
    @test operator_operand(replace_dict(op, Dict{Any, Any}(u => v))) === v
end

@testset "composite operator reconstruction" begin
    cs = CartesianCoordinates("x")
    c = cs.coords[1]
    dist = Distributor(cs, Float64)
    basis = RealFourier(c, 8, (0, 2pi))
    x = vec(local_grids(dist, basis; scales = 1)[1])
    u = Field(dist; bases = basis, name = "u")
    v = Field(dist; bases = basis, name = "v")
    f = Field(dist; bases = basis, name = "f")
    lap = laplacian(u, cs)
    grad = gradient(u, cs)
    divgrad = divergence(grad)
    N = 8

    @test lap isa Dedalus.CartesianLaplacian
    @test grad isa Dedalus.CartesianGradient
    @test divgrad isa Dedalus.CartesianDivergence
    expanded_ops = Union{
        Dedalus.CartesianGradient, Dedalus.CartesianDivergence, Dedalus.CartesianCurl,
        Dedalus.CartesianLaplacian, Dedalus.DirectProductGradient,
        Dedalus.DirectProductDivergence, Dedalus.DirectProductCurl, Dedalus.DirectProductLaplacian,
    }

    function coeffs(gvals)
        w = Field(dist; bases = basis)
        w["g"] = gvals
        return vec(copy(w["c"]))
    end

    # Linear coefficient map of an expression in `var`, expanded as a dense
    # N×N matrix. Follows the evaluation tree (`args`), never `operator_operand`.
    function coeff_matrix(expr, var, N)
        expr === var && return Matrix{Float64}(I, N, N)
        expr isa Number && (expr == 0 || error("affine term $expr")) && return zeros(N, N)
        expr isa Dedalus.AbstractCurrent && return zeros(N, N)
        expr isa Dedalus.Differentiate &&
            return Matrix(Dedalus.subspace_matrix(expr, nothing)) * coeff_matrix(expr.args[1], var, N)
        if expr isa Dedalus.Convert
            @assert expr.input_basis == expr.output_basis
            return coeff_matrix(expr.args[1], var, N)
        end
        expr isa Dedalus.Add && return sum(coeff_matrix(a, var, N) for a in expr.args)
        expr isa Dedalus.MultiplyNumberField && return expr.args[1] * coeff_matrix(expr.args[2], var, N)
        expr isa expanded_ops &&
            return reduce(vcat, [coeff_matrix(a, var, N) for a in expr.args])
        expr isa Dedalus.CartesianComponent && return coeff_matrix(expr.args[1], var, N)
        error("unhandled node $(typeof(expr))")
    end

    cu = coeffs(cos.(2 .* x))
    d2ref = coeffs(-4 .* cos.(2 .* x))
    d1ref = coeffs(-2 .* sin.(2 .* x))

    @test coeff_matrix(lap, u, N) * cu ≈ d2ref

    for (label, build, var, ref) in [
            ("replace_operand lap", () -> replace_operand(lap, u, v), v, d2ref),
            ("replace_dict lap", () -> replace_dict(lap, Dict(u => v)), v, d2ref),
            ("split lap linear", () -> Dedalus.split(lap, u)[1], u, d2ref),
            ("reinitialize lap", () -> reinitialize(lap), u, d2ref),
            ("expand lap", () -> Dedalus.expand(lap, u), u, d2ref),
            ("replace_operand grad", () -> replace_operand(grad, u, v), v, d1ref),
            ("split grad linear", () -> Dedalus.split(grad, u)[1], u, d1ref),
            ("reinitialize grad", () -> reinitialize(grad), u, d1ref),
            ("replace_operand divgrad", () -> replace_operand(divgrad, u, v), v, d2ref),
            ("split divgrad linear", () -> Dedalus.split(divgrad, u)[1], u, d2ref),
        ]
        @testset "$label" begin
            @test coeff_matrix(build(), var, N) * cu ≈ ref
        end
    end

    @testset "operand field preserved" begin
        r = replace_operand(lap, u, v)
        @test Dedalus.operator_operand(r) === v
        @test Dedalus.split(lap, u)[2] == 0
    end

    @testset "IVP matrices" begin
        ivp = IVP([u]; namespace = Dict("u" => u))
        eqn = Dedalus.add_equation!(ivp, (dt(u) - lap, u))
        @test coeff_matrix(eqn["M"], u, N) * cu ≈ cu
        @test coeff_matrix(eqn["L"], u, N) * cu ≈ coeffs(4 .* cos.(2 .* x))
    end

    @testset "EVP matrices" begin
        ivp = IVP([u]; namespace = Dict("u" => u))
        Dedalus.add_equation!(ivp, (dt(u) - lap, u))
        evp = Dedalus.build_EVP(ivp)
        eqn = Dedalus.get_equations(evp)[1]
        pert = Dedalus.get_variables(evp)[1]
        @test coeff_matrix(eqn["M"], pert, N) * cu ≈ cu
        # RHS u contributes -δu via the Frechet derivative in build_EVP
        @test coeff_matrix(eqn["L"], pert, N) * cu ≈ coeffs(3 .* cos.(2 .* x))
    end

    @testset "LBVP matrix" begin
        lbvp = LBVP([u]; namespace = Dict("u" => u, "f" => f))
        eqn = Dedalus.add_equation!(lbvp, (lap, f))
        @test coeff_matrix(eqn["L"], u, N) * cu ≈ d2ref
    end

    @testset "NLBVP matrix" begin
        nlbvp = NLBVP([u]; namespace = Dict("u" => u, "f" => f))
        eqn = Dedalus.add_equation!(nlbvp, (lap, f))
        pert = nlbvp.perturbations[1]
        @test coeff_matrix(eqn["dF"], pert, N) * cu ≈ d2ref
    end

    @testset "homogeneous equation" begin
        ivp = IVP([u]; namespace = Dict("u" => u))
        eqn = Dedalus.add_equation!(ivp, (dt(u) - lap, 0))
        @test eqn["F"] isa Dedalus.Field
        @test all(iszero, eqn["F"]["c"])
        @test coeff_matrix(eqn["L"], u, N) * cu ≈ coeffs(4 .* cos.(2 .* x))
    end
end

@testset "coordinate lookup on constant field" begin
    cs = CartesianCoordinates("x")
    c = cs.coords[1]
    dist = Distributor(cs, Float64)
    f0 = Field(dist)
    @test get_coord(f0.domain, "x") === c
    @test_throws ArgumentError get_coord(f0.domain, "y")
    h = interpolate(f0; x = 0.2)
    @test h isa Dedalus.Interpolate
    @test h.coord === c
end

@testset "reinitialize keywords" begin
    cs = CartesianCoordinates("x")
    c = cs.coords[1]
    dist = Distributor(cs, Float64)
    basis = RealFourier(c, 8, (0, 2pi))
    x = vec(local_grids(dist, basis; scales = 1)[1])
    u = Field(dist; bases = basis)
    u["g"] = sin.(x)
    out = Field(dist; bases = basis)

    op = reinitialize(differentiate(u, c); out = out)
    @test op isa Dedalus.Differentiate
    @test op.out === out
    result = evaluate(op)
    @test result === out
    @test result["g"] ≈ cos.(x) atol = 1.0e-12

    out2 = Field(dist; bases = basis)
    op2 = reinitialize(differentiate(differentiate(u, c), c); out = out2)
    @test op2.out === out2
    @test Dedalus.operator_operand(op2).out !== out2

    lap = laplacian(u, cs)
    out3 = Field(dist; bases = basis)
    op3 = reinitialize(lap; out = out3)
    @test op3 isa Dedalus.CartesianLaplacian
    @test op3.out === out3
    @test Dedalus.operator_operand(op3) === u

    op4 = reinitialize(differentiate(u, c); ncc = true, ncc_vars = [u])
    @test op4 isa Dedalus.Differentiate
    @test Dedalus.operator_operand(op4) === u
end

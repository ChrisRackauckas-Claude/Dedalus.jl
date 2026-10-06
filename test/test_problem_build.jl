using Test
using Dedalus

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

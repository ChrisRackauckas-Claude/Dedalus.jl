using Test
using Dedalus

module MockMPI
    const SUM = :sum
    const MAX = :max
end

@testset "Unbound-name regressions" begin
    @testset "jacobi_quadrature days=$days" for days in (-2, 0, 3)
        for n in (1, 2, 8), (a, b) in ((0.0, 0.0), (-0.9, 0.2), (2.0, 1.0))
            z, w = Dedalus.jacobi_quadrature(n, a, b; days = days)
            @test all(isfinite, z)
            @test all(>(0), w)
            # ∫(1-z)^a (1+z)^b ((1+z)/2)^k dz = jacobi_mass(a, b + k) / 2^k, exact for k ≤ 2n-1
            for k in 0:(2n - 1)
                exact = Dedalus.jacobi_mass(a, b + k) / 2^k
                @test sum(w .* ((z .+ 1) ./ 2) .^ k) ≈ exact rtol = 1.0e-12
            end
        end
    end

    @testset "jacobi_quadrature days ≤ 1 performs one Newton step" begin
        for (a, b) in ((0.0, 0.0), (-0.9, 0.2))
            @test Dedalus.jacobi_quadrature(8, a, b; days = 0) ==
                Dedalus.jacobi_quadrature(8, a, b; days = 1)
        end
    end

    @testset "field MPI reductions use the injected MPI module" begin
        c = CartesianCoordinates("x", "y")
        d = Distributor(c, Float64)
        f = Field(d)
        f.data .= 3
        reduce_mock(x, op) = op === MockMPI.SUM ? 2x : (op === MockMPI.MAX ? x : error("unexpected op $op"))
        gather_mock(send, recv, op) = op === MockMPI.SUM ? (recv .= 2 .* send) : error("unexpected op $op")
        d.comm = (size = 2, rank = 0, allreduce = reduce_mock, Allreduce = gather_mock)
        old = Dedalus._mpi_module[]
        Dedalus._mpi_module[] = MockMPI
        try
            @test Dedalus.allreduce_data_norm(f) == sqrt(18)
            @test Dedalus.allreduce_data_norm(f; order = Inf) == 3
            @test Dedalus.allgather_data(f) == 2 .* f.data
        finally
            Dedalus._mpi_module[] = old
        end
    end
end

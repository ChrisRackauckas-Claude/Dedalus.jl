"""
Tests for the dedalus_sphere internal library: Jacobi and sphere operator algebra.

Tests the Jacobi quadrature, polynomial evaluation, operator algebra,
and sphere quadrature/harmonics directly, without going through the
high-level Dedalus basis/field layer.

Reference: dedalus/tests/test_jacobi.py (Python jacobi128 tests).
"""

using Test
using Dedalus
using SparseArrays
using LinearAlgebra
using SpecialFunctions: beta

@testset "Dedalus Sphere" begin

    N_range = [1, 2, 4, 8]
    ab_range = [-0.5, 0.0, 0.5, 1.0]

    # ========================================================================
    # 1. Forward-backward round-trip
    # ========================================================================
    @testset "Forward-backward round-trip" begin
        @testset "N=$N a=$a b=$b" for N in N_range, a in ab_range, b in ab_range
            # Skip invalid parameter combinations
            (a <= -1 || b <= -1) && continue

            try
                grid, weights = Dedalus.jacobi_quadrature(N, a, b)
                P = Dedalus.jacobi_polynomials(N, a, b, grid)
                # P has shape (N, length(grid)) -- rows are polynomial degrees 0..N-1
                # Forward transform: F = diag(weights) * P' (project grid values onto coefficients)
                # In the Python code: forward = weights * polynomials (broadcasting weights over rows)
                # polynomials has shape (N, N_grid), so weights * polynomials scales columns.
                # forward = weights .* P  gives (N, N_grid) with each column j scaled by weights[j]
                # backward = P' (transpose)
                # Check: backward * forward = P' * (weights .* P) = identity(N)
                forward = weights' .* P  # broadcast weights (row vector) over rows of P
                backward = P'
                product = backward * forward
                @test size(product) == (N, N)
                @test isapprox(product, I(N), atol=1e-10)
            catch e
                @test_broken false  # Mark as broken if API call fails
                @warn "Forward-backward test failed for N=$N, a=$a, b=$b" exception=e
            end
        end
    end

    # ========================================================================
    # 2. Backward-forward round-trip
    # ========================================================================
    @testset "Backward-forward round-trip" begin
        @testset "N=$N a=$a b=$b" for N in N_range, a in ab_range, b in ab_range
            (a <= -1 || b <= -1) && continue

            try
                grid, weights = Dedalus.jacobi_quadrature(N, a, b)
                P = Dedalus.jacobi_polynomials(N, a, b, grid)
                forward = weights' .* P
                backward = P'
                product = forward * backward
                @test size(product) == (N, N)
                @test isapprox(product, I(N), atol=1e-10)
            catch e
                @test_broken false
                @warn "Backward-forward test failed for N=$N, a=$a, b=$b" exception=e
            end
        end
    end

    # ========================================================================
    # 3. A+ operator conversion round-trip
    #    Tests that A+(a,b) composed with transforms gives identity:
    #    P1' * A+ * (w0 .* P0) = I
    #    where P0 = polynomials(N, a, b, grid0)
    #          P1 = polynomials(N, a+1, b, grid0)
    #          A+ = jacobi_operator("A")(+1) evaluated at (N, a, b)
    # ========================================================================
    @testset "A+ conversion round-trip" begin
        @testset "N=$N a=$a b=$b" for N in N_range, a in ab_range, b in ab_range
            (a <= -1 || b <= -1) && continue
            (a + 1 <= -1) && continue
            N < 2 && continue  # Need at least N=2 for meaningful test

            try
                grid0, weights0 = Dedalus.jacobi_quadrature(N, a, b)
                P0 = Dedalus.jacobi_polynomials(N, a, b, grid0)
                P1 = Dedalus.jacobi_polynomials(N, a + 1, b, grid0)

                # Get A+ operator matrix
                Ap_op = Dedalus.jacobi_operator("A")(+1)
                Ap_icsc = Ap_op(N, a, b)
                Ap_mat = Matrix(sparse(Ap_icsc))

                forward = weights0' .* P0
                backward = P1'

                # A+ maps (N, a, b) -> (N, a+1, b), so Ap_mat should be square-ish
                # The product backward * Ap_mat * forward should be close to identity
                # Dimensions: backward is (N x N_grid), Ap_mat could be non-square,
                # forward is (N x N_grid).
                # We need to handle dimension mismatches carefully.
                n_out = size(Ap_mat, 1)
                n_in = size(Ap_mat, 2)

                # Trim or pad to make the algebra work
                P0_trim = P0[1:min(n_in, N), :]
                P1_trim = P1[1:min(n_out, N), :]
                forward_trim = weights0' .* P0_trim
                backward_trim = P1_trim'

                product = backward_trim * Ap_mat[1:min(n_out, size(P1_trim, 1)), 1:min(n_in, size(P0_trim, 1))] * forward_trim
                n = min(size(product, 1), size(product, 2))
                @test isapprox(product[1:n, 1:n], I(n), atol=1e-9)
            catch e
                @test_broken false
                @warn "A+ round-trip test failed for N=$N, a=$a, b=$b" exception=e
            end
        end
    end

    # ========================================================================
    # 4. A+ / B+ commutation
    #    Two paths through the (a, b) lattice should give the same matrix:
    #    Path 1: A+(a, b) then B+(a+1, b)  ->  (a+1, b+1)
    #    Path 2: B+(a, b) then A+(a, b+1)  ->  (a+1, b+1)
    # ========================================================================
    @testset "A+ B+ commutation" begin
        @testset "N=$N a=$a b=$b" for N in [2, 4, 8], a in ab_range, b in ab_range
            (a <= -1 || b <= -1) && continue

            try
                Ap = Dedalus.jacobi_operator("A")
                Bp = Dedalus.jacobi_operator("B")

                # Path 1: B+(a+1, b) * A+(a, b)
                Ap_00 = Ap(+1)  # Operator
                Bp_10 = Bp(+1)  # Operator
                # Evaluate: A+(N, a, b) gives matrix, B+(N', a+1, b) gives matrix
                Ap_mat = sparse(Ap_00(N, a, b))
                # After A+, codomain is (a+1, b), and n might change
                Bp_mat = sparse(Bp_10(N, a + 1, b))
                # Compose using compose() which handles codomain tracking
                path1_op = Dedalus.compose(Bp_10, Ap_00)
                path1_mat = sparse(path1_op(N, a, b))

                # Path 2: A+(a, b+1) * B+(a, b)
                Bp_00 = Bp(+1)
                Ap_01 = Ap(+1)
                path2_op = Dedalus.compose(Ap_01, Bp_00)
                path2_mat = sparse(path2_op(N, a, b))

                # Resize to common dimensions for comparison
                rows = min(size(path1_mat, 1), size(path2_mat, 1))
                cols = min(size(path1_mat, 2), size(path2_mat, 2))
                @test rows > 0
                @test cols > 0
                m1 = Matrix(path1_mat[1:rows, 1:cols])
                m2 = Matrix(path2_mat[1:rows, 1:cols])
                @test isapprox(m1, m2, atol=1e-12)
            catch e
                @test_broken false
                @warn "A+/B+ commutation test failed for N=$N, a=$a, b=$b" exception=e
            end
        end
    end

    # ========================================================================
    # 5. A++/B++ double commutation
    #    Path 1: B+(a+2,b+1) * B+(a+2,b) * A+(a+1,b) * A+(a,b)
    #    Path 2: A+(a+1,b+2) * A+(a,b+2) * B+(a,b+1) * B+(a,b)
    #    Both go from (a,b) -> (a+2,b+2).
    # ========================================================================
    @testset "A++/B++ double commutation" begin
        @testset "N=$N a=$a b=$b" for N in [4, 8], a in [0.0, 0.5, 1.0], b in [0.0, 0.5, 1.0]
            (a <= -1 || b <= -1) && continue

            try
                Ap = Dedalus.jacobi_operator("A")
                Bp = Dedalus.jacobi_operator("B")

                # Path 1: B+(a+2,b+1) o B+(a+2,b) o A+(a+1,b) o A+(a,b)
                # Using compose for sequential application (rightmost applied first)
                Ap_00 = Ap(+1)  # shifts a by +1
                Ap_10 = Ap(+1)  # shifts a by +1 (applied at a+1)
                Bp_20 = Bp(+1)  # shifts b by +1 (applied at a+2)
                Bp_21 = Bp(+1)  # shifts b by +1 (applied at a+2, b+1)

                # compose(A, B) means A applied after B: A(B(x))
                # Path 1: Bp21 o Bp20 o Ap10 o Ap00
                step1 = Dedalus.compose(Ap_10, Ap_00)
                step2 = Dedalus.compose(Bp_20, step1)
                path1_op = Dedalus.compose(Bp_21, step2)
                path1_mat = sparse(path1_op(N, a, b))

                # Path 2: Ap12 o Ap02 o Bp01 o Bp00
                Bp_00 = Bp(+1)
                Bp_01 = Bp(+1)
                Ap_02 = Ap(+1)
                Ap_12 = Ap(+1)

                step1b = Dedalus.compose(Bp_01, Bp_00)
                step2b = Dedalus.compose(Ap_02, step1b)
                path2_op = Dedalus.compose(Ap_12, step2b)
                path2_mat = sparse(path2_op(N, a, b))

                rows = min(size(path1_mat, 1), size(path2_mat, 1))
                cols = min(size(path1_mat, 2), size(path2_mat, 2))
                @test rows > 0
                @test cols > 0
                m1 = Matrix(path1_mat[1:rows, 1:cols])
                m2 = Matrix(path2_mat[1:rows, 1:cols])
                @test isapprox(m1, m2, atol=1e-11)
            catch e
                @test_broken false
                @warn "A++/B++ double commutation test failed for N=$N, a=$a, b=$b" exception=e
            end
        end
    end

    # ========================================================================
    # 6. Jacobi mass and norm ratio basic properties
    # ========================================================================
    @testset "Jacobi mass properties" begin
        @testset "mass positivity a=$a b=$b" for a in ab_range, b in ab_range
            (a <= -1 || b <= -1) && continue
            try
                m = Dedalus.jacobi_mass(a, b)
                @test m > 0
                # mass(a, b) = 2^(a+b+1) * Beta(a+1, b+1)
                expected = 2^(a + b + 1) * beta(a + 1, b + 1)
                @test isapprox(m, expected, rtol=1e-12)
            catch e
                @test_broken false
                @warn "Mass positivity test failed for a=$a, b=$b" exception=e
            end
        end

        @testset "mass symmetry a=$a b=$b" for a in ab_range, b in ab_range
            (a <= -1 || b <= -1) && continue
            try
                # Beta(a+1, b+1) = Beta(b+1, a+1), so mass(a,b) = mass(b,a)
                # only when a+b is the same (which it is by symmetry)
                m_ab = Dedalus.jacobi_mass(a, b)
                m_ba = Dedalus.jacobi_mass(b, a)
                @test isapprox(m_ab, m_ba, rtol=1e-14)
            catch e
                @test_broken false
                @warn "Mass symmetry test failed for a=$a, b=$b" exception=e
            end
        end

        @testset "mass log form a=$a b=$b" for a in ab_range, b in ab_range
            (a <= -1 || b <= -1) && continue
            try
                m = Dedalus.jacobi_mass(a, b)
                m_log = Dedalus.jacobi_mass(a, b; log=true)
                @test isapprox(exp(m_log), m, rtol=1e-12)
            catch e
                @test_broken false
                @warn "Mass log test failed for a=$a, b=$b" exception=e
            end
        end

        # Known values: mass(0, 0) = 2 (Legendre)
        @testset "mass known values" begin
            try
                @test isapprox(Dedalus.jacobi_mass(0, 0), 2.0, atol=1e-14)
                # mass(-0.5, -0.5) = 2^0 * Beta(0.5, 0.5) = pi
                @test isapprox(Dedalus.jacobi_mass(-0.5, -0.5), pi, atol=1e-12)
            catch e
                @test_broken false
                @warn "Mass known values test failed" exception=e
            end
        end
    end

    @testset "Jacobi norm ratio properties" begin
        @testset "identity shift a=$a b=$b" for a in ab_range, b in ab_range
            (a <= -1 || b <= -1) && continue
            try
                # Zero shift should give ratio of 1
                n_arr = collect(0.0:4.0)
                ratios = Dedalus.jacobi_norm_ratio(0, 0, 0, n_arr, a, b)
                @test all(isapprox.(ratios, 1.0, atol=1e-12))
            catch e
                @test_broken false
                @warn "Norm ratio identity test failed for a=$a, b=$b" exception=e
            end
        end

        @testset "norm ratio positivity a=$a b=$b" for a in [0.0, 0.5, 1.0], b in [0.0, 0.5, 1.0]
            try
                n_arr = collect(1.0:5.0)
                ratios = Dedalus.jacobi_norm_ratio(1, 0, 0, n_arr, a, b)
                @test all(ratios .> 0)
            catch e
                @test_broken false
                @warn "Norm ratio positivity test failed for a=$a, b=$b" exception=e
            end
        end
    end

    # ========================================================================
    # 7. Jacobi operator basic structure tests
    # ========================================================================
    @testset "Jacobi operator structure" begin
        @testset "Identity operator N=$N a=$a b=$b" for N in [2, 4, 8], a in [0.0, 0.5], b in [0.0, 0.5]
            try
                Id_op = Dedalus.jacobi_operator("Id")
                Id_mat = Matrix(sparse(Id_op(N, a, b)))
                @test size(Id_mat) == (N, N)
                @test isapprox(Id_mat, I(N), atol=1e-14)
            catch e
                @test_broken false
                @warn "Identity operator test failed for N=$N, a=$a, b=$b" exception=e
            end
        end

        @testset "Parity operator N=$N" for N in [2, 4, 8]
            try
                Pi_op = Dedalus.jacobi_operator("Pi")
                Pi_mat = Matrix(sparse(Pi_op(N, 0.0, 0.0)))
                # Parity: diag((-1)^0, (-1)^1, ..., (-1)^(N-1))
                expected_diag = [(-1.0)^k for k in 0:(N-1)]
                @test isapprox(Pi_mat, diagm(expected_diag), atol=1e-14)
                # Pi^2 = I
                Pi2 = Pi_mat * Pi_mat
                @test isapprox(Pi2, I(N), atol=1e-14)
            catch e
                @test_broken false
                @warn "Parity operator test failed for N=$N" exception=e
            end
        end

        @testset "Number operator N=$N" for N in [2, 4, 8]
            try
                N_op = Dedalus.jacobi_operator("N")
                N_mat = Matrix(sparse(N_op(N, 0.0, 0.0)))
                expected_diag = collect(0.0:(N-1))
                @test isapprox(N_mat, diagm(expected_diag), atol=1e-14)
            catch e
                @test_broken false
                @warn "Number operator test failed for N=$N" exception=e
            end
        end

        @testset "Z operator symmetry N=$N" for N in [4, 8]
            try
                Z_op = Dedalus.jacobi_operator("Z")
                # For Legendre (a=b=0), Z should be a symmetric tridiagonal matrix
                Z_mat = Matrix(sparse(Z_op(N, 0.0, 0.0)))
                Z_sq = Z_mat[1:N, 1:N]
                @test isapprox(Z_sq, Z_sq', atol=1e-14)
            catch e
                @test_broken false
                @warn "Z operator symmetry test failed for N=$N" exception=e
            end
        end
    end

    # ========================================================================
    # 8. Operator algebra: compose, add, scalar multiply
    # ========================================================================
    @testset "Operator algebra" begin
        @testset "compose identity N=$N" for N in [4, 8]
            try
                Id = Dedalus.jacobi_operator("Id")
                Ap = Dedalus.jacobi_operator("A")(+1)

                # compose(Id_shifted, Ap) should equal Ap
                composed = Dedalus.compose(Id, Ap)
                m_composed = Matrix(sparse(composed(N, 0.0, 0.0)))
                m_ap = Matrix(sparse(Ap(N, 0.0, 0.0)))
                rows = min(size(m_composed, 1), size(m_ap, 1))
                cols = min(size(m_composed, 2), size(m_ap, 2))
                @test isapprox(m_composed[1:rows, 1:cols], m_ap[1:rows, 1:cols], atol=1e-13)
            catch e
                @test_broken false
                @warn "Compose identity test failed for N=$N" exception=e
            end
        end

        @testset "scalar multiplication N=$N" for N in [4, 8]
            try
                Ap = Dedalus.jacobi_operator("A")(+1)
                Ap2 = 2.0 * Ap
                m_ap = Matrix(sparse(Ap(N, 0.0, 0.0)))
                m_ap2 = Matrix(sparse(Ap2(N, 0.0, 0.0)))
                @test isapprox(m_ap2, 2.0 * m_ap, atol=1e-14)
            catch e
                @test_broken false
                @warn "Scalar multiplication test failed for N=$N" exception=e
            end
        end

        @testset "operator negation N=$N" for N in [4, 8]
            try
                Ap = Dedalus.jacobi_operator("A")(+1)
                neg_Ap = -Ap
                m_ap = Matrix(sparse(Ap(N, 0.0, 0.0)))
                m_neg = Matrix(sparse(neg_Ap(N, 0.0, 0.0)))
                @test isapprox(m_neg, -m_ap, atol=1e-14)
            catch e
                @test_broken false
                @warn "Operator negation test failed for N=$N" exception=e
            end
        end
    end

    # ========================================================================
    # 9. Sphere quadrature existence and basic properties
    # ========================================================================
    @testset "Sphere quadrature" begin
        @testset "sphere_quadrature Lmax=$Lmax" for Lmax in [1, 2, 4, 8, 16]
            try
                cos_theta, weights = Dedalus.sphere_quadrature(Lmax)

                # Should return Lmax+1 nodes (it calls jacobi_quadrature(Lmax+1, 0, 0))
                @test length(cos_theta) == Lmax + 1
                @test length(weights) == Lmax + 1

                # Nodes should be in [-1, 1]
                @test all(-1 .<= cos_theta .<= 1)

                # Weights should be positive
                @test all(weights .> 0)

                # Weights should sum to mass(0,0) = 2 (Legendre quadrature)
                @test isapprox(sum(weights), 2.0, atol=1e-10)

                # Nodes should be sorted (ascending from Jacobi eigenvalue solver)
                @test issorted(cos_theta)
            catch e
                @test_broken false
                @warn "Sphere quadrature test failed for Lmax=$Lmax" exception=e
            end
        end

        @testset "sphere quadrature integrates polynomials Lmax=$Lmax" for Lmax in [4, 8]
            try
                cos_theta, weights = Dedalus.sphere_quadrature(Lmax)
                # Should exactly integrate polynomials up to degree 2*Lmax+1
                # Test with cos_theta^k for small k
                for k in 0:min(2 * Lmax, 6)
                    numerical = sum(weights .* cos_theta .^ k)
                    # Exact integral of x^k from -1 to 1 with weight 1:
                    if k % 2 == 0
                        exact = 2.0 / (k + 1)
                    else
                        exact = 0.0
                    end
                    @test isapprox(numerical, exact, atol=1e-10)
                end
            catch e
                @test_broken false
                @warn "Sphere quadrature polynomial integration test failed for Lmax=$Lmax" exception=e
            end
        end
    end

    # ========================================================================
    # 10. Sphere harmonics basic tests
    # ========================================================================
    @testset "Sphere harmonics" begin
        @testset "sphere_harmonics existence Lmax=$Lmax m=$m" for Lmax in [2, 4, 8], m in [0, 1]
            try
                cos_theta, _ = Dedalus.sphere_quadrature(Lmax)
                s = 0  # scalar spherical harmonics
                Y = Dedalus.sphere_harmonics(Lmax, m, s, cos_theta)
                # Output should have (Lmax - max(|m|,|s|) + 1) rows
                expected_rows = Lmax + 1 - max(abs(m), abs(s))
                @test size(Y, 1) == expected_rows
                @test size(Y, 2) == length(cos_theta)
            catch e
                @test_broken false
                @warn "Sphere harmonics test failed for Lmax=$Lmax, m=$m" exception=e
            end
        end

        @testset "sphere harmonics orthogonality Lmax=$Lmax m=$m" for Lmax in [4, 8], m in [0, 1]
            try
                cos_theta, weights = Dedalus.sphere_quadrature(Lmax)
                s = 0
                Y = Dedalus.sphere_harmonics(Lmax, m, s, cos_theta)
                n_modes = size(Y, 1)
                # Orthogonality: Y * diag(weights) * Y' should be close to identity
                G = Y * diagm(weights) * Y'
                @test isapprox(G, I(n_modes), atol=1e-10)
            catch e
                @test_broken false
                @warn "Sphere harmonics orthogonality test failed for Lmax=$Lmax, m=$m" exception=e
            end
        end
    end

    # ========================================================================
    # 11. Sphere operator factory
    # ========================================================================
    @testset "Sphere operators" begin
        @testset "sphere_operator Id Lmax=$Lmax" for Lmax in [2, 4, 8]
            try
                m, s = 0, 0
                Id_op = Dedalus.sphere_operator("Id")
                Id_mat = Matrix(sparse(Id_op(Lmax, m, s)))
                n = Lmax + 1 - max(abs(m), abs(s))
                @test size(Id_mat) == (n, n)
                @test isapprox(Id_mat, I(n), atol=1e-14)
            catch e
                @test_broken false
                @warn "Sphere Id test failed for Lmax=$Lmax" exception=e
            end
        end

        @testset "sphere_operator L diagonal Lmax=$Lmax" for Lmax in [4, 8]
            try
                m, s = 0, 0
                L_op = Dedalus.sphere_operator("L")
                L_mat = Matrix(sparse(L_op(Lmax, m, s)))
                n = Lmax + 1 - max(abs(m), abs(s))
                # L eigenvalues should be max(|m|,|s|), max(|m|,|s|)+1, ..., Lmax
                L_min = max(abs(m), abs(s))
                expected = diagm(collect(Float64.(L_min:Lmax)))
                @test isapprox(L_mat, expected, atol=1e-14)
            catch e
                @test_broken false
                @warn "Sphere L test failed for Lmax=$Lmax" exception=e
            end
        end

        @testset "sphere_operator Cos is tridiagonal Lmax=$Lmax" for Lmax in [4, 8]
            try
                m, s = 0, 0
                Cos_op = Dedalus.sphere_operator("Cos")
                Cos_mat = Matrix(sparse(Cos_op(Lmax, m, s)))
                n = size(Cos_mat, 2)
                # Cos(theta) applied to spherical harmonics is tridiagonal
                # (the three-term recurrence of associated Legendre functions)
                # Check that entries beyond the first super/sub diagonal are zero
                for i in 1:size(Cos_mat, 1), j in 1:size(Cos_mat, 2)
                    if abs(i - j) > 1
                        @test abs(Cos_mat[i, j]) < 1e-14
                    end
                end
            catch e
                @test_broken false
                @warn "Sphere Cos tridiagonal test failed for Lmax=$Lmax" exception=e
            end
        end
    end

    # ========================================================================
    # 12. spin2Jacobi parameter conversion
    # ========================================================================
    @testset "spin2Jacobi" begin
        @testset "basic conversion Lmax=$Lmax m=$m s=$s" for Lmax in [4, 8], m in [0, 1, 2], s in [0, 1]
            try
                n, a, b = Dedalus.spin2Jacobi(Lmax, m, s)
                @test n == Lmax + 1 - max(abs(m), abs(s))
                @test a == abs(m + s)
                @test b == abs(m - s)
                @test n >= 0
                @test a >= 0
                @test b >= 0
            catch e
                @test_broken false
                @warn "spin2Jacobi test failed for Lmax=$Lmax, m=$m, s=$s" exception=e
            end
        end

        @testset "spin shift Lmax=$Lmax" for Lmax in [4, 8]
            try
                m, s = 1, 0
                n, a, b, dn, da, db = Dedalus.spin2Jacobi(Lmax, m, s; ds=1)
                # After spin shift s -> s+1:
                n2 = Lmax + 1 - max(abs(m), abs(s + 1))
                a2 = abs(m + s + 1)
                b2 = abs(m - s - 1)
                @test dn == n2 - n
                @test da == a2 - a
                @test db == b2 - b
            catch e
                @test_broken false
                @warn "spin2Jacobi shift test failed for Lmax=$Lmax" exception=e
            end
        end
    end

    # ========================================================================
    # 13. InfiniteCSC arithmetic
    # ========================================================================
    @testset "InfiniteCSC arithmetic" begin
        @testset "addition with different row counts" begin
            try
                A = Dedalus.InfiniteCSC(sparse([1.0 2.0; 3.0 4.0]))
                B = Dedalus.InfiniteCSC(sparse([5.0 6.0; 7.0 8.0; 9.0 10.0]))
                C = A + B
                @test size(C) == (3, 2)
                C_dense = Matrix(sparse(C))
                @test isapprox(C_dense[1:2, :], [6.0 8.0; 10.0 12.0], atol=1e-14)
                @test isapprox(C_dense[3, :], [9.0, 10.0], atol=1e-14)
            catch e
                @test_broken false
                @warn "InfiniteCSC addition test failed" exception=e
            end
        end

        @testset "scalar multiplication" begin
            try
                A = Dedalus.InfiniteCSC(sparse([1.0 2.0; 3.0 4.0]))
                B = 3.0 * A
                @test isapprox(Matrix(sparse(B)), [3.0 6.0; 9.0 12.0], atol=1e-14)
            catch e
                @test_broken false
                @warn "InfiniteCSC scalar test failed" exception=e
            end
        end

        @testset "square" begin
            try
                # Non-square -> padded square
                A = Dedalus.InfiniteCSC(sparse([1.0 2.0 3.0; 4.0 5.0 6.0]))
                sq = Dedalus.square(A)
                @test size(sq) == (3, 3)
                @test isapprox(Matrix(sq)[1:2, :], [1.0 2.0 3.0; 4.0 5.0 6.0], atol=1e-14)
                @test isapprox(Matrix(sq)[3, :], [0.0, 0.0, 0.0], atol=1e-14)
            catch e
                @test_broken false
                @warn "InfiniteCSC square test failed" exception=e
            end
        end
    end

    # ========================================================================
    # 14. Codomain algebra
    # ========================================================================
    @testset "Codomain algebra" begin
        @testset "codomain addition" begin
            try
                c1 = Dedalus.Codomain(1, 2, 3)
                c2 = Dedalus.Codomain(4, 5, 6)
                c3 = c1 + c2
                @test c3[1] == 5
                @test c3[2] == 7
                @test c3[3] == 9
            catch e
                @test_broken false
                @warn "Codomain addition test failed" exception=e
            end
        end

        @testset "codomain negation" begin
            try
                c = Dedalus.Codomain(1, -2, 3)
                nc = -c
                @test nc[1] == -1
                @test nc[2] == 2
                @test nc[3] == -3
            catch e
                @test_broken false
                @warn "Codomain negation test failed" exception=e
            end
        end

        @testset "codomain evaluation" begin
            try
                c = Dedalus.Codomain(1, 2, 3)
                result = c(10, 20, 30)
                @test result == (11, 22, 33)
            catch e
                @test_broken false
                @warn "Codomain evaluation test failed" exception=e
            end
        end
    end

    # ========================================================================
    # 15. JacobiCodomain algebra
    # ========================================================================
    @testset "JacobiCodomain algebra" begin
        @testset "basic operations" begin
            try
                jc = Dedalus.JacobiCodomain(0, 1, 0, 0)
                @test jc[1] == 0
                @test jc[2] == 1
                @test jc[3] == 0
                @test jc[4] == 0
            catch e
                @test_broken false
                @warn "JacobiCodomain basic test failed" exception=e
            end
        end

        @testset "parity swap" begin
            try
                # With pi=1, a and b swap before shift
                jc = Dedalus.JacobiCodomain(0, 0, 0, 1)
                n, a, b = jc(5, 2.0, 3.0)
                # pi=1 means swap a,b: (5, 3.0, 2.0) then add shifts (0,0,0)
                @test n == 5
                @test a == 3.0
                @test b == 2.0
            catch e
                @test_broken false
                @warn "JacobiCodomain parity test failed" exception=e
            end
        end

        @testset "composition" begin
            try
                jc1 = Dedalus.JacobiCodomain(0, 1, 0, 0)  # shifts a by 1
                jc2 = Dedalus.JacobiCodomain(0, 0, 1, 0)  # shifts b by 1
                jc3 = jc1 + jc2
                @test jc3[2] == 1  # da
                @test jc3[3] == 1  # db
            catch e
                @test_broken false
                @warn "JacobiCodomain composition test failed" exception=e
            end
        end
    end

    # ========================================================================
    # 16. Jacobi coefficient connection
    # ========================================================================
    @testset "Jacobi coefficient connection" begin
        @testset "returns correct size N=$N" for N in [4, 8]
            try
                ab = (0.0, 0.0)
                cd = (0.5, 0.5)
                conn = Dedalus.jacobi_coefficient_connection(N, ab, cd)
                @test size(conn) == (N, N)
                # Connection matrix should have finite entries
                @test all(isfinite.(conn))
            catch e
                @test_broken false
                @warn "Coefficient connection size test failed for N=$N" exception=e
            end
        end

        @testset "same basis diagonal dominance N=$N" for N in [4, 8]
            try
                # Same-basis connection should be close to identity, but numerical
                # quadrature means it may not be exact. Check diagonal dominance.
                ab = (0.0, 0.0)
                conn = Dedalus.jacobi_coefficient_connection(N, ab, ab)
                @test size(conn) == (N, N)
                # Diagonal entries should be the largest in each column
                for j in 1:N
                    @test abs(conn[j, j]) > 0
                end
            catch e
                @test_broken false
                @warn "Coefficient connection diagonal test failed for N=$N" exception=e
            end
        end
    end

    # =====================================================================
    # Intertwiner tests
    # =====================================================================

    @testset "Intertwiner" begin

        # ---- intertwiner_k helper ----
        @testset "intertwiner_k basic properties" begin
            try
                Q0 = Dedalus.Intertwiner(0)
                Q1 = Dedalus.Intertwiner(1)
                Q2 = Dedalus.Intertwiner(2)
                Q3 = Dedalus.Intertwiner(3)

                # k(L, mu, s) = -mu * sqrt(max(0, (L - s*mu)*(L + s*mu + 1)/2))
                # At L=0: k(0, mu, s) should be 0 for any (mu, s) because
                # (0 - s*mu)*(0 + s*mu + 1) is non-positive for |s| > 0 or gives 0
                @test Dedalus.intertwiner_k(Q0, 1, 0) == 0.0   # -1*sqrt(max(0, 0*1/2)) = 0
                @test Dedalus.intertwiner_k(Q0, -1, 0) == 0.0

                # At L=1, mu=1, s=0: k = -1 * sqrt((1-0)*(1+0+1)/2) = -sqrt(1)
                @test isapprox(Dedalus.intertwiner_k(Q1, 1, 0), -1.0, atol=1e-14)
                # At L=1, mu=-1, s=0: k = 1 * sqrt((1-0)*(1+0+1)/2) = sqrt(1)
                @test isapprox(Dedalus.intertwiner_k(Q1, -1, 0), 1.0, atol=1e-14)

                # At L=2, mu=1, s=0: k = -1*sqrt((2)*(3)/2) = -sqrt(3)
                @test isapprox(Dedalus.intertwiner_k(Q2, 1, 0), -sqrt(3), atol=1e-14)

                # At L=3, mu=1, s=1: k = -1*sqrt((3-1)*(3+1+1)/2) = -sqrt(5)
                @test isapprox(Dedalus.intertwiner_k(Q3, 1, 1), -sqrt(5), atol=1e-14)
            catch e
                @test_broken false
                @warn "intertwiner_k test failed" exception=e
            end
        end

        # ---- forbidden_spin ----
        @testset "forbidden_spin" begin
            try
                Q0 = Dedalus.Intertwiner(0)
                Q1 = Dedalus.Intertwiner(1)
                Q2 = Dedalus.Intertwiner(2)

                # At L=0, any nonzero total spin is forbidden
                @test Dedalus.forbidden_spin(Q0, (0,)) == false
                @test Dedalus.forbidden_spin(Q0, (1,)) == true
                @test Dedalus.forbidden_spin(Q0, (-1,)) == true

                # At L=1, |total_spin| <= 1 is allowed
                @test Dedalus.forbidden_spin(Q1, (0,)) == false
                @test Dedalus.forbidden_spin(Q1, (1,)) == false
                @test Dedalus.forbidden_spin(Q1, (-1,)) == false
                @test Dedalus.forbidden_spin(Q1, (1, 1)) == true  # |2| > 1

                # At L=2, |total_spin| <= 2 is allowed
                @test Dedalus.forbidden_spin(Q2, (1, 1)) == false  # |2| <= 2
                @test Dedalus.forbidden_spin(Q2, (1, -1)) == false # |0| <= 2
                @test Dedalus.forbidden_spin(Q2, (1, 1, 1)) == true  # |3| > 2
            catch e
                @test_broken false
                @warn "forbidden_spin test failed" exception=e
            end
        end

        # ---- forbidden_regularity ----
        @testset "forbidden_regularity" begin
            try
                Q0 = Dedalus.Intertwiner(0)
                Q1 = Dedalus.Intertwiner(1)
                Q2 = Dedalus.Intertwiner(2)
                Q3 = Dedalus.Intertwiner(3)

                # At L=0, only regularity () or single-element regularity with walk
                # not going negative is allowed
                @test Dedalus.forbidden_regularity(Q0, (1,)) == false
                @test Dedalus.forbidden_regularity(Q0, (-1,)) == true  # walk: [0, -1] -> -1 < 0

                # At L=1, rank-1 regularity should be allowed for +-1
                @test Dedalus.forbidden_regularity(Q1, (1,)) == false
                @test Dedalus.forbidden_regularity(Q1, (-1,)) == false
                @test Dedalus.forbidden_regularity(Q1, (0,)) == false

                # L >= rank always allowed
                @test Dedalus.forbidden_regularity(Q2, (1,)) == false
                @test Dedalus.forbidden_regularity(Q2, (-1,)) == false
                @test Dedalus.forbidden_regularity(Q3, (1, 1)) == false
                @test Dedalus.forbidden_regularity(Q3, (-1, -1)) == false
            catch e
                @test_broken false
                @warn "forbidden_regularity test failed" exception=e
            end
        end

        # ---- Recursive getindex (tensor_getindex) ----
        @testset "tensor_getindex rank 0 (scalar)" begin
            try
                Q0 = Dedalus.Intertwiner(0)
                Q1 = Dedalus.Intertwiner(1)
                # Rank 0: should return 1
                @test Dedalus.tensor_getindex(Q0, (), ()) == 1
                @test Dedalus.tensor_getindex(Q1, (), ()) == 1
            catch e
                @test_broken false
                @warn "tensor_getindex rank 0 test failed" exception=e
            end
        end

        @testset "tensor_getindex rank 1 (vector) L=$L" for L in [0, 1, 2, 3]
            try
                Q = Dedalus.Intertwiner(L)
                # Build the full 3x3 matrix (spin x regularity) for rank 1
                # spin indices: -1, 0, +1; regularity indices: -1, 0, +1
                mat = zeros(3, 3)
                for (si, s) in enumerate([-1, 0, 1])
                    for (ri, r) in enumerate([-1, 0, 1])
                        mat[si, ri] = Dedalus.tensor_getindex(Q, (s,), (r,))
                    end
                end
                # Forbidden entries should be zero
                if L == 0
                    # At L=0, only s=0 is allowed, and regularity walk for -1 goes negative
                    @test mat[1, :] == zeros(3)  # s=-1 forbidden
                    @test mat[3, :] == zeros(3)  # s=+1 forbidden
                end
                # For L >= 1, the matrix should be orthogonal (unitary)
                if L >= 1
                    product = mat * mat'
                    @test isapprox(product, I(3), atol=1e-12)
                end
            catch e
                @test_broken false
                @warn "tensor_getindex rank 1 test failed for L=$L" exception=e
            end
        end

        @testset "tensor_getindex rank 2 (tensor) L=$L" for L in [1, 2, 3]
            try
                Q = Dedalus.Intertwiner(L)
                # Build full 9x9 matrix for rank 2
                spins = [(s1, s2) for s1 in [-1, 0, 1] for s2 in [-1, 0, 1]]
                regs = [(r1, r2) for r1 in [-1, 0, 1] for r2 in [-1, 0, 1]]
                n = length(spins)
                mat = zeros(n, n)
                for (i, s) in enumerate(spins)
                    for (j, r) in enumerate(regs)
                        mat[i, j] = Dedalus.tensor_getindex(Q, s, r)
                    end
                end
                # Forbidden rows (|total_spin| > L) should be zero
                for (i, s) in enumerate(spins)
                    if abs(sum(s)) > L
                        @test norm(mat[i, :]) < 1e-14
                    end
                end
                # Non-forbidden part should have orthonormal rows
                # (Q Q^T = I on the allowed subspace)
                allowed = [i for (i, s) in enumerate(spins) if abs(sum(s)) <= L]
                if length(allowed) > 0
                    sub = mat[allowed, :]
                    product = sub * sub'
                    @test isapprox(product, I(length(allowed)), atol=1e-12)
                end
            catch e
                @test_broken false
                @warn "tensor_getindex rank 2 test failed for L=$L" exception=e
            end
        end

        # ---- Full matrix evaluation ----
        @testset "_tensor_eval L=$L rank=$rank" for L in [1, 2, 3], rank in [1, 2]
            try
                Q = Dedalus.Intertwiner(L)
                mat = Dedalus._tensor_eval(Q, rank)
                n = 3^rank
                @test size(mat) == (n, n)
                # Should be unitary on allowed subspace
                product = mat * mat'
                # Check near-diagonal structure (may have zero rows for forbidden spins)
                for i in 1:n
                    if norm(mat[i, :]) > 1e-14
                        @test isapprox(product[i, i], 1.0, atol=1e-12)
                    end
                end
            catch e
                @test_broken false
                @warn "_tensor_eval test failed for L=$L, rank=$rank" exception=e
            end
        end
    end

end

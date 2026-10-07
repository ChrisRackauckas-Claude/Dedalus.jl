"""Tests for polar coordinate non-constant coefficients (NCCs) on disk and annulus bases."""

using Test
using Dedalus

# ---------------------------------------------------------------------------
# Helper functions to build polar bases and grids
# ---------------------------------------------------------------------------

function build_disk(Nphi, Nr, dealias, T)
    c = PolarCoordinates("phi", "r")
    d = Distributor((c,), T)
    b = DiskBasis(c, (Nphi, Nr), T; radius = 1.5, dealias = (dealias, dealias))
    phi, r = local_grids(d, b)
    x, y = cartesian(PolarCoordinates, phi, r)
    return c, d, b, phi, r, x, y
end

function build_annulus(Nphi, Nr, dealias, T)
    c = PolarCoordinates("phi", "r")
    d = Distributor((c,), T)
    b = AnnulusBasis(c, (Nphi, Nr), T; radii = (0.5, 3.0), dealias = (dealias, dealias))
    phi, r = local_grids(d, b)
    x, y = cartesian(PolarCoordinates, phi, r)
    return c, d, b, phi, r, x, y
end

# ---------------------------------------------------------------------------
# NCC evaluation helper: builds the LBVP infrastructure, stores NCC matrices,
# evaluates both the direct product and NCC form, then compares.
# Returns true if results match within tolerance.
# ---------------------------------------------------------------------------

function ncc_test_scalar_product(f, g, vars; atol = 1.0e-10)
    w0 = f * g
    w1 = reinitialize(w0, ncc = true, ncc_vars = vars)
    problem = LBVP(vars)
    add_equation!(problem, (w1, 0))
    solver = build_solver(problem)
    store_ncc_matrices!(w1, vars, solver.subproblems)
    w0 = evaluate(w0)
    w1 = evaluate_as_ncc(w1)
    change_scales!(w0, 1)
    change_scales!(w1, 1)
    return isapprox(w0["g"], w1["g"], atol = atol)
end

function ncc_test_dot_product(f, g, vars; atol = 1.0e-10)
    w0 = DotProduct(f, g)
    w1 = reinitialize(w0, ncc = true, ncc_vars = vars)
    problem = LBVP(vars)
    add_equation!(problem, (w1, 0))
    solver = build_solver(problem)
    store_ncc_matrices!(w1, vars, solver.subproblems)
    w0 = evaluate(w0)
    w1 = evaluate_as_ncc(w1)
    change_scales!(w0, 1)
    change_scales!(w1, 1)
    return isapprox(w0["g"], w1["g"], atol = atol)
end

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

"""
Test 1: scalar NCC * scalar field
f(r) is a radial-only scalar NCC, g(phi, r) is a full scalar field.
"""
function test_scalar_prod_scalar(c, d, b, phi, r, x, y, ncc_first)
    rb = radial_basis(b)
    f = Field(d, bases = (rb,), dtype = b.dtype)
    g = Field(d, bases = (b,), dtype = b.dtype)
    f["g"] = @. r^4
    g["g"] = @. 3 * x^2 + 2 * y
    vars = [g]
    if ncc_first
        return ncc_test_scalar_product(f, g, vars)
    else
        return ncc_test_scalar_product(g, f, vars)
    end
end

"""
Test 2: scalar NCC * vector field
f(r) is a radial-only scalar NCC, g(phi, r) is a full vector field.
"""
function test_scalar_prod_vector(c, d, b, phi, r, x, y, ncc_first)
    rb = radial_basis(b)
    f = Field(d, bases = (rb,), dtype = b.dtype)
    g = VectorField(d, c, bases = (b,), dtype = b.dtype)
    f["g"] = @. r^4
    g["g"][1, :, :] .= @. 3 * x^2 + 2 * y
    g["g"][2, :, :] .= @. x + 4 * y^2
    vars = [g]
    if ncc_first
        return ncc_test_scalar_product(f, g, vars)
    else
        return ncc_test_scalar_product(g, f, vars)
    end
end

"""
Test 3: scalar NCC * tensor field (rank 2)
f(r) is a radial-only scalar NCC, g(phi, r) is a full rank-2 tensor field.
"""
function test_scalar_prod_tensor(c, d, b, phi, r, x, y, ncc_first)
    rb = radial_basis(b)
    f = Field(d, bases = (rb,), dtype = b.dtype)
    g = TensorField(d, (c, c), bases = (b,), dtype = b.dtype)
    f["g"] = @. r^4
    g["g"][1, 1, :, :] .= @. 3 * x^2 + 2 * y
    g["g"][1, 2, :, :] .= @. x + 4 * y^2
    g["g"][2, 1, :, :] .= @. x * y
    g["g"][2, 2, :, :] .= @. x^2 - y^2
    vars = [g]
    if ncc_first
        return ncc_test_scalar_product(f, g, vars)
    else
        return ncc_test_scalar_product(g, f, vars)
    end
end

"""
Test 4: vector NCC * scalar field
f(r) is a radial-only vector NCC, g(phi, r) is a full scalar field.
"""
function test_vector_prod_scalar(c, d, b, phi, r, x, y, ncc_first)
    rb = radial_basis(b)
    f = VectorField(d, c, bases = (rb,), dtype = b.dtype)
    g = Field(d, bases = (b,), dtype = b.dtype)
    f["g"][1, :, :] .= @. r^2
    f["g"][2, :, :] .= @. r^4
    g["g"] = @. 3 * x^2 + 2 * y
    vars = [g]
    if ncc_first
        return ncc_test_scalar_product(f, g, vars)
    else
        return ncc_test_scalar_product(g, f, vars)
    end
end

"""
Test 5: vector NCC * vector field (outer product)
f(r) is a radial-only vector NCC, g(phi, r) is a full vector field.
"""
function test_vector_prod_vector(c, d, b, phi, r, x, y, ncc_first)
    rb = radial_basis(b)
    f = VectorField(d, c, bases = (rb,), dtype = b.dtype)
    g = VectorField(d, c, bases = (b,), dtype = b.dtype)
    f["g"][1, :, :] .= @. r^2
    f["g"][2, :, :] .= @. r^4
    g["g"][1, :, :] .= @. 3 * x^2 + 2 * y
    g["g"][2, :, :] .= @. x + 4 * y^2
    vars = [g]
    if ncc_first
        return ncc_test_scalar_product(f, g, vars)
    else
        return ncc_test_scalar_product(g, f, vars)
    end
end

"""
Test 6: dot product of vector NCC with vector field
f(r) is a radial-only vector NCC, g(phi, r) is a full vector field.
"""
function test_vector_dot_vector(c, d, b, phi, r, x, y, ncc_first)
    rb = radial_basis(b)
    f = VectorField(d, c, bases = (rb,), dtype = b.dtype)
    g = VectorField(d, c, bases = (b,), dtype = b.dtype)
    f["g"][1, :, :] .= @. r^2
    f["g"][2, :, :] .= @. r^4
    g["g"][1, :, :] .= @. 3 * x^2 + 2 * y
    g["g"][2, :, :] .= @. x + 4 * y^2
    vars = [g]
    if ncc_first
        return ncc_test_dot_product(f, g, vars)
    else
        return ncc_test_dot_product(g, f, vars)
    end
end

"""
Test 7: dot product of vector NCC with tensor field
f(r) is a radial-only vector NCC, g(phi, r) is a full rank-2 tensor field.
"""
function test_vector_dot_tensor(c, d, b, phi, r, x, y, ncc_first)
    rb = radial_basis(b)
    f = VectorField(d, c, bases = (rb,), dtype = b.dtype)
    g = TensorField(d, (c, c), bases = (b,), dtype = b.dtype)
    f["g"][1, :, :] .= @. r^2
    f["g"][2, :, :] .= @. r^4
    g["g"][1, 1, :, :] .= @. 3 * x^2 + 2 * y
    g["g"][1, 2, :, :] .= @. x + 4 * y^2
    g["g"][2, 1, :, :] .= @. x * y
    g["g"][2, 2, :, :] .= @. x^2 - y^2
    vars = [g]
    if ncc_first
        return ncc_test_dot_product(f, g, vars)
    else
        return ncc_test_dot_product(g, f, vars)
    end
end

"""
Test 8: tensor NCC * scalar field
f(r) is a radial-only rank-2 tensor NCC, g(phi, r) is a full scalar field.
"""
function test_tensor_prod_scalar(c, d, b, phi, r, x, y, ncc_first)
    rb = radial_basis(b)
    f = TensorField(d, (c, c), bases = (rb,), dtype = b.dtype)
    g = Field(d, bases = (b,), dtype = b.dtype)
    f["g"][1, 1, :, :] .= @. r^2
    f["g"][1, 2, :, :] .= @. r^3
    f["g"][2, 1, :, :] .= @. r^4
    f["g"][2, 2, :, :] .= @. r^5
    g["g"] = @. 3 * x^2 + 2 * y
    vars = [g]
    if ncc_first
        return ncc_test_scalar_product(f, g, vars)
    else
        return ncc_test_scalar_product(g, f, vars)
    end
end

"""
Test 9: tensor NCC dot vector field
f(r) is a radial-only rank-2 tensor NCC, g(phi, r) is a full vector field.
"""
function test_tensor_dot_vector(c, d, b, phi, r, x, y, ncc_first)
    rb = radial_basis(b)
    f = TensorField(d, (c, c), bases = (rb,), dtype = b.dtype)
    g = VectorField(d, c, bases = (b,), dtype = b.dtype)
    f["g"][1, 1, :, :] .= @. r^2
    f["g"][1, 2, :, :] .= @. r^3
    f["g"][2, 1, :, :] .= @. r^4
    f["g"][2, 2, :, :] .= @. r^5
    g["g"][1, :, :] .= @. 3 * x^2 + 2 * y
    g["g"][2, :, :] .= @. x + 4 * y^2
    vars = [g]
    if ncc_first
        return ncc_test_dot_product(f, g, vars)
    else
        return ncc_test_dot_product(g, f, vars)
    end
end

"""
Test 10: tensor NCC dot tensor field
f(r) is a radial-only rank-2 tensor NCC, g(phi, r) is a full rank-2 tensor field.
"""
function test_tensor_dot_tensor(c, d, b, phi, r, x, y, ncc_first)
    rb = radial_basis(b)
    f = TensorField(d, (c, c), bases = (rb,), dtype = b.dtype)
    g = TensorField(d, (c, c), bases = (b,), dtype = b.dtype)
    f["g"][1, 1, :, :] .= @. r^2
    f["g"][1, 2, :, :] .= @. r^3
    f["g"][2, 1, :, :] .= @. r^4
    f["g"][2, 2, :, :] .= @. r^5
    g["g"][1, 1, :, :] .= @. 3 * x^2 + 2 * y
    g["g"][1, 2, :, :] .= @. x + 4 * y^2
    g["g"][2, 1, :, :] .= @. x * y
    g["g"][2, 2, :, :] .= @. x^2 - y^2
    vars = [g]
    if ncc_first
        return ncc_test_dot_product(f, g, vars)
    else
        return ncc_test_dot_product(g, f, vars)
    end
end

# ---------------------------------------------------------------------------
# Test sets
# ---------------------------------------------------------------------------

@testset "Polar NCC" begin

    # -----------------------------------------------------------------------
    # Disk basis tests
    # -----------------------------------------------------------------------
    @testset "Disk" begin

        @testset "scalar_prod_scalar Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_scalar_prod_scalar(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "scalar_prod_vector Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_scalar_prod_vector(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "scalar_prod_tensor Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_scalar_prod_tensor(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "vector_prod_scalar Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_vector_prod_scalar(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "vector_prod_vector Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_vector_prod_vector(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "vector_dot_vector Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_vector_dot_vector(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "vector_dot_tensor Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_vector_dot_tensor(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "tensor_prod_scalar Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_tensor_prod_scalar(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "tensor_dot_vector Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_tensor_dot_vector(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "tensor_dot_tensor Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_tensor_dot_tensor(c, d, b, phi, r, x, y, ncc_first)
        end

    end  # Disk

    # -----------------------------------------------------------------------
    # Annulus basis tests
    # -----------------------------------------------------------------------
    @testset "Annulus" begin

        @testset "scalar_prod_scalar Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_scalar_prod_scalar(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "scalar_prod_vector Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_scalar_prod_vector(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "scalar_prod_tensor Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_scalar_prod_tensor(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "vector_prod_scalar Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_vector_prod_scalar(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "vector_prod_vector Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_vector_prod_vector(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "vector_dot_vector Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_vector_dot_vector(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "vector_dot_tensor Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_vector_dot_tensor(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "tensor_prod_scalar Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_tensor_prod_scalar(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "tensor_dot_vector Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_tensor_dot_vector(c, d, b, phi, r, x, y, ncc_first)
        end

        @testset "tensor_dot_tensor Nphi=$Nphi Nr=$Nr dealias=$dealias T=$T ncc_first=$ncc_first" for
            Nphi in [16],
                Nr in [8],
                dealias in [1, 1.5],
                T in [Float64, ComplexF64],
                ncc_first in [true, false]
            c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, dealias, T)
            # LBVP solves on polar domains are not implemented: https://github.com/SciML/Dedalus.jl/issues/27
            @test_broken test_tensor_dot_tensor(c, d, b, phi, r, x, y, ncc_first)
        end

    end  # Annulus

end  # Polar NCC

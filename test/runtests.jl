using Test
using Dedalus

@testset "Dedalus.jl" begin
    include("test_clenshaw.jl")
    include("test_transforms.jl")
    include("test_fourier_operators.jl")
    include("test_jacobi_operators.jl")
    include("test_cardinal_operators.jl")
    include("test_grid_operators.jl")
    include("test_cartesian_operators.jl")
    include("test_cartesian_ncc.jl")
    include("test_lbvp.jl")
    include("test_evp.jl")
    include("test_ivp.jl")
    include("test_nlbvp.jl")
    include("test_output.jl")
    include("test_cfl.jl")
end

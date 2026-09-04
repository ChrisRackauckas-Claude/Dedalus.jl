# DedalusSphere: include file for the dedalus_sphere library
#
# This file includes all dedalus_sphere modules in dependency order.
# All functions and types are defined in the parent Dedalus module scope
# (not a separate submodule), matching how they were designed and tested.
#
# The dedalus_sphere library provides the spherical geometry algebra system:
# - Lazy Operator/Codomain algebra for composing spectral operators
# - Jacobi polynomial operator framework
# - Spin-weighted spherical harmonics
# - Tensor operator algebra in spin space
# - Zernike radial polynomials for disk domains
# - Shell and annulus domain operators
# - Clenshaw summation for NCC matrix construction

# Layer 1: Base framework (no internal deps)
include("tuple_tools.jl")
include("operators.jl")

# Layer 2: Jacobi operator algebra (depends on operators)
include("jacobi.jl")

# Layer 3: Spin operators (depends on operators, tuple_tools)
include("spin_operators.jl")

# Layer 4: Geometry-specific modules (depend on jacobi, spin_operators)
include("sphere.jl")
include("zernike.jl")
include("shell.jl")
include("annulus.jl")
include("clenshaw.jl")

# Layer 5: High-level wrappers (depend on sphere, spin_operators)
include("sphere_wrapper.jl")
include("ball_wrapper.jl")

module Dedalus

__precompile__(false)

const VERSION = "3.0.5"

# ============================================================
# Standard library imports
# ============================================================
using LinearAlgebra
using SparseArrays
using TOML
using Logging
using Random

# ============================================================
# External package imports
# ============================================================
using FFTW
using SpecialFunctions
using HDF5

# ============================================================
# Layer 1: Tools and utilities (no internal dependencies)
# ============================================================
include("tools/exceptions.jl")
include("tools/cache.jl")
include("tools/config.jl")
include("tools/general.jl")
include("tools/dispatch.jl")
include("tools/linalg.jl")
include("tools/array.jl")
include("tools/jacobi.jl")
include("tools/clenshaw.jl")
include("tools/parsing.jl")
include("tools/logging.jl")
include("tools/parallel.jl")
include("tools/random_arrays.jl")

# ============================================================
# Layer 2: Core type foundations
# ============================================================
include("core/coords.jl")
include("core/domain.jl")

# ============================================================
# Layer 3: Field and expression infrastructure
# ============================================================
include("core/field.jl")
include("core/future.jl")
include("core/system.jl")

# ============================================================
# Layer 4: Operators and arithmetic
# ============================================================
include("core/arithmetic.jl")
include("core/operators.jl")

# ============================================================
# Layer 4b: dedalus_sphere library (used by bases and operators)
# ============================================================
include("libraries/dedalus_sphere/DedalusSphere.jl")

# ============================================================
# Layer 5: Spectral bases and transforms
# ============================================================
include("core/basis.jl")
include("core/transforms.jl")

# ============================================================
# Layer 6: Distribution and matrix solvers
# ============================================================
include("core/distributor.jl")
include("libraries/matsolvers.jl")

# ============================================================
# Layer 7: Problem setup and solver pipeline
# ============================================================
include("core/problems.jl")
include("core/subsystems.jl")
include("core/solvers.jl")
include("core/timesteppers.jl")

# ============================================================
# Layer 8: Evaluation and output
# ============================================================
include("core/evaluator.jl")

# ============================================================
# Layer 9: Extras
# ============================================================
include("extras/flow_tools.jl")
include("extras/plot_tools.jl")
include("extras/quick_domains.jl")

# ============================================================
# Public API exports (matching Python dedalus/public.py)
# ============================================================

# Coordinates
export Coordinate, CartesianCoordinates, S2Coordinates,
       PolarCoordinates, SphericalCoordinates, DirectProduct

# Distributor
export Distributor

# Fields
export Field, ScalarField, VectorField, TensorField, LockedField

# Operators (constructor functions)
export differentiate, interpolate, integrate, average, lift,
       gradient, divergence, curl, laplacian,
       trace_op, transpose_components, skew,
       radial_component, angular_component, azimuthal_component,
       grid_op, coeff_op, time_derivative, convert_operand

# Basis types (1D from Milestone 1, 2D curvilinear from Milestone 2)
export Jacobi, ChebyshevT, ChebyshevU, Legendre, Ultraspherical,
       ComplexFourier, RealFourier, Fourier,
       DiskBasis, AnnulusBasis, SphereBasis

# Problems
export IVP, EVP, LBVP, NLBVP

# Solvers
export LinearBoundaryValueSolver, EigenvalueSolver,
       InitialValueSolver, NonlinearBoundaryValueSolver

# Timesteppers
export CNAB1, CNAB2, MCNAB2, CNLF2, SBDF1, SBDF2, SBDF3, SBDF4,
       RK111, RK222, RK443, RKSMR, RKGFY

# Extras – flow tools
export CFL, GlobalFlowProperty

# Extras – plot tools
export FieldWrapper, DimWrapper,
       PlotBox, xbox, ybox,
       PlotFrame, bottom_left, top_right,
       MultiFigure, subfigure_axes,
       get_1d_vertices, quad_mesh, pad_limits, get_plane,
       plot_bot, plot_bot_2d, plot_bot_3d

# Extras – quick domains
export quick_fourier, quick_chebyshev,
       quick_fourier_2d, quick_fourier_3d,
       quick_channel_2d, quick_channel_3d

# Warn if threading is not disabled
if get(ENV, "OMP_NUM_THREADS", "") != "1"
    @warn "Threading has not been disabled. This may degrade Dedalus performance.\n" *
          "We strongly suggest setting OMP_NUM_THREADS=1."
end

end # module Dedalus

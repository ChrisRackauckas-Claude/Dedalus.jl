"""
    Spectral transform classes for Dedalus.jl

Julia translation of `dedalus/core/transforms.py`. Provides the 1D separable
transform types that convert fields between grid (physical) space and
coefficient (spectral) space along a single axis.

## Type hierarchy

    Transform                          (abstract)
    +-- SeparableTransform             (abstract -- 1D axis-independent)
    |   +-- SeparableMatrixTransform   (abstract -- via dense mat-mul)
    |   |   +-- JacobiMMT
    |   |   +-- ComplexFourierMMT
    |   |   +-- RealFourierMMT
    |   +-- JacobiTransform            (abstract -- Jacobi polynomial base)
    |   |   +-- JacobiMMT              (also inherits SeparableMatrixTransform)
    |   +-- ComplexFourierTransform     (abstract)
    |   |   +-- ComplexFourierMMT
    |   |   +-- ComplexFFT             (abstract -- FFT-based)
    |   |   |   +-- FFTWComplexFFT
    |   +-- RealFourierTransform       (abstract)
    |   |   +-- RealFourierMMT
    |   |   +-- RealFFT                (abstract -- via real-to-complex FFT)
    |   |   |   +-- FFTWRealFFT

## Key translation choices

- Python class hierarchy with multiple inheritance --> Julia abstract types +
  composition.  Where Python uses MI (e.g. `JacobiMMT(JacobiTransform,
  SeparableMatrixTransform)`), Julia types hold the relevant fields directly
  and dispatch is handled by method specialization.
- `@CachedAttribute` --> lazily-computed fields via `CachedAttribute` from
  `tools/cache.jl`.
- `@CachedMethod` --> `CachedMethod` from `tools/cache.jl`.
- `numpy` --> Julia arrays; column-major layout is natural.
- `axis` parameters are **1-based** (Julia convention).
- FFTW integration uses `FFTW.jl` (`plan_fft`, `plan_rfft`, etc.) instead of
  the custom Python `fftw_wrappers`.
- `register_transform` decorator --> `register_transform!` function that
  populates a basis-class dictionary.
- `axslice` is imported from `tools/array.jl`.

## Forward references

`AbstractBasis` is assumed to be defined in `core/domain.jl`.  Concrete basis
types (`Jacobi`, `RealFourier`, `ComplexFourier`) are not yet created; we
reference them via forward-declared abstract types where needed.
"""

using LinearAlgebra
using FFTW

# Pull helpers from the tools modules (assumed already included in the parent module).
# axslice, apply_dense, apply_sparse, apply_matrix are from tools/array.jl
# build_grid, build_weights, build_polynomials, conversion_matrix from tools/jacobi.jl
# CachedAttribute, get_cached!, CachedMethod from tools/cache.jl
# get_config, get_config_bool from tools/config.jl

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

"""FFTW planning rigor flag, read from `[transforms-fftw] PLANNING_RIGOR`."""
function _get_fftw_rigor()
    try
        return get_config("transforms-fftw", "PLANNING_RIGOR")
    catch
        return "measure"
    end
end

"""Whether to dealias before spectral conversion (Jacobi transforms)."""
function _get_dealias_before_converting()
    try
        return get_config_bool("transforms", "DEALIAS_BEFORE_CONVERTING")
    catch
        return true
    end
end

"""Map rigor name strings to FFTW flag constants."""
const FFTW_RIGOR_FLAGS = Dict{String, UInt32}(
    "estimate"   => FFTW.ESTIMATE,
    "measure"    => FFTW.MEASURE,
    "patient"    => FFTW.PATIENT,
    "exhaustive" => FFTW.EXHAUSTIVE,
)

"""
    fftw_flag(rigor::AbstractString) -> UInt32

Convert a planning-rigor name to the corresponding `FFTW` flag constant.
Defaults to `FFTW.MEASURE` for unrecognized strings.
"""
function fftw_flag(rigor::AbstractString)
    return get(FFTW_RIGOR_FLAGS, lowercase(rigor), FFTW.MEASURE)
end

# ---------------------------------------------------------------------------
# Transform registration
# ---------------------------------------------------------------------------

"""
    register_transform!(basis_transforms::Dict, name::Symbol, cls)

Register a transform constructor `cls` under `name` in the given
`basis_transforms` dictionary.  This is the Julia analogue of the Python
`@register_transform(basis, name)` decorator.

# Example
```julia
register_transform!(Jacobi_transforms, :matrix, JacobiMMT)
```
"""
function register_transform!(basis_transforms::Dict{Symbol, Any}, name::Symbol, cls)
    basis_transforms[name] = cls
    return cls
end

# ============================================================================
# Abstract type hierarchy
# ============================================================================

"""
    Transform

Abstract base type for all spectral transforms.
"""
abstract type Transform end

"""
    SeparableTransform <: Transform

Abstract base for transforms that operate along a single axis, independent
of all other dimensions.  Concrete subtypes must implement:

- `forward!(t, gdata, cdata, axis)` -- grid-to-coefficient transform.
- `backward!(t, cdata, gdata, axis)` -- coefficient-to-grid transform.
"""
abstract type SeparableTransform <: Transform end

"""
    forward!(t::SeparableTransform, gdata, cdata, axis)

Apply the forward (grid --> coefficient) transform along `axis`.
"""
function forward!(::SeparableTransform, gdata, cdata, axis)
    error("forward! not implemented for this transform type")
end

"""
    backward!(t::SeparableTransform, cdata, gdata, axis)

Apply the backward (coefficient --> grid) transform along `axis`.
"""
function backward!(::SeparableTransform, cdata, gdata, axis)
    error("backward! not implemented for this transform type")
end

# ============================================================================
# JacobiTransform
# ============================================================================

"""
    JacobiTransform <: SeparableTransform

Abstract base for Jacobi polynomial transforms.

# Fields (expected in concrete subtypes)
- `N::Int` -- grid size along transform dimension.
- `M::Int` -- coefficient size along transform dimension.
- `a`, `b` -- Jacobi parameters for the polynomials.
- `a0`, `b0` -- Jacobi parameters for the quadrature grid.
- `dealias_before_converting::Bool`
"""
abstract type JacobiTransform <: SeparableTransform end

# ============================================================================
# JacobiMMT -- matrix multiply transform for Jacobi polynomials
# ============================================================================

"""
    JacobiMMT <: JacobiTransform

Jacobi polynomial transform via explicit matrix multiplication (MMT).

Forward: coefficient = forward_matrix * grid_data  (quadrature + conversion)
Backward: grid_data  = backward_matrix * coefficients  (polynomial evaluation)

The transform matrices are lazily computed and cached.
"""
mutable struct JacobiMMT <: JacobiTransform
    N::Int
    M::Int
    a::Float64
    b::Float64
    a0::Float64
    b0::Float64
    dealias_before_converting::Bool
    _forward_matrix::CachedAttribute
    _backward_matrix::CachedAttribute
end

"""
    JacobiMMT(grid_size, coeff_size, a, b, a0, b0;
              dealias_before_converting=nothing)

Construct a Jacobi matrix-multiply transform.

# Arguments
- `grid_size::Int` -- grid size (N) along transform dimension.
- `coeff_size::Int` -- coefficient size (M) along transform dimension.
- `a, b` -- Jacobi parameters for the target polynomial family.
- `a0, b0` -- Jacobi parameters for the quadrature grid.
- `dealias_before_converting` -- if `nothing`, read from config.
"""
function JacobiMMT(grid_size::Int, coeff_size::Int, a, b, a0, b0;
                   dealias_before_converting::Union{Bool, Nothing}=nothing)
    if dealias_before_converting === nothing
        dealias_before_converting = _get_dealias_before_converting()
    end
    N = grid_size
    M = coeff_size

    # -- lazy forward matrix --------------------------------------------------
    fwd_cache = CachedAttribute{Matrix{Float64}}(function()
        # Gauss quadrature with base (a0, b0) polynomials
        base_grid = build_grid(N, a0, b0)
        base_polynomials = build_polynomials(max(M, N), a0, b0, base_grid)
        base_weights = build_weights(N, a0, b0)
        base_transform = base_polynomials .* base_weights'
        # Zero higher coefficients for transforms with grid_size < coeff_size
        if size(base_transform, 1) > N
            base_transform[N+1:end, :] .= 0
        end
        if dealias_before_converting
            # Truncate to specified coeff_size
            base_transform = base_transform[1:M, :]
        end
        # Spectral conversion
        if a == a0 && b == b0
            forward_mat = base_transform
        else
            nrows = size(base_transform, 1)
            conversion = conversion_matrix(nrows, a0, b0, a, b)
            forward_mat = conversion * base_transform
        end
        if !dealias_before_converting
            # Truncate to specified coeff_size
            forward_mat = forward_mat[1:M, :]
        end
        return Matrix{Float64}(forward_mat)
    end)

    # -- lazy backward matrix -------------------------------------------------
    bwd_cache = CachedAttribute{Matrix{Float64}}(function()
        # Construct polynomials on the base grid
        base_grid = build_grid(N, a0, b0)
        polynomials = build_polynomials(M, a, b, base_grid)
        # Zero higher polynomials for transforms with grid_size < coeff_size
        if size(polynomials, 1) > N
            polynomials[N+1:end, :] .= 0
        end
        # Transpose for Julia's column-major layout
        return Matrix{Float64}(polynomials')
    end)

    return JacobiMMT(N, M, Float64(a), Float64(b), Float64(a0), Float64(b0),
                     dealias_before_converting, fwd_cache, bwd_cache)
end

"""Retrieve the (lazily built) forward transform matrix."""
forward_matrix(t::JacobiMMT) = get_cached!(t._forward_matrix)

"""Retrieve the (lazily built) backward transform matrix."""
backward_matrix(t::JacobiMMT) = get_cached!(t._backward_matrix)

function forward!(t::JacobiMMT, gdata::AbstractArray, cdata::AbstractArray, axis::Int)
    apply_dense(forward_matrix(t), gdata, axis; out=cdata)
    return cdata
end

function backward!(t::JacobiMMT, cdata::AbstractArray, gdata::AbstractArray, axis::Int)
    apply_dense(backward_matrix(t), cdata, axis; out=gdata)
    return gdata
end

# ============================================================================
# ComplexFourierTransform
# ============================================================================

"""
    ComplexFourierTransform <: SeparableTransform

Abstract base for complex-to-complex Fourier transforms.

Stores grid size `N`, coefficient size `M`, and derived wavenumber limits:
- `KN = (N - 1) / 2` -- max fully-resolved mode on the grid.
- `KM = (M - 1) / 2` -- max retained mode in coefficient space.
- `Kmax = min(KN, KM)` -- max wavenumber used in transforms.

A unit-amplitude normalization is used.
"""
abstract type ComplexFourierTransform <: SeparableTransform end

"""
    wavenumbers(t::ComplexFourierTransform) -> Vector{Int}

One-dimensional global wavenumber array for complex Fourier modes.
Ordering: `[0, 1, ..., KM, KM+1, -KM, ..., -1]` (standard FFT order for `M`
points, with odd `M` having a Nyquist mode that gets zeroed).
"""
function wavenumbers(N::Int, M::Int, KM::Int)
    k = collect(0:M-1)
    return @. (k + KM) % M - KM
end

# ============================================================================
# ComplexFourierMMT -- matrix multiply transform for complex Fourier
# ============================================================================

"""
    ComplexFourierMMT <: ComplexFourierTransform

Complex-to-complex Fourier transform via explicit matrix multiplication.
"""
mutable struct ComplexFourierMMT <: ComplexFourierTransform
    N::Int
    M::Int
    KN::Int
    KM::Int
    Kmax::Int
    _forward_matrix::CachedAttribute
    _backward_matrix::CachedAttribute
end

function ComplexFourierMMT(grid_size::Int, coeff_size::Int)
    N = grid_size
    M = coeff_size
    KN = (N - 1) >> 1
    KM = (M - 1) >> 1
    Kmax = min(KN, KM)

    wn = wavenumbers(N, M, KM)

    fwd_cache = CachedAttribute{Matrix{ComplexF64}}(function()
        K = reshape(wn, :, 1)          # column vector (M x 1)
        X = reshape(collect(0:N-1), 1, :)  # row vector (1 x N)
        dX = N / 2 / pi
        quadrature = exp.(-im .* K .* X ./ dX) ./ N
        # Zero Nyquist and higher modes
        mask = abs.(K) .<= Kmax
        quadrature .*= mask
        return Matrix{ComplexF64}(quadrature)
    end)

    bwd_cache = CachedAttribute{Matrix{ComplexF64}}(function()
        K = reshape(wn, 1, :)          # row vector (1 x M)
        X = reshape(collect(0:N-1), :, 1)  # column vector (N x 1)
        dX = N / 2 / pi
        functions = exp.(im .* K .* X ./ dX)
        # Zero Nyquist and higher modes
        mask = abs.(K) .<= Kmax
        functions .*= mask
        return Matrix{ComplexF64}(functions)
    end)

    return ComplexFourierMMT(N, M, KN, KM, Kmax, fwd_cache, bwd_cache)
end

forward_matrix(t::ComplexFourierMMT) = get_cached!(t._forward_matrix)
backward_matrix(t::ComplexFourierMMT) = get_cached!(t._backward_matrix)

function forward!(t::ComplexFourierMMT, gdata::AbstractArray, cdata::AbstractArray, axis::Int)
    apply_dense(forward_matrix(t), gdata, axis; out=cdata)
    return cdata
end

function backward!(t::ComplexFourierMMT, cdata::AbstractArray, gdata::AbstractArray, axis::Int)
    apply_dense(backward_matrix(t), cdata, axis; out=gdata)
    return gdata
end

# ============================================================================
# ComplexFFT -- FFT-based complex Fourier transforms
# ============================================================================

"""
    resize_coeffs_complex!(data_in, data_out, axis, Kmax, rescale)

Resize and rescale coefficients in standard FFT format by intermediate
padding/truncation.  Copies positive and negative frequency bands from
`data_in` to `data_out`, zeroing intermediate (unresolved) frequencies.
`rescale` is applied multiplicatively; pass `nothing` for a pure copy.
"""
function resize_coeffs_complex!(data_in::AbstractArray, data_out::AbstractArray,
                                axis::Int, Kmax::Int,
                                rescale::Union{Nothing, Real})
    nd = ndims(data_in)
    if Kmax == 0
        posfreq = axslice(axis, 1, 1)
        if rescale === nothing
            selectdim(data_out, axis, 1) .= selectdim(data_in, axis, 1)
        else
            selectdim(data_out, axis, 1) .= selectdim(data_in, axis, 1) .* rescale
        end
        nout = size(data_out, axis)
        if nout > 1
            for i in 2:nout
                selectdim(data_out, axis, i) .= 0
            end
        end
    else
        nout = size(data_out, axis)
        # Positive frequencies: indices 1 through Kmax+1
        for i in 1:Kmax+1
            if rescale === nothing
                selectdim(data_out, axis, i) .= selectdim(data_in, axis, i)
            else
                selectdim(data_out, axis, i) .= selectdim(data_in, axis, i) .* rescale
            end
        end
        # Zero intermediate (bad) frequencies
        neg_start = nout - Kmax + 1  # first negative freq index in output
        if Kmax + 2 <= neg_start - 1
            for i in (Kmax+2):(neg_start-1)
                selectdim(data_out, axis, i) .= 0
            end
        end
        # Negative frequencies: last Kmax indices
        nin = size(data_in, axis)
        for j in 0:Kmax-1
            src_idx = nin - Kmax + 1 + j
            dst_idx = nout - Kmax + 1 + j
            if rescale === nothing
                selectdim(data_out, axis, dst_idx) .= selectdim(data_in, axis, src_idx)
            else
                selectdim(data_out, axis, dst_idx) .= selectdim(data_in, axis, src_idx) .* rescale
            end
        end
    end
    return data_out
end

# ============================================================================
# FFTWComplexFFT
# ============================================================================

"""
    FFTWComplexFFT <: ComplexFourierTransform

Complex-to-complex FFT using FFTW.jl.

Plans are built lazily on first use and cached for reuse.
"""
mutable struct FFTWComplexFFT <: ComplexFourierTransform
    N::Int
    M::Int
    KN::Int
    KM::Int
    Kmax::Int
    rigor::UInt32
    _plan_cache::Dict{Tuple{NTuple, Int}, Any}  # (shape, axis) -> (fwd_plan, bwd_plan)
end

function FFTWComplexFFT(grid_size::Int, coeff_size::Int;
                        rigor::Union{Nothing, AbstractString}=nothing)
    N = grid_size
    M = coeff_size
    KN = (N - 1) >> 1
    KM = (M - 1) >> 1
    Kmax = min(KN, KM)
    if rigor === nothing
        rigor = _get_fftw_rigor()
    end
    flag = fftw_flag(rigor)
    return FFTWComplexFFT(N, M, KN, KM, Kmax, flag, Dict{Tuple{NTuple, Int}, Any}())
end

"""
    _get_plans(t::FFTWComplexFFT, gshape, axis) -> (fwd_plan, bwd_plan)

Build or retrieve cached FFTW plans for a complex FFT along `axis` with
array shape `gshape`.
"""
function _get_plans(t::FFTWComplexFFT, gshape::Tuple, axis::Int)
    key = (gshape, axis)
    if haskey(t._plan_cache, key)
        return t._plan_cache[key]
    end
    # Create a temporary array to build the plan
    tmp = zeros(ComplexF64, gshape)
    fwd = plan_fft(tmp, axis; flags=t.rigor)
    bwd = plan_ifft(tmp, axis; flags=t.rigor)
    plans = (fwd, bwd)
    t._plan_cache[key] = plans
    return plans
end

function forward!(t::FFTWComplexFFT, gdata::AbstractArray{ComplexF64},
                  cdata::AbstractArray{ComplexF64}, axis::Int)
    fwd, _ = _get_plans(t, size(gdata), axis)
    temp = fwd * gdata  # FFTW forward (unnormalized)
    # Resize and rescale for unit-amplitude normalization
    resize_coeffs_complex!(temp, cdata, axis, t.Kmax, 1.0 / t.N)
    return cdata
end

function backward!(t::FFTWComplexFFT, cdata::AbstractArray{ComplexF64},
                   gdata::AbstractArray{ComplexF64}, axis::Int)
    _, bwd = _get_plans(t, size(gdata), axis)
    # Resize without rescaling into a temporary buffer sized like gdata
    temp = similar(gdata)
    resize_coeffs_complex!(cdata, temp, axis, t.Kmax, nothing)
    # Execute FFTW backward (ifft includes 1/N normalization)
    # FFTW.jl's plan_ifft already includes the 1/N factor, but we need
    # unit-amplitude convention: backward should multiply by N then ifft.
    # Since plan_ifft divides by N, and we want sum(ck * exp(...)), we
    # need to pre-multiply by N so that ifft(N * temp) = sum.
    temp .*= t.N
    result = bwd * temp
    copyto!(gdata, result)
    return gdata
end

# ============================================================================
# RealFourierTransform
# ============================================================================

"""
    RealFourierTransform <: SeparableTransform

Abstract base for real-to-real Fourier transforms.

Coefficient ordering uses interleaved cosine and minus-sine components:
`[a(0), b(0), a(1), b(1), ..., a(KM), b(KM)]`
where `b(0) = 0` always.

Unit-amplitude normalization is used.
"""
abstract type RealFourierTransform <: SeparableTransform end

"""
    wavenumbers_real(KM::Int) -> Vector{Int}

One-dimensional global wavenumber array for real Fourier modes.
Each wavenumber `k` appears twice (cosine and minus-sine components).
"""
function wavenumbers_real(KM::Int)
    return repeat(collect(0:KM); inner=2)
end

# ============================================================================
# RealFourierMMT
# ============================================================================

"""
    RealFourierMMT <: RealFourierTransform

Real-to-real Fourier transform via explicit matrix multiplication.
"""
mutable struct RealFourierMMT <: RealFourierTransform
    N::Int
    M::Int
    KN::Int
    KM::Int
    Kmax::Int
    _forward_matrix::CachedAttribute
    _backward_matrix::CachedAttribute
end

function RealFourierMMT(grid_size::Int, coeff_size::Int)
    N = grid_size
    M = coeff_size
    KN = (N - 1) >> 1
    KM = (M - 1) >> 1
    Kmax = min(KN, KM)

    wn = wavenumbers_real(KM)
    M_eff = max(2, M)  # account for sin and cos parts of m=0

    fwd_cache = CachedAttribute{Matrix{Float64}}(function()
        K = reshape(wn[1:2:end], :, 1)      # unique wavenumbers, column
        X = reshape(collect(0:N-1), 1, :)    # row
        dX = N / 2 / pi
        quadrature = zeros(Float64, M_eff, N)
        # Cosine rows (even indices: 1, 3, 5, ...)
        for i in 1:2:M_eff
            k_idx = (i + 1) >> 1  # 1-based index into K
            k_val = wn[min(i, length(wn))]
            quadrature[i, :] .= (2 / N) .* cos.(k_val .* collect(0:N-1) ./ dX)
        end
        # Minus-sine rows (even indices: 2, 4, 6, ...)
        for i in 2:2:M_eff
            k_idx = i >> 1  # 1-based index into unique K
            k_val = wn[min(i, length(wn))]
            quadrature[i, :] .= -(2 / N) .* sin.(k_val .* collect(0:N-1) ./ dX)
        end
        # k=0 cos row is just 1/N
        quadrature[1, :] .= 1 / N
        # Zero Nyquist and higher modes
        for i in 1:M_eff
            if wn[min(i, length(wn))] > Kmax
                quadrature[i, :] .= 0
            end
        end
        return Matrix{Float64}(quadrature[1:M, :])
    end)

    bwd_cache = CachedAttribute{Matrix{Float64}}(function()
        K = reshape(wn[1:2:end], 1, :)       # unique wavenumbers, row
        X = reshape(collect(0:N-1), :, 1)     # column
        dX = N / 2 / pi
        functions = zeros(Float64, N, M_eff)
        # Cosine columns (odd indices)
        for j in 1:2:M_eff
            k_val = wn[min(j, length(wn))]
            functions[:, j] .= cos.(k_val .* collect(0:N-1) ./ dX)
        end
        # Minus-sine columns (even indices)
        for j in 2:2:M_eff
            k_val = wn[min(j, length(wn))]
            functions[:, j] .= -sin.(k_val .* collect(0:N-1) ./ dX)
        end
        # Zero Nyquist and higher modes
        for j in 1:M_eff
            if wn[min(j, length(wn))] > Kmax
                functions[:, j] .= 0
            end
        end
        return Matrix{Float64}(functions[:, 1:M])
    end)

    return RealFourierMMT(N, M, KN, KM, Kmax, fwd_cache, bwd_cache)
end

forward_matrix(t::RealFourierMMT) = get_cached!(t._forward_matrix)
backward_matrix(t::RealFourierMMT) = get_cached!(t._backward_matrix)

function forward!(t::RealFourierMMT, gdata::AbstractArray, cdata::AbstractArray, axis::Int)
    apply_dense(forward_matrix(t), gdata, axis; out=cdata)
    return cdata
end

function backward!(t::RealFourierMMT, cdata::AbstractArray, gdata::AbstractArray, axis::Int)
    apply_dense(backward_matrix(t), cdata, axis; out=gdata)
    return gdata
end

# ============================================================================
# RealFFT -- real-to-real FFT via real-to-complex algorithms
# ============================================================================

"""
    unpack_rescale_real!(temp, cdata, axis, Kmax, rescale)

Unpack complex RFFT output `temp` into interleaved real cosine/minus-sine
coefficients in `cdata`, applying `rescale` for unit-amplitude normalization.

Layout of `cdata` along `axis`:
  [a(0), b(0)=0, a(1), b(1), ..., a(KM), b(KM), zeros...]
"""
function unpack_rescale_real!(temp::AbstractArray, cdata::AbstractArray,
                              axis::Int, Kmax::Int, rescale::Real)
    # k = 0 cosine coefficient
    selectdim(cdata, axis, 1) .= real.(selectdim(temp, axis, 1)) .* rescale
    # k = 0 minus-sine is always zero
    if size(cdata, axis) >= 2
        selectdim(cdata, axis, 2) .= 0
    end
    # 1 <= k <= Kmax: unpack real and imaginary parts
    for k in 1:Kmax
        cos_idx = 2 * k + 1   # 1-based index for a(k)
        msin_idx = 2 * k + 2  # 1-based index for b(k)
        temp_idx = k + 1      # 1-based index into rfft output
        if cos_idx <= size(cdata, axis)
            selectdim(cdata, axis, cos_idx) .= real.(selectdim(temp, axis, temp_idx)) .* (2 * rescale)
        end
        if msin_idx <= size(cdata, axis)
            selectdim(cdata, axis, msin_idx) .= imag.(selectdim(temp, axis, temp_idx)) .* (2 * rescale)
        end
    end
    # Zero k > Kmax data
    first_zero = 2 * (Kmax + 1) + 1
    ncoeff = size(cdata, axis)
    if first_zero <= ncoeff
        for i in first_zero:ncoeff
            selectdim(cdata, axis, i) .= 0
        end
    end
    return cdata
end

"""
    repack_rescale_real!(cdata, temp, axis, Kmax, rescale)

Repack interleaved real cosine/minus-sine coefficients from `cdata` into
complex coefficients in `temp` for an inverse RFFT, applying `rescale`.
`rescale` may be `nothing` for no scaling.
"""
function repack_rescale_real!(cdata::AbstractArray, temp::AbstractArray,
                              axis::Int, Kmax::Int,
                              rescale::Union{Nothing, Real})
    # k = 0 data
    if rescale === nothing
        selectdim(temp, axis, 1) .= selectdim(cdata, axis, 1)
    else
        selectdim(temp, axis, 1) .= selectdim(cdata, axis, 1) .* rescale
    end
    # 1 <= k <= Kmax: repack cos and msin into complex
    for k in 1:Kmax
        cos_idx = 2 * k + 1
        msin_idx = 2 * k + 2
        temp_idx = k + 1
        cos_part = selectdim(cdata, axis, cos_idx)
        msin_part = selectdim(cdata, axis, msin_idx)
        if rescale === nothing
            selectdim(temp, axis, temp_idx) .= complex.(cos_part .* 0.5, msin_part .* 0.5)
        else
            selectdim(temp, axis, temp_idx) .= complex.(cos_part .* (rescale / 2), msin_part .* (rescale / 2))
        end
    end
    # Zero k > Kmax in temp
    ntemp = size(temp, axis)
    if Kmax + 2 <= ntemp
        for i in (Kmax+2):ntemp
            selectdim(temp, axis, i) .= 0
        end
    end
    return temp
end

# ============================================================================
# FFTWRealFFT
# ============================================================================

"""
    FFTWRealFFT <: RealFourierTransform

Real-to-real FFT using FFTW.jl's real-to-complex (`plan_rfft`) and
complex-to-real (`plan_irfft`) transforms.

Plans are built lazily on first use and cached.
"""
mutable struct FFTWRealFFT <: RealFourierTransform
    N::Int
    M::Int
    KN::Int
    KM::Int
    Kmax::Int
    rigor::UInt32
    _plan_cache::Dict{Tuple, Any}
end

function FFTWRealFFT(grid_size::Int, coeff_size::Int;
                     rigor::Union{Nothing, AbstractString}=nothing)
    N = grid_size
    M = coeff_size
    KN = (N - 1) >> 1
    KM = (M - 1) >> 1
    Kmax = min(KN, KM)
    if rigor === nothing
        rigor = _get_fftw_rigor()
    end
    flag = fftw_flag(rigor)
    return FFTWRealFFT(N, M, KN, KM, Kmax, flag, Dict{Tuple, Any}())
end

"""
    _get_plans(t::FFTWRealFFT, gshape, axis) -> (rfft_plan, irfft_plan)

Build or retrieve cached FFTW plans for a real FFT along `axis` with
array shape `gshape`.
"""
function _get_plans(t::FFTWRealFFT, gshape::Tuple, axis::Int)
    key = (gshape, axis)
    if haskey(t._plan_cache, key)
        return t._plan_cache[key]
    end
    # Create temporary arrays for planning
    tmp_real = zeros(Float64, gshape)
    fwd = plan_rfft(tmp_real, axis; flags=t.rigor)
    # For irfft we need the complex-shaped array
    cshape = collect(gshape)
    cshape[axis] = (gshape[axis] >> 1) + 1  # N/2 + 1
    tmp_complex = zeros(ComplexF64, Tuple(cshape))
    bwd = plan_irfft(tmp_complex, gshape[axis], axis; flags=t.rigor)
    plans = (fwd, bwd)
    t._plan_cache[key] = plans
    return plans
end

function forward!(t::FFTWRealFFT, gdata::AbstractArray{Float64},
                  cdata::AbstractArray{Float64}, axis::Int)
    fwd, _ = _get_plans(t, size(gdata), axis)
    # Execute real FFT (gives complex output of size N/2+1 along axis)
    temp = fwd * gdata
    # Unpack from complex form and rescale
    unpack_rescale_real!(temp, cdata, axis, t.Kmax, 1.0 / t.N)
    return cdata
end

function backward!(t::FFTWRealFFT, cdata::AbstractArray{Float64},
                   gdata::AbstractArray{Float64}, axis::Int)
    _, bwd = _get_plans(t, size(gdata), axis)
    N = t.N
    # Allocate complex temporary (N/2+1 along axis)
    cshape = collect(size(gdata))
    cshape[axis] = (N >> 1) + 1
    temp = zeros(ComplexF64, Tuple(cshape))
    # Repack into complex form and rescale
    # irfft includes the 1/N factor, so we pre-multiply by N
    repack_rescale_real!(cdata, temp, axis, t.Kmax, Float64(N))
    # Execute inverse real FFT
    result = bwd * temp
    copyto!(gdata, result)
    return gdata
end

# ============================================================================
# CosineTransform (stub for completeness)
# ============================================================================

"""
    CosineTransform <: SeparableTransform

Abstract base for cosine transforms (DCT-based). Coefficient ordering is
simply `[a(0), a(1), ..., a(KM)]`.
"""
abstract type CosineTransform <: SeparableTransform end

# ============================================================================
# Utility: reduced_view helpers
# ============================================================================

"""
    reduced_view_3(data::AbstractArray, axis::Int)

Reshape `data` into a 3D array `(N0, N1, N2)` where `N1 = size(data, axis)`,
`N0 = prod(size(data)[1:axis-1])`, `N2 = prod(size(data)[axis+1:end])`.
"""
function reduced_view_3(data::AbstractArray, axis::Int)
    s = size(data)
    N0 = prod(s[1:axis-1]; init=1)
    N1 = s[axis]
    N2 = prod(s[axis+1:end]; init=1)
    return reshape(data, N0, N1, N2)
end

"""
    reduced_view_4(data::AbstractArray, axis::Int)

Reshape `data` into a 4D array `(N0, N1, N2, N3)` where `N1 = size(data, axis)`,
`N2 = size(data, axis+1)`, `N0 = prod(dims before axis)`,
`N3 = prod(dims after axis+1)`.
"""
function reduced_view_4(data::AbstractArray, axis::Int)
    s = size(data)
    N0 = prod(s[1:axis-1]; init=1)
    N1 = s[axis]
    N2 = s[axis+1]
    N3 = prod(s[axis+2:end]; init=1)
    return reshape(data, N0, N1, N2, N3)
end

"""
    reduced_view_5(data::AbstractArray, axis::Int)

Reshape `data` into a 5D array.
"""
function reduced_view_5(data::AbstractArray, axis::Int)
    s = size(data)
    N0 = prod(s[1:axis-1]; init=1)
    N1 = s[axis]
    N2 = s[axis+1]
    N3 = s[axis+2]
    N4 = prod(s[axis+3:end]; init=1)
    return reshape(data, N0, N1, N2, N3, N4)
end

# ============================================================================
# transform_plan concrete methods for each basis type
# (Defined here because transform types are not yet available in basis.jl)
# ============================================================================

"""
    transform_plan(b::JacobiBasis, dist, grid_size)

Build or retrieve cached JacobiMMT transform plan for the given grid size.
"""
function transform_plan(b::JacobiBasis, dist, grid_size)
    cache_key = grid_size
    cached = get(b._transform_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    plan = JacobiMMT(grid_size, b._size, b.a, b.b, b.a0, b.b0)
    b._transform_cache[cache_key] = plan
    return plan
end

"""
    transform_plan(b::ComplexFourierBasis, dist, grid_size)

Build or retrieve cached FFTWComplexFFT transform plan for the given grid size.
"""
function transform_plan(b::ComplexFourierBasis, dist, grid_size)
    cache_key = grid_size
    cached = get(b._transform_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    plan = FFTWComplexFFT(grid_size, b._size)
    b._transform_cache[cache_key] = plan
    return plan
end

"""
    transform_plan(b::RealFourierBasis, dist, grid_size)

Build or retrieve cached FFTWRealFFT transform plan for the given grid size.
"""
function transform_plan(b::RealFourierBasis, dist, grid_size)
    cache_key = grid_size
    cached = get(b._transform_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    plan = FFTWRealFFT(grid_size, b._size)
    b._transform_cache[cache_key] = plan
    return plan
end

# ============================================================================
# subspace_matrix dispatch methods (operator + basis type)
# (Defined here because basis types are not available in operators.jl)
# ============================================================================

function subspace_matrix(op::Differentiate, layout)
    basis = op.input_basis
    if basis === nothing
        return sparse(I, 1, 1)
    end
    if basis isa JacobiBasis
        return differentiate_jacobi_matrix(basis, op.output_basis)
    elseif basis isa ComplexFourierBasis
        N = basis._size
        wn = native_wavenumbers(basis)
        diag_vals = ComplexF64[1im * k / basis.COV.stretch for k in wn]
        return sparse(Diagonal(diag_vals))
    elseif basis isa RealFourierBasis
        N = basis._size
        wn = native_wavenumbers(basis)
        mat = spzeros(Float64, N, N)
        for k_idx in 1:2:N
            k_val = wn[k_idx] / basis.COV.stretch
            if k_idx + 1 <= N
                mat[k_idx, k_idx+1] = -k_val
                mat[k_idx+1, k_idx] = k_val
            end
        end
        return mat
    else
        n = basis_size(basis)
        return sparse(I, n, n)
    end
end

function subspace_matrix(op::Convert, layout)
    input_basis = op.input_basis
    output_basis = op.output_basis
    if input_basis === nothing && output_basis !== nothing
        if output_basis isa JacobiBasis
            unit_amplitude = 1.0 / output_basis.constant_mode_value
            N = output_basis._size
            col = zeros(N)
            col[1] = unit_amplitude
            return sparse(reshape(col, :, 1))
        else
            N = output_basis isa IntervalBasis ? output_basis._size : 1
            return sparse(I, N, 1)
        end
    elseif input_basis !== nothing && output_basis !== nothing
        if input_basis isa JacobiBasis && output_basis isa JacobiBasis
            return convert_jacobi_matrix(input_basis, output_basis)
        else
            N = basis_size(input_basis)
            return sparse(I, N, N)
        end
    else
        return sparse(I, 1, 1)
    end
end

function subspace_matrix(op::Interpolate, layout)
    basis = op.input_basis
    if basis === nothing
        return sparse(I, 1, 1)
    end
    if basis isa JacobiBasis
        return sparse(interpolate_jacobi_matrix(basis, op.position))
    elseif basis isa ComplexFourierBasis
        return sparse(interpolate_complex_fourier_matrix(basis, op.position))
    elseif basis isa RealFourierBasis
        return sparse(interpolate_real_fourier_matrix(basis, op.position))
    elseif basis isa CardinalBasis && op.position isa Integer
        return sparse(interpolate_cardinal_matrix(basis, op.position))
    else
        n = basis_size(basis)
        return sparse(I, 1, n)
    end
end

function subspace_matrix(op::Integrate, layout)
    basis = op.input_basis
    if basis === nothing
        return sparse(I, 1, 1)
    end
    if basis isa JacobiBasis
        return sparse(integrate_jacobi_matrix(basis))
    elseif basis isa CardinalBasis
        return sparse(integrate_cardinal_matrix(basis))
    else
        n = basis_size(basis)
        return sparse(ones(1, n))
    end
end

function subspace_matrix(op::Average, layout)
    basis = op.input_basis
    if basis === nothing
        return sparse(I, 1, 1)
    end
    if basis isa JacobiBasis
        return sparse(average_jacobi_matrix(basis))
    elseif basis isa CardinalBasis
        return sparse(average_cardinal_matrix(basis))
    else
        n = basis_size(basis)
        return sparse(fill(1.0 / n, 1, n))
    end
end

function subspace_matrix(op::Lift, layout)
    basis = op.output_basis
    if basis === nothing
        return sparse(I, 1, 1)
    end
    N = basis_size(basis)
    mode = N + op.n + 1  # convert negative index to 1-based
    if op.input_basis === nothing
        col = spzeros(N, 1)
        if 1 <= mode <= N
            col[mode, 1] = 1.0
        end
        return col
    else
        n_in = basis_size(op.input_basis)
        mat = spzeros(N, n_in)
        if 1 <= mode <= N
            mat[mode, 1:min(n_in, 1)] .= 1.0
        end
        return mat
    end
end

# ============================================================================
# Exports
# ============================================================================

const FourierTransform = Union{ComplexFourierTransform, RealFourierTransform}

export Transform,
       SeparableTransform,
       JacobiTransform,
       JacobiMMT,
       FourierTransform,
       ComplexFourierTransform,
       ComplexFourierMMT,
       FFTWComplexFFT,
       RealFourierTransform,
       RealFourierMMT,
       FFTWRealFFT,
       CosineTransform,
       forward!,
       backward!,
       forward_matrix,
       backward_matrix,
       wavenumbers,
       wavenumbers_real,
       register_transform!,
       fftw_flag,
       resize_coeffs_complex!,
       unpack_rescale_real!,
       repack_rescale_real!,
       reduced_view_3,
       reduced_view_4,
       reduced_view_5

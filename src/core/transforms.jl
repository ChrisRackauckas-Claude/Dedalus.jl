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
function _get_fftw_rigor()::String
    try
        return string(get_config("transforms-fftw", "PLANNING_RIGOR"))
    catch
        return "measure"
    end
end

"""Whether to dealias before spectral conversion (Jacobi transforms)."""
function _get_dealias_before_converting()::Bool
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
            @inbounds for i in 2:nout
                selectdim(data_out, axis, i) .= 0
            end
        end
    else
        nout = size(data_out, axis)
        # Positive frequencies: indices 1 through Kmax+1 (Kmax = min(KN, KM) <= nout-1)
        @inbounds for i in 1:Kmax+1
            if rescale === nothing
                selectdim(data_out, axis, i) .= selectdim(data_in, axis, i)
            else
                selectdim(data_out, axis, i) .= selectdim(data_in, axis, i) .* rescale
            end
        end
        # Zero intermediate (bad) frequencies
        neg_start = nout - Kmax + 1  # first negative freq index in output
        if Kmax + 2 <= neg_start - 1
            @inbounds for i in (Kmax+2):(neg_start-1)
                selectdim(data_out, axis, i) .= 0
            end
        end
        # Negative frequencies: last Kmax indices
        nin = size(data_in, axis)
        @inbounds for j in 0:Kmax-1
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
    _plan_cache::Dict{Tuple, Any}  # (shape, axis) -> (fwd_plan, bwd_plan, fwd_buf, bwd_buf)
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
    return FFTWComplexFFT(N, M, KN, KM, Kmax, flag, Dict{Tuple, Any}())
end

"""
    _get_plans(t::FFTWComplexFFT, gshape, axis) -> (fwd_plan, bwd_plan, fwd_buf, bwd_buf)

Build or retrieve cached FFTW plans and working buffers for a complex FFT
along `axis` with array shape `gshape`.  Uses in-place plans (`plan_fft!`,
`plan_bfft!`) to avoid per-call allocations.
"""
function _get_plans(t::FFTWComplexFFT, gshape::Tuple, axis::Int)
    key = (gshape, axis)
    if haskey(t._plan_cache, key)
        return t._plan_cache[key]
    end
    # Create working buffers and in-place plans
    fwd_buf = zeros(ComplexF64, gshape)
    bwd_buf = zeros(ComplexF64, gshape)
    fwd = plan_fft!(fwd_buf, axis; flags=t.rigor)
    bwd = plan_bfft!(bwd_buf, axis; flags=t.rigor)
    entry = (fwd, bwd, fwd_buf, bwd_buf)
    t._plan_cache[key] = entry
    return entry
end

function forward!(t::FFTWComplexFFT, gdata::AbstractArray{ComplexF64},
                  cdata::AbstractArray{ComplexF64}, axis::Int)
    fwd, _, fwd_buf, _ = _get_plans(t, size(gdata), axis)
    # Copy input into the plan buffer and apply in-place FFT
    copyto!(fwd_buf, gdata)
    fwd * fwd_buf  # in-place forward FFT (unnormalized)
    # Resize and rescale for unit-amplitude normalization
    resize_coeffs_complex!(fwd_buf, cdata, axis, t.Kmax, 1.0 / t.N)
    return cdata
end

function backward!(t::FFTWComplexFFT, cdata::AbstractArray{ComplexF64},
                   gdata::AbstractArray{ComplexF64}, axis::Int)
    _, bwd, _, bwd_buf = _get_plans(t, size(gdata), axis)
    # Resize without rescaling into the cached buffer
    resize_coeffs_complex!(cdata, bwd_buf, axis, t.Kmax, nothing)
    # Unit-amplitude convention: backward should produce sum(ck * exp(...)).
    # plan_bfft! computes the unnormalized backward FFT (no 1/N factor),
    # which is exactly the sum we want -- no pre-multiplication needed.
    bwd * bwd_buf  # in-place backward FFT (unnormalized)
    copyto!(gdata, bwd_buf)
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
    @inbounds for k in 1:Kmax
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
        @inbounds for i in first_zero:ncoeff
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
    ncdata = size(cdata, axis)
    @inbounds for k in 1:Kmax
        cos_idx = 2 * k + 1
        msin_idx = 2 * k + 2
        if cos_idx > ncdata || msin_idx > ncdata
            break
        end
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
        @inbounds for i in (Kmax+2):ntemp
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
    _plan_cache::Dict{Tuple, Any}  # (shape, axis) -> (fwd_plan, bwd_plan, fwd_buf, bwd_cbuf, bwd_rbuf)
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
    _get_plans(t::FFTWRealFFT, gshape, axis) -> (rfft_plan, irfft_plan, fwd_buf, bwd_cbuf, bwd_rbuf)

Build or retrieve cached FFTW plans and working buffers for a real FFT
along `axis` with array shape `gshape`.
"""
function _get_plans(t::FFTWRealFFT, gshape::Tuple, axis::Int)
    key = (gshape, axis)
    if haskey(t._plan_cache, key)
        return t._plan_cache[key]
    end
    # Create working buffers and plans
    # Forward: real input buffer, complex output allocated by plan
    fwd_buf = zeros(Float64, gshape)
    fwd = plan_rfft(fwd_buf, axis; flags=t.rigor)
    # Backward: complex input buffer, real output buffer
    cshape = collect(gshape)
    cshape[axis] = (gshape[axis] >> 1) + 1  # N/2 + 1
    bwd_cbuf = zeros(ComplexF64, Tuple(cshape))
    bwd_rbuf = zeros(Float64, gshape)
    bwd = plan_irfft(bwd_cbuf, gshape[axis], axis; flags=t.rigor)
    entry = (fwd, bwd, fwd_buf, bwd_cbuf, bwd_rbuf)
    t._plan_cache[key] = entry
    return entry
end

function forward!(t::FFTWRealFFT, gdata::AbstractArray{Float64},
                  cdata::AbstractArray{Float64}, axis::Int)
    fwd, _, fwd_buf, _, _ = _get_plans(t, size(gdata), axis)
    # Copy input into the plan buffer and execute real FFT
    copyto!(fwd_buf, gdata)
    temp = fwd * fwd_buf  # rfft returns a new complex array (FFTW requirement)
    # Unpack from complex form and rescale
    unpack_rescale_real!(temp, cdata, axis, t.Kmax, 1.0 / t.N)
    return cdata
end

function backward!(t::FFTWRealFFT, cdata::AbstractArray{Float64},
                   gdata::AbstractArray{Float64}, axis::Int)
    _, bwd, _, bwd_cbuf, bwd_rbuf = _get_plans(t, size(gdata), axis)
    N = t.N
    # Repack into complex form using the cached buffer and rescale
    # irfft includes the 1/N factor, so we pre-multiply by N
    bwd_cbuf .= 0
    repack_rescale_real!(cdata, bwd_cbuf, axis, t.Kmax, Float64(N))
    # Execute inverse real FFT into preallocated buffer
    mul!(bwd_rbuf, bwd, bwd_cbuf)
    copyto!(gdata, bwd_rbuf)
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
        error("subspace_matrix(::Differentiate) not implemented for basis type $(typeof(basis))")
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
            error("subspace_matrix(::Convert) not implemented for input_basis $(typeof(input_basis)) and output_basis $(typeof(output_basis))")
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
# NonSeparableTransform
# ============================================================================

"""
    NonSeparableTransform <: Transform

Abstract base for transforms that couple multiple axes (e.g. colatitude
transforms on the sphere that depend on azimuthal mode number *m*).

Concrete subtypes must implement:

- `forward_reduced!(t, gdata, cdata)` -- forward transform on 4D reduced arrays.
- `backward_reduced!(t, cdata, gdata)` -- backward transform on 4D reduced arrays.
"""
abstract type NonSeparableTransform <: Transform end

"""
    forward!(t::NonSeparableTransform, gdata, cdata, axis)

Apply the forward (grid --> coefficient) non-separable transform.
Reduces the input to a 4D view centred on `axis` and `axis+1`, then
delegates to `forward_reduced!`.
"""
function forward!(t::NonSeparableTransform, gdata::AbstractArray,
                  cdata::AbstractArray, axis::Int)
    gdata4 = reduced_view_4(gdata, axis)
    cdata4 = reduced_view_4(cdata, axis)
    forward_reduced!(t, gdata4, cdata4)
    return cdata
end

"""
    backward!(t::NonSeparableTransform, cdata, gdata, axis)

Apply the backward (coefficient --> grid) non-separable transform.
Reduces the input to a 4D view centred on `axis` and `axis+1`, then
delegates to `backward_reduced!`.
"""
function backward!(t::NonSeparableTransform, cdata::AbstractArray,
                   gdata::AbstractArray, axis::Int)
    cdata4 = reduced_view_4(cdata, axis)
    gdata4 = reduced_view_4(gdata, axis)
    backward_reduced!(t, cdata4, gdata4)
    return gdata
end

# ============================================================================
# SWSHColatitudeTransform
# ============================================================================

"""
    SWSHColatitudeTransform <: NonSeparableTransform

Spin-weighted spherical harmonic (SWSH) colatitude transform.

Operates per azimuthal mode number *m*, applying a dense matrix multiply
along the colatitude axis to convert between grid values and spectral
coefficients.

# Fields
- `Ntheta::Int` -- number of colatitude grid points.
- `Lmax::Int` -- maximum spherical-harmonic degree.
- `m_maps` -- tuple of `(m, mg_slice, mc_slice, ell_slice)` entries.
- `s::Int` -- spin weight.
- `_cache::Dict{Symbol,Any}` -- lazily-computed matrix cache.
"""
mutable struct SWSHColatitudeTransform <: NonSeparableTransform
    Ntheta::Int
    Lmax::Int
    m_maps::Any   # tuple of (m, mg_slice, mc_slice, ell_slice) entries
    s::Int
    _cache::Dict{Symbol,Any}
end

"""
    SWSHColatitudeTransform(Ntheta, Lmax, m_maps, s)

Construct a SWSH colatitude transform.
"""
function SWSHColatitudeTransform(Ntheta::Int, Lmax::Int, m_maps, s::Int)
    return SWSHColatitudeTransform(Ntheta, Lmax, m_maps, s, Dict{Symbol,Any}())
end

"""
    _quadrature(t::SWSHColatitudeTransform)

Cached Gauss quadrature `(cos_grid, weights)` for the sphere.
"""
function _quadrature(t::SWSHColatitudeTransform)
    cached = get(t._cache, :quadrature, nothing)
    if cached !== nothing
        return cached
    end
    result = sphere_quadrature(t.Ntheta - 1)
    t._cache[:quadrature] = result
    return result
end

"""
    _forward_SWSH_matrices(t::SWSHColatitudeTransform)

Cached dictionary of forward transform matrices, keyed by azimuthal
mode number *m*.  Each matrix has shape `(Lmax+1-|m|, Ntheta)` and
includes quadrature weights so that the transform is:

    coefficients = matrix * grid_values
"""
function _forward_SWSH_matrices(t::SWSHColatitudeTransform)
    cached = get(t._cache, :forward_SWSH_matrices, nothing)
    if cached !== nothing
        return cached
    end
    cos_grid, weights = _quadrature(t)
    Lmax = t.Lmax
    m_matrices = Dict{Int, Union{Nothing, Matrix{Float64}}}()
    for (m, _, _, _) in t.m_maps
        if haskey(m_matrices, m)
            continue
        end
        if m > Lmax
            # Don't make matrices for m's that will be dropped after transform
            m_matrices[m] = nothing
        else
            Y = sphere_harmonics(Lmax, m, t.s, cos_grid)  # shape (Lmax-Lmin+1, Ntheta)
            # Pad to shape (Lmax+1-|m|, Ntheta) so transforms don't depend on Lmin
            Lmin = max(abs(m), abs(t.s))
            Yfull = zeros(Float64, Lmax + 1 - abs(m), t.Ntheta)
            # Y rows map to ell = Lmin:Lmax, which in padded array start at row Lmin-|m|+1
            pad_start = Lmin - abs(m) + 1  # 1-based
            n_Y_rows = size(Y, 1)
            Yfull[pad_start:pad_start + n_Y_rows - 1, :] .= Float64.(Y .* weights')
            # Zero higher coefficients than can be correctly computed with base Gauss quadrature
            max_valid = t.Ntheta - abs(m)  # 1-based count of valid rows
            if max_valid + 1 <= size(Yfull, 1)
                Yfull[max_valid + 1:end, :] .= 0.0
            end
            m_matrices[m] = copy(Yfull)
        end
    end
    t._cache[:forward_SWSH_matrices] = m_matrices
    return m_matrices
end

"""
    _backward_SWSH_matrices(t::SWSHColatitudeTransform)

Cached dictionary of backward transform matrices, keyed by azimuthal
mode number *m*.  Each matrix has shape `(Ntheta, Lmax+1-|m|)`:

    grid_values = matrix * coefficients
"""
function _backward_SWSH_matrices(t::SWSHColatitudeTransform)
    cached = get(t._cache, :backward_SWSH_matrices, nothing)
    if cached !== nothing
        return cached
    end
    cos_grid, weights = _quadrature(t)
    Lmax = t.Lmax
    m_matrices = Dict{Int, Union{Nothing, Matrix{Float64}}}()
    for (m, _, _, _) in t.m_maps
        if haskey(m_matrices, m)
            continue
        end
        if m > Lmax
            m_matrices[m] = nothing
        else
            Y = sphere_harmonics(Lmax, m, t.s, cos_grid)  # shape (Lmax-Lmin+1, Ntheta)
            # Pad to shape (Ntheta, Lmax+1-|m|) so transforms don't depend on Lmin
            Lmin = max(abs(m), abs(t.s))
            Yfull = zeros(Float64, t.Ntheta, Lmax + 1 - abs(m))
            pad_start = Lmin - abs(m) + 1  # 1-based
            n_Y_rows = size(Y, 1)
            Yfull[:, pad_start:pad_start + n_Y_rows - 1] .= Float64.(Y')
            # Zero higher coefficients than can be correctly computed with base Gauss quadrature
            max_valid = t.Ntheta - abs(m)
            if max_valid + 1 <= size(Yfull, 2)
                Yfull[:, max_valid + 1:end] .= 0.0
            end
            m_matrices[m] = copy(Yfull)
        end
    end
    t._cache[:backward_SWSH_matrices] = m_matrices
    return m_matrices
end

"""
    forward_reduced!(t::SWSHColatitudeTransform, gdata, cdata)

Forward SWSH colatitude transform on 4D reduced arrays.
For each `(m, mg_slice, mc_slice, ell_slice)` in `m_maps`, applies the
forward matrix along axis 3 (the colatitude axis in the reduced view).
"""
function forward_reduced!(t::SWSHColatitudeTransform,
                          gdata::AbstractArray, cdata::AbstractArray)
    m_matrices = _forward_SWSH_matrices(t)
    Lmax = t.Lmax
    for (m, mg_slice, mc_slice, ell_slice) in t.m_maps
        # Skip transforms when |m| > Lmax
        if abs(m) <= Lmax
            grm = @view gdata[:, mg_slice, :, :]
            crm = @view cdata[:, mc_slice, ell_slice, :]
            apply_matrix(m_matrices[m], grm, 2; out=crm)
        end
    end
    return nothing
end

"""
    backward_reduced!(t::SWSHColatitudeTransform, cdata, gdata)

Backward SWSH colatitude transform on 4D reduced arrays.
"""
function backward_reduced!(t::SWSHColatitudeTransform,
                           cdata::AbstractArray, gdata::AbstractArray)
    m_matrices = _backward_SWSH_matrices(t)
    Lmax = t.Lmax
    for (m, mg_slice, mc_slice, ell_slice) in t.m_maps
        if abs(m) > Lmax
            # Write zeros because they'll be used by the inverse azimuthal transform
            gdata[:, mg_slice, :, :] .= 0
        else
            grm = @view gdata[:, mg_slice, :, :]
            crm = @view cdata[:, mc_slice, ell_slice, :]
            apply_matrix(m_matrices[m], crm, 2; out=grm)
        end
    end
    return nothing
end

# ============================================================================
# DiskRadialTransform
# ============================================================================

"""
    DiskRadialTransform <: NonSeparableTransform

Disk radial transform using Zernike polynomials.

Operates per azimuthal mode number *m*, applying a dense matrix multiply
along the radial axis to convert between grid values and spectral
coefficients.

# Fields
- `Nphi::Int` -- number of azimuthal grid points.
- `Nmax::Int` -- maximum radial polynomial degree.
- `N2g::Int` -- radial grid size (grid_shape along radial axis).
- `N2c::Int` -- radial coefficient size (Nmax + 1).
- `m_maps` -- tuple of `(m, mg_slice, mc_slice, n_slice)` entries.
- `s::Int` -- spin weight.
- `k::Int` -- regularity parameter.
- `alpha::Float64` -- Jacobi parameter for the radial quadrature.
- `dtype::DataType` -- element type.
- `dealias_before_converting::Bool` -- whether to truncate before spectral conversion.
- `_cache::Dict{Symbol,Any}` -- lazily-computed matrix cache.
"""
mutable struct DiskRadialTransform <: NonSeparableTransform
    Nphi::Int
    Nmax::Int
    N2g::Int
    N2c::Int
    m_maps::Any   # tuple of (m, mg_slice, mc_slice, n_slice) entries
    s::Int
    k::Int
    alpha::Float64
    dtype::DataType
    dealias_before_converting::Bool
    _cache::Dict{Symbol,Any}
end

"""
    DiskRadialTransform(grid_shape, basis_shape, axis, m_maps, s, k, alpha;
                        dtype=ComplexF64, dealias_before_converting=nothing)

Construct a disk radial transform.

# Arguments
- `grid_shape` -- shape of the grid-space array.
- `basis_shape` -- `(Nphi, Nmax+1)`.
- `axis` -- 1-based radial axis index.
- `m_maps` -- tuple of `(m, mg_slice, mc_slice, n_slice)`.
- `s` -- spin weight.
- `k` -- regularity parameter.
- `alpha` -- Jacobi parameter.
- `dtype` -- element type (default `ComplexF64`).
- `dealias_before_converting` -- if `nothing`, read from config.
"""
function DiskRadialTransform(grid_shape, basis_shape, axis::Int, m_maps, s::Int,
                             k::Int, alpha;
                             dtype::DataType=ComplexF64,
                             dealias_before_converting::Union{Bool, Nothing}=nothing)
    Nphi = basis_shape[1]
    Nmax = basis_shape[2] - 1
    N2g = grid_shape[axis]
    N2c = Nmax + 1
    if dealias_before_converting === nothing
        dealias_before_converting = _get_dealias_before_converting()
    end
    return DiskRadialTransform(Nphi, Nmax, N2g, N2c, m_maps, s, k, Float64(alpha),
                               dtype, dealias_before_converting, Dict{Symbol,Any}())
end

"""
    _quadrature(t::DiskRadialTransform)

Cached Gauss quadrature `(z_grid, weights)` for the disk.
"""
function _quadrature(t::DiskRadialTransform)
    cached = get(t._cache, :quadrature, nothing)
    if cached !== nothing
        return cached
    end
    result = zernike_quadrature(2, t.N2g; k=Int(t.alpha))
    t._cache[:quadrature] = result
    return result
end

"""
    _forward_matrices(t::DiskRadialTransform)

Cached dictionary of forward transform matrices, keyed by azimuthal
mode number *m*.  Each matrix maps grid values to spectral coefficients,
incorporating quadrature weights and optional spectral conversion.
"""
function _forward_matrices(t::DiskRadialTransform)
    cached = get(t._cache, :forward_matrices, nothing)
    if cached !== nothing
        return cached
    end
    z_grid, weights = _quadrature(t)
    m_list = Tuple(entry[1] for entry in t.m_maps)
    m_matrices = Dict{Int, Matrix{Float64}}()
    for m in m_list
        if !haskey(m_matrices, m)
            # Gauss quadrature with base (k=0) polynomials
            Nmin = zernike_min_degree(abs(m))
            Nc = max(max(t.N2g, t.N2c) - Nmin, 0)
            W = zernike_polynomials(2, Nc, t.alpha, abs(m + t.s), z_grid)  # shape (Nc, Ng)
            W = W .* weights'
            # Zero higher coefficients than can be correctly computed with base Gauss quadrature
            dN = abs(m + t.s) ÷ 2
            zero_start = max(t.N2g - dN, 0) + 1  # 1-based
            if zero_start <= size(W, 1)
                W[zero_start:end, :] .= 0.0
            end
            if t.dealias_before_converting
                # Truncate to specified coeff_size
                trunc_rows = max(t.N2c - Nmin, 0)
                W = W[1:trunc_rows, :]
            end
            # Spectral conversion
            if t.k > 0
                conversion_op = zernike_operator(2, "E")(+1)^t.k
                conv_matrix = conversion_op(size(W, 1), t.alpha, abs(m + t.s))
                # conv_matrix may be InfiniteCSC or sparse; convert to dense for matmul
                W = Matrix{Float64}(sparse(conv_matrix)) * W
            end
            if !t.dealias_before_converting
                # Truncate to specified coeff_size
                trunc_rows = max(t.N2c - Nmin, 0)
                W = W[1:trunc_rows, :]
            end
            m_matrices[m] = Matrix{Float64}(W)
        end
    end
    t._cache[:forward_matrices] = m_matrices
    return m_matrices
end

"""
    _backward_matrices(t::DiskRadialTransform)

Cached dictionary of backward transform matrices, keyed by azimuthal
mode number *m*.  Each matrix maps spectral coefficients to grid values.
"""
function _backward_matrices(t::DiskRadialTransform)
    cached = get(t._cache, :backward_matrices, nothing)
    if cached !== nothing
        return cached
    end
    z_grid, weights = _quadrature(t)
    m_list = Tuple(entry[1] for entry in t.m_maps)
    m_matrices = Dict{Int, Matrix{Float64}}()
    for m in m_list
        if !haskey(m_matrices, m)
            # Construct polynomials on the base grid
            Nmin = zernike_min_degree(abs(m))
            Nc = max(t.N2c - Nmin, 0)
            W = zernike_polynomials(2, Nc, t.k + t.alpha, abs(m + t.s), z_grid)
            # Zero higher coefficients than can be correctly computed with base Gauss quadrature
            dN = abs(m + t.s) ÷ 2
            zero_start = max(t.N2g - dN, 0) + 1  # 1-based
            if zero_start <= size(W, 1)
                W[zero_start:end, :] .= 0.0
            end
            # Transpose for Julia's column-major layout
            m_matrices[m] = Matrix{Float64}(W')
        end
    end
    t._cache[:backward_matrices] = m_matrices
    return m_matrices
end

"""
    forward_reduced!(t::DiskRadialTransform, gdata, cdata)

Forward disk radial transform on 4D reduced arrays.
For each `(m, mg_slice, mc_slice, n_slice)` in `m_maps`, applies the
forward matrix along axis 3 (the radial axis in the reduced view).
"""
function forward_reduced!(t::DiskRadialTransform,
                          gdata::AbstractArray, cdata::AbstractArray)
    m_matrices = _forward_matrices(t)
    Nmax = t.Nmax
    for (m, mg_slice, mc_slice, n_slice) in t.m_maps
        # Skip transforms when |m| > 2*Nmax
        if abs(m) <= 2 * Nmax
            grm = @view gdata[:, mg_slice, :, :]
            crm = @view cdata[:, mc_slice, n_slice, :]
            apply_matrix(m_matrices[m], grm, 2; out=crm)
        end
    end
    return nothing
end

"""
    backward_reduced!(t::DiskRadialTransform, cdata, gdata)

Backward disk radial transform on 4D reduced arrays.
"""
function backward_reduced!(t::DiskRadialTransform,
                           cdata::AbstractArray, gdata::AbstractArray)
    m_matrices = _backward_matrices(t)
    Nmax = t.Nmax
    for (m, mg_slice, mc_slice, n_slice) in t.m_maps
        if abs(m) > 2 * Nmax
            # Write zeros because they'll be used by the inverse azimuthal transform
            gdata[:, mg_slice, :, :] .= 0
        else
            grm = @view gdata[:, mg_slice, :, :]
            crm = @view cdata[:, mc_slice, n_slice, :]
            apply_matrix(m_matrices[m], crm, 2; out=grm)
        end
    end
    return nothing
end

# ============================================================================
# transform_plan methods for non-separable bases
# ============================================================================

"""
    transform_plan(b::SphereBasis, dist, Ntheta, s)

Build (or retrieve cached) a SWSHColatitudeTransform plan for the sphere
basis colatitude direction.
"""
function transform_plan(b::SphereBasis, dist, Ntheta::Int, s::Int)
    cache_key = (:transform_plan, objectid(dist), Ntheta, s)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    maps = m_maps(b, dist)
    plan = SWSHColatitudeTransform(Ntheta, b.Lmax, maps, s)
    b._cache[cache_key] = plan
    return plan
end

"""
    transform_plan(b::DiskBasis, dist, grid_shape_val, axis, s)

Build (or retrieve cached) a DiskRadialTransform plan for the disk basis
radial direction.
"""
function transform_plan(b::DiskBasis, dist, grid_shape_val, axis::Int, s::Int)
    cache_key = (:transform_plan, objectid(dist), grid_shape_val, axis, s)
    cached = get(b._cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end
    maps = m_maps(b, dist)
    basis_shape = b.shape
    plan = DiskRadialTransform(grid_shape_val, basis_shape, axis, maps, s,
                               b.k, b.alpha; dtype=b.dtype)
    b._cache[cache_key] = plan
    return plan
end

# ============================================================================
# BallRadialTransform
# ============================================================================

"""
    BallRadialTransform <: NonSeparableTransform

Ball radial transform using Zernike polynomials.

Operates per spherical-harmonic degree *ell*, applying a dense matrix multiply
along the radial axis to convert between grid values and spectral
coefficients.  This is the 3D analogue of `DiskRadialTransform` (dimension 3
instead of 2).

Unlike `DiskRadialTransform`, the ball transform iterates over `ell_maps`
(ell, m_ind, ell_ind) triples and uses `reduced_view_5` to expose the
(azimuthal, colatitude, radial) axes simultaneously.

Regularity enforcement: when `regindex` is non-empty, the `Intertwiner`
spin-operator algebra is used to detect forbidden regularity components
at each *ell*.  Forbidden components are zeroed out.

# Fields
- `N3g::Int` -- radial grid size.
- `N3c::Int` -- radial coefficient size (Nmax + 1).
- `ell_maps` -- tuple of `(ell, m_ind, ell_ind)` entries.
- `regindex` -- regularity index (CartesianIndex or tuple).
- `regtotal::Int` -- total regularity.
- `k::Int` -- regularity parameter.
- `alpha::Float64` -- Jacobi parameter for the radial quadrature.
- `dealias_before_converting::Bool` -- whether to truncate before spectral conversion.
- `_cache::Dict{Symbol,Any}` -- lazily-computed matrix cache.
"""
mutable struct BallRadialTransform <: NonSeparableTransform
    N3g::Int
    N3c::Int
    ell_maps::Any   # tuple of (ell, m_ind, ell_ind) entries
    regindex::Any    # CartesianIndex or tuple -- regularity index
    regtotal::Int
    k::Int
    alpha::Float64
    dealias_before_converting::Bool
    _cache::Dict{Symbol,Any}
end

"""
    BallRadialTransform(grid_shape, coeff_size, axis, ell_maps, regindex,
                        regtotal, k, alpha;
                        dtype=ComplexF64, dealias_before_converting=nothing)

Construct a ball radial transform.

# Arguments
- `grid_shape` -- shape of the grid-space array.
- `coeff_size` -- number of radial coefficients (Nmax + 1).
- `axis` -- 1-based radial axis index.
- `ell_maps` -- tuple of `(ell, m_ind, ell_ind)` entries.
- `regindex` -- regularity index (from `pairs(regularity_classes(...))`).
- `regtotal` -- total regularity value.
- `k` -- regularity parameter.
- `alpha` -- Jacobi parameter.
- `dtype` -- element type (default `ComplexF64`).
- `dealias_before_converting` -- if `nothing`, read from config.
"""
function BallRadialTransform(grid_shape, coeff_size::Int, axis::Int, ell_maps,
                             regindex, regtotal, k, alpha;
                             dtype::DataType=ComplexF64,
                             dealias_before_converting::Union{Bool, Nothing}=nothing)
    N3g = grid_shape[axis]
    N3c = coeff_size
    if dealias_before_converting === nothing
        dealias_before_converting = _get_dealias_before_converting()
    end
    return BallRadialTransform(N3g, N3c, ell_maps, regindex, Int(regtotal),
                               Int(k), Float64(alpha), dealias_before_converting,
                               Dict{Symbol,Any}())
end

"""
    _is_forbidden(t::BallRadialTransform, ell) -> Bool

Check whether the regularity component for the given `ell` is forbidden
by the Intertwiner spin-operator algebra.  Returns `false` for scalar
fields (empty `regindex`).
"""
function _is_forbidden(t::BallRadialTransform, ell)
    # Regularity basis ordering: index 1 -> -1, index 2 -> +1, index 3 -> 0
    Rb = [-1, 1, 0]
    reg_tuple = Tuple(t.regindex)
    if length(reg_tuple) == 0
        return false
    end
    Q = Intertwiner(ell; indexing=(-1, +1, 0))
    reg_vals = Tuple(Rb[r] for r in reg_tuple)
    return forbidden_regularity(Q, reg_vals)
end

"""
    forward!(t::BallRadialTransform, gdata, cdata, axis)

Apply the forward (grid --> coefficient) ball radial transform.
Uses `reduced_view_5` to expose the 5D structure
`(pre, azimuth, colatitude, radial, post)`, then delegates to
`forward_reduced!`.
"""
function forward!(t::BallRadialTransform, gdata::AbstractArray,
                  cdata::AbstractArray, axis::Int)
    # In Python: reduced_view_5(data, axis-2) with 0-based axis
    # In Julia: reduced_view_5(data, axis-2) with 1-based axis
    # The radial axis is at position `axis`, colatitude at `axis-1`,
    # azimuth at `axis-2`, so we reduce starting at `axis-2`.
    gdata5 = reduced_view_5(gdata, axis - 2)
    cdata5 = reduced_view_5(cdata, axis - 2)
    forward_reduced!(t, gdata5, cdata5)
    return cdata
end

"""
    backward!(t::BallRadialTransform, cdata, gdata, axis)

Apply the backward (coefficient --> grid) ball radial transform.
"""
function backward!(t::BallRadialTransform, cdata::AbstractArray,
                   gdata::AbstractArray, axis::Int)
    cdata5 = reduced_view_5(cdata, axis - 2)
    gdata5 = reduced_view_5(gdata, axis - 2)
    backward_reduced!(t, cdata5, gdata5)
    return gdata
end

"""
    _quadrature(t::BallRadialTransform)

Cached Gauss quadrature `(z_grid, weights)` for the ball (dimension 3).
"""
function _quadrature(t::BallRadialTransform)
    cached = get(t._cache, :quadrature, nothing)
    if cached !== nothing
        return cached
    end
    result = zernike_quadrature(3, t.N3g; k=Int(t.alpha))
    t._cache[:quadrature] = result
    return result
end

"""
    _forward_matrices(t::BallRadialTransform)

Cached dictionary of forward transform matrices, keyed by spherical-harmonic
degree *ell*.  Each matrix maps grid values to spectral coefficients,
incorporating quadrature weights and optional spectral conversion.

When the regularity component is forbidden for a given `ell` (as determined
by the `Intertwiner`), the matrix is set to all zeros.
"""
function _forward_matrices(t::BallRadialTransform)
    cached = get(t._cache, :forward_matrices, nothing)
    if cached !== nothing
        return cached
    end
    z_grid, weights = _quadrature(t)
    ell_list = Tuple(entry[1] for entry in t.ell_maps)
    ell_matrices = Dict{Int, Matrix{Float64}}()
    for ell in ell_list
        if !haskey(ell_matrices, ell)
            Nmin = zernike_min_degree(ell)
            if _is_forbidden(t, ell)
                # Forbidden regularity: zero matrix
                ell_matrices[ell] = zeros(Float64, max(t.N3c - Nmin, 0), t.N3g)
            else
                # Gauss quadrature with base (k=0) polynomials
                Nc = max(max(t.N3g, t.N3c) - Nmin, 0)
                W = zernike_polynomials(3, Nc, t.alpha, ell + t.regtotal, z_grid)  # shape (Nc, Ng)
                W = W .* weights'
                # Zero higher coefficients than can be correctly computed with base Gauss quadrature
                dN = fld(ell + t.regtotal, 2)
                zero_start = max(t.N3g - dN, 0) + 1  # 1-based
                if zero_start <= size(W, 1)
                    W[zero_start:end, :] .= 0.0
                end
                if t.dealias_before_converting
                    # Truncate to specified coeff_size
                    trunc_rows = max(t.N3c - Nmin, 0)
                    W = W[1:trunc_rows, :]
                end
                # Spectral conversion
                if t.k > 0
                    conversion_op = zernike_operator(3, "E")(+1)^t.k
                    conv_matrix = conversion_op(size(W, 1), t.alpha, ell + t.regtotal)
                    # conv_matrix may be InfiniteCSC or sparse; convert to dense for matmul
                    W = Matrix{Float64}(sparse(conv_matrix)) * W
                end
                if !t.dealias_before_converting
                    # Truncate to specified coeff_size
                    trunc_rows = max(t.N3c - Nmin, 0)
                    W = W[1:trunc_rows, :]
                end
                ell_matrices[ell] = Matrix{Float64}(W)
            end
        end
    end
    t._cache[:forward_matrices] = ell_matrices
    return ell_matrices
end

"""
    _backward_matrices(t::BallRadialTransform)

Cached dictionary of backward transform matrices, keyed by spherical-harmonic
degree *ell*.  Each matrix maps spectral coefficients to grid values.

When the regularity component is forbidden for a given `ell`, the matrix
is set to all zeros.
"""
function _backward_matrices(t::BallRadialTransform)
    cached = get(t._cache, :backward_matrices, nothing)
    if cached !== nothing
        return cached
    end
    z_grid, weights = _quadrature(t)
    ell_list = Tuple(entry[1] for entry in t.ell_maps)
    ell_matrices = Dict{Int, Matrix{Float64}}()
    for ell in ell_list
        if !haskey(ell_matrices, ell)
            Nmin = zernike_min_degree(ell)
            if _is_forbidden(t, ell)
                # Forbidden regularity: zero matrix
                ell_matrices[ell] = zeros(Float64, t.N3g, max(t.N3c - Nmin, 0))
            else
                # Construct polynomials on the base grid
                Nc = max(t.N3c - Nmin, 0)
                W = zernike_polynomials(3, Nc, t.alpha + t.k, ell + t.regtotal, z_grid)
                # Zero higher coefficients than can be correctly computed with base Gauss quadrature
                dN = fld(ell + t.regtotal, 2)
                zero_start = max(t.N3g - dN, 0) + 1  # 1-based
                if zero_start <= size(W, 1)
                    W[zero_start:end, :] .= 0.0
                end
                # Transpose for Julia's column-major layout
                ell_matrices[ell] = Matrix{Float64}(W')
            end
        end
    end
    t._cache[:backward_matrices] = ell_matrices
    return ell_matrices
end

"""
    forward_reduced!(t::BallRadialTransform, gdata, cdata)

Forward ball radial transform on 5D reduced arrays.
For each `(ell, m_ind, ell_ind)` in `ell_maps`, applies the forward matrix
along axis 4 (the radial axis in the reduced 5D view).

The coefficient data is indexed with `Nmin+1:end` along the radial axis
(1-based) to account for the ell-dependent minimum Zernike degree.
"""
function forward_reduced!(t::BallRadialTransform,
                          gdata::AbstractArray, cdata::AbstractArray)
    ell_matrices = _forward_matrices(t)
    for (ell, m_ind, ell_ind) in t.ell_maps
        Nmin = zernike_min_degree(ell)
        # In the 5D reduced view: (pre, azimuth, colatitude, radial, post)
        # m_ind indexes the azimuth axis (dim 2)
        # ell_ind indexes the colatitude axis (dim 3)
        # radial is dim 4
        grm = @view gdata[:, m_ind, ell_ind, :, :]
        # Nmin+1 for 1-based indexing (Python Nmin: maps to Julia Nmin+1)
        crm = @view cdata[:, m_ind, ell_ind, Nmin+1:end, :]
        apply_matrix(ell_matrices[ell], grm, 2; out=crm)
    end
    return nothing
end

"""
    backward_reduced!(t::BallRadialTransform, cdata, gdata)

Backward ball radial transform on 5D reduced arrays.
"""
function backward_reduced!(t::BallRadialTransform,
                           cdata::AbstractArray, gdata::AbstractArray)
    ell_matrices = _backward_matrices(t)
    for (ell, m_ind, ell_ind) in t.ell_maps
        Nmin = zernike_min_degree(ell)
        grm = @view gdata[:, m_ind, ell_ind, :, :]
        crm = @view cdata[:, m_ind, ell_ind, Nmin+1:end, :]
        apply_matrix(ell_matrices[ell], crm, 2; out=grm)
    end
    return nothing
end

# ============================================================================
# transform_plan methods for BallRadialBasis and BallBasis
# ============================================================================

# Register BallRadialTransform in both _ball_radial_transforms and
# _ball_basis_transforms under the "matrix" key.
_ball_radial_transforms["matrix"] = BallRadialTransform
_ball_basis_transforms["matrix"] = BallRadialTransform

# ============================================================================
# Exports
# ============================================================================

const FourierTransform = Union{ComplexFourierTransform, RealFourierTransform}

export Transform,
       SeparableTransform,
       NonSeparableTransform,
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
       SWSHColatitudeTransform,
       DiskRadialTransform,
       BallRadialTransform,
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

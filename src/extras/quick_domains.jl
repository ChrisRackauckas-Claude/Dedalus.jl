"""
Quick setup of common domains.

Translated from dedalus/extras/quick_domains.py.  Provides convenience
constructor functions that create common domain configurations:

- `quick_fourier`: 1D periodic domain (Fourier basis)
- `quick_chebyshev`: 1D bounded domain (Chebyshev basis)
- `quick_fourier_2d`: 2D doubly-periodic domain
- `quick_fourier_3d`: 3D triply-periodic domain
- `quick_channel_2d`: 2D channel (periodic x, bounded y)
- `quick_channel_3d`: 3D channel (periodic x/y, bounded z)

Each function returns `(coords_or_coord, dist, bases_or_basis)`.
"""

# ---------------------------------------------------------------------------
# 1D domains
# ---------------------------------------------------------------------------

"""
    quick_fourier(N::Int; dealias=3//2, dtype=Float64)
        -> (coord, dist, xbasis)

Create a 1D periodic Fourier domain on [0, 2π].

# Arguments
- `N`:       number of grid points
- `dealias`: dealiasing factor (default `3//2`)
- `dtype`:   element type (default `Float64`); selects `RealFourier` or
             `ComplexFourier` via the `Fourier` factory function
"""
function quick_fourier(N::Int; dealias=3//2, dtype::DataType=Float64)
    coord = Coordinate("x")
    dist = Distributor(coord, dtype)
    xbasis = Fourier(coord, N, (0.0, 2π); dealias=(Float64(dealias),), dtype=dtype)
    return coord, dist, xbasis
end

"""
    quick_chebyshev(N::Int; dealias=3//2, dtype=Float64)
        -> (coord, dist, xbasis)

Create a 1D bounded Chebyshev domain on [-1, 1].

# Arguments
- `N`:       number of grid points
- `dealias`: dealiasing factor (default `3//2`)
- `dtype`:   element type (default `Float64`)
"""
function quick_chebyshev(N::Int; dealias=3//2, dtype::DataType=Float64)
    coord = Coordinate("x")
    dist = Distributor(coord, dtype)
    xbasis = ChebyshevT(coord, N, (-1.0, 1.0); dealias=(Float64(dealias),))
    return coord, dist, xbasis
end

# ---------------------------------------------------------------------------
# 2D domains
# ---------------------------------------------------------------------------

"""
    quick_fourier_2d(N::Int; dealias=3//2, dtype=Float64)
        -> (coords, dist, (xbasis, ybasis))

Create a 2D doubly-periodic Fourier domain on [0, 2π]².

# Arguments
- `N`:       number of grid points per direction
- `dealias`: dealiasing factor (default `3//2`)
- `dtype`:   element type (default `Float64`)
"""
function quick_fourier_2d(N::Int; dealias=3//2, dtype::DataType=Float64)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords, dtype)
    xbasis = Fourier(coords[1], N, (0.0, 2π); dealias=(Float64(dealias),), dtype=dtype)
    ybasis = Fourier(coords[2], N, (0.0, 2π); dealias=(Float64(dealias),), dtype=dtype)
    return coords, dist, (xbasis, ybasis)
end

"""
    quick_fourier_3d(N::Int; dealias=3//2, dtype=Float64)
        -> (coords, dist, (xbasis, ybasis, zbasis))

Create a 3D triply-periodic Fourier domain on [0, 2π]³.

# Arguments
- `N`:       number of grid points per direction
- `dealias`: dealiasing factor (default `3//2`)
- `dtype`:   element type (default `Float64`)
"""
function quick_fourier_3d(N::Int; dealias=3//2, dtype::DataType=Float64)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords, dtype)
    xbasis = Fourier(coords[1], N, (0.0, 2π); dealias=(Float64(dealias),), dtype=dtype)
    ybasis = Fourier(coords[2], N, (0.0, 2π); dealias=(Float64(dealias),), dtype=dtype)
    zbasis = Fourier(coords[3], N, (0.0, 2π); dealias=(Float64(dealias),), dtype=dtype)
    return coords, dist, (xbasis, ybasis, zbasis)
end

"""
    quick_channel_2d(N::Int; dealias=3//2, dtype=Float64)
        -> (coords, dist, (xbasis, ybasis))

Create a 2D channel domain: periodic in x ∈ [0, 2π], bounded in y ∈ [-1, 1].

# Arguments
- `N`:       number of grid points per direction
- `dealias`: dealiasing factor (default `3//2`)
- `dtype`:   element type (default `Float64`)
"""
function quick_channel_2d(N::Int; dealias=3//2, dtype::DataType=Float64)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords, dtype)
    xbasis = Fourier(coords[1], N, (0.0, 2π); dealias=(Float64(dealias),), dtype=dtype)
    ybasis = ChebyshevT(coords[2], N, (-1.0, 1.0); dealias=(Float64(dealias),))
    return coords, dist, (xbasis, ybasis)
end

"""
    quick_channel_3d(N::Int; dealias=3//2, dtype=Float64)
        -> (coords, dist, (xbasis, ybasis, zbasis))

Create a 3D channel domain: periodic in x, y ∈ [0, 2π], bounded in z ∈ [-1, 1].

# Arguments
- `N`:       number of grid points per direction
- `dealias`: dealiasing factor (default `3//2`)
- `dtype`:   element type (default `Float64`)
"""
function quick_channel_3d(N::Int; dealias=3//2, dtype::DataType=Float64)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords, dtype)
    xbasis = Fourier(coords[1], N, (0.0, 2π); dealias=(Float64(dealias),), dtype=dtype)
    ybasis = Fourier(coords[2], N, (0.0, 2π); dealias=(Float64(dealias),), dtype=dtype)
    zbasis = ChebyshevT(coords[3], N, (-1.0, 1.0); dealias=(Float64(dealias),))
    return coords, dist, (xbasis, ybasis, zbasis)
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export quick_fourier, quick_chebyshev,
       quick_fourier_2d, quick_fourier_3d,
       quick_channel_2d, quick_channel_3d

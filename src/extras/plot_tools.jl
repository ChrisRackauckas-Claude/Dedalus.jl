"""
Plotting helper utilities for Dedalus fields.

Translated from dedalus/extras/plot_tools.py.  Provides:

- `FieldWrapper`: wraps a Dedalus Field to mimic an HDF5-dataset interface
- `DimWrapper`: wraps per-axis dimension metadata
- `PlotBox`: 2D-vector-like object for image sizes and offsets
- `PlotFrame`: non-uniform frame around an image
- `MultiFigure`: array-of-images layout computation
- Data extraction utilities: `get_plane`, `quad_mesh`, `get_1d_vertices`,
  `pad_limits`
- Convenience functions: `plot_bot`, `plot_bot_2d`, `plot_bot_3d`

Plotting functions return the computed data, meshes, and layout parameters
rather than producing plots directly — users may use any Julia plotting
backend (Plots.jl, Makie.jl, etc.).
"""

# ---------------------------------------------------------------------------
# FieldWrapper
# ---------------------------------------------------------------------------

"""
    FieldWrapper

Wraps a Dedalus `Field` to mimic an HDF5-dataset-like interface for the
plotting helpers.

# Fields
- `field`: the underlying Dedalus `Field`
- `attrs`: `Dict{String,Any}` with at minimum a `"name"` key
- `dims`:  `Vector{DimWrapper}`, one per data axis
"""
struct FieldWrapper
    field::Field
    attrs::Dict{String,Any}
    dims::Vector{Any}  # Vector{DimWrapper}; use Any to avoid forward-ref issues

    function FieldWrapper(field::Field)
        attrs = Dict{String,Any}("name" => something(field.name, ""))
        ndim = field.domain isa Nothing ? 0 : length(field.domain.bases)
        dims = [DimWrapper(field, ax) for ax in 1:ndim]
        new(field, attrs, dims)
    end
end

"""
    Base.getindex(fw::FieldWrapper, item...)

Index into the wrapped field's data.
"""
Base.getindex(fw::FieldWrapper, item...) = fw.field.data[item...]

"""
    Base.size(fw::FieldWrapper)

Return the shape of the wrapped field's data.
"""
Base.size(fw::FieldWrapper) = size(fw.field.data)

"""
    shape(fw::FieldWrapper) -> Tuple

Return the shape of the wrapped field's data (Python-style helper).
"""
shape(fw::FieldWrapper) = size(fw.field.data)

# ---------------------------------------------------------------------------
# DimWrapper
# ---------------------------------------------------------------------------

"""
    DimWrapper

Wraps per-axis dimension metadata for a Dedalus `Field`, providing label and
grid/mode data access.

# Fields
- `field`: the Dedalus `Field`
- `axis`:  1-based axis index
- `basis`: the basis for this axis
- `dist`:  the field's distributor
"""
struct DimWrapper
    field::Field
    axis::Int
    basis::Any      # Basis type
    dist::Any       # Distributor

    function DimWrapper(field::Field, axis::Int)
        basis = field.domain.bases[axis]
        dist = field.dist
        new(field, axis, basis, dist)
    end
end

"""
    label(dw::DimWrapper) -> String

Return the axis label: coordinate name in grid space, coordinate name + " mode"
in coefficient space.
"""
function label(dw::DimWrapper)
    if dw.field.layout.grid_space[dw.axis]
        return dw.basis.coord.name
    else
        return dw.basis.coord.name * " mode"
    end
end

"""
    Base.getindex(dw::DimWrapper, item...)

In grid space, returns the global grid for the axis at the current scale.
In coefficient space, returns the local modes.
"""
function Base.getindex(dw::DimWrapper, item...)
    if dw.field.layout.grid_space[dw.axis]
        scale = dw.field.scales[dw.axis]
        return vec(global_grid(dw.basis, dw.dist, scale))
    else
        return vec(local_modes(dw.dist, dw.basis))
    end
end

# ---------------------------------------------------------------------------
# PlotBox  (Python: Box)
# ---------------------------------------------------------------------------

"""
    PlotBox

2D-vector-like object for representing image sizes and offsets.

# Fields
- `x::Float64`: width
- `y::Float64`: height
"""
mutable struct PlotBox
    x::Float64
    y::Float64
end

"""
    xbox(b::PlotBox) -> PlotBox

Box retaining only the x (width) component.
"""
xbox(b::PlotBox) = PlotBox(b.x, 0.0)

"""
    ybox(b::PlotBox) -> PlotBox

Box retaining only the y (height) component.
"""
ybox(b::PlotBox) = PlotBox(0.0, b.y)

Base.:+(a::PlotBox, b::PlotBox) = PlotBox(a.x + b.x, a.y + b.y)

Base.:*(a::PlotBox, s::Real) = PlotBox(a.x * s, a.y * s)
Base.:*(s::Real, a::PlotBox) = a * s
Base.:*(a::PlotBox, b::PlotBox) = PlotBox(a.x * b.x, a.y * b.y)

Base.:/(a::PlotBox, s::Real) = PlotBox(a.x / s, a.y / s)
Base.:/(a::PlotBox, b::PlotBox) = PlotBox(a.x / b.x, a.y / b.y)

# ---------------------------------------------------------------------------
# PlotFrame  (Python: Frame)
# ---------------------------------------------------------------------------

"""
    PlotFrame

Non-uniform frame (margin/padding) around an image.

# Fields
- `top::Float64`
- `bottom::Float64`
- `left::Float64`
- `right::Float64`
"""
mutable struct PlotFrame
    top::Float64
    bottom::Float64
    left::Float64
    right::Float64
end

"""
    bottom_left(f::PlotFrame) -> PlotBox

Offset vector to the bottom-left corner of the frame interior.
"""
bottom_left(f::PlotFrame) = PlotBox(f.left, f.bottom)

"""
    top_right(f::PlotFrame) -> PlotBox

Offset vector to the top-right corner of the frame interior.
"""
top_right(f::PlotFrame) = PlotBox(f.right, f.top)

"""
    Base.:+(f::PlotFrame, b::PlotBox) -> PlotBox

A frame added to a box yields the outer box (frame surrounds the inner box).
"""
Base.:+(f::PlotFrame, b::PlotBox) = PlotBox(f.left + b.x + f.right, f.bottom + b.y + f.top)
Base.:+(b::PlotBox, f::PlotFrame) = f + b

Base.:*(f::PlotFrame, s::Real) = PlotFrame(s * f.top, s * f.bottom, s * f.left, s * f.right)
Base.:*(s::Real, f::PlotFrame) = f * s

# ---------------------------------------------------------------------------
# MultiFigure
# ---------------------------------------------------------------------------

"""
    MultiFigure

An array of generic images within a figure, computing layout coordinates.

Rather than creating an actual matplotlib figure, this type performs the
layout arithmetic and stores the resulting geometry so that any plotting
backend can place axes at the correct positions.

# Fields
- `nrows::Int`: number of image rows
- `ncols::Int`: number of image columns
- `image::PlotBox`: image dimensions (after scaling)
- `pad::PlotFrame`: padding around each image (after scaling)
- `margin::PlotFrame`: margin around the entire grid (after scaling)
- `fig::PlotBox`: total figure dimensions
- `figsize::Tuple{Int,Int}`: integer figure size `(width, height)`

# Constructor
    MultiFigure(nrows, ncols, image, pad, margin; scale=1.0)
"""
struct MultiFigure
    nrows::Int
    ncols::Int
    image::PlotBox
    pad::PlotFrame
    margin::PlotFrame
    fig::PlotBox
    figsize::Tuple{Int,Int}

    function MultiFigure(nrows::Int, ncols::Int, image::PlotBox, pad::PlotFrame,
                         margin::PlotFrame; scale::Real=1.0)
        # Build composite boxes
        subfig = pad + image
        fig = margin + nrows * ybox(subfig) + ncols * xbox(subfig)

        # Rectify scaling so fig dimensions are integers
        intscale = ceil(scale * fig.y) / fig.y
        extra_w = ceil(intscale * fig.x) - intscale * fig.x

        # Apply scale
        image_s = image * intscale
        pad_s = pad * intscale
        margin_s = PlotFrame(margin.top * intscale,
                             margin.bottom * intscale,
                             margin.left * intscale + extra_w / 2,
                             margin.right * intscale + extra_w / 2)

        # Rebuild composite boxes
        subfig_s = pad_s + image_s
        fig_s = margin_s + nrows * ybox(subfig_s) + ncols * xbox(subfig_s)

        figx = Int(round(fig_s.x))
        figy = Int(round(fig_s.y))

        new(nrows, ncols, image_s, pad_s, margin_s, fig_s, (figx, figy))
    end
end

"""
    subfigure_axes(mf::MultiFigure, i::Int, j::Int, rect::NTuple{4,Real})
        -> NTuple{4,Float64}

Compute figure-fraction rectangle for a sub-axes within image `(i, j)`.

# Arguments
- `i`: 1-based row index (top = 1)
- `j`: 1-based column index (left = 1)
- `rect`: `(left, bottom, width, height)` in fractions of image width/height

Returns `(fig_left, fig_bottom, fig_width, fig_height)` in figure fractions.
"""
function subfigure_axes(mf::MultiFigure, i::Int, j::Int,
                        rect::NTuple{4,<:Real})
    # Reverse row index (row 1 is at top, like Python)
    irev = mf.nrows - i  # 0-based reversed index
    subfig = mf.pad + mf.image
    offset = bottom_left(mf.margin) + irev * ybox(subfig) + (j - 1) * xbox(subfig) + bottom_left(mf.pad)

    imstart = PlotBox(Float64(rect[1]), Float64(rect[2]))
    imshape = PlotBox(Float64(rect[3]), Float64(rect[4]))
    figstart = (offset + imstart * mf.image) / mf.fig
    figshape = imshape * mf.image / mf.fig

    return (figstart.x, figstart.y, figshape.x, figshape.y)
end

# ---------------------------------------------------------------------------
# get_1d_vertices
# ---------------------------------------------------------------------------

"""
    get_1d_vertices(grid::AbstractVector; cut_edges::Bool=false) -> Vector{Float64}

Compute vertex positions dividing a 1D grid into cells.

Interior vertices are placed halfway between grid points. Edge vertices are
either placed at the grid edges (`cut_edges=true`) or reflected outward by
half the edge spacing (`cut_edges=false`, the default).
"""
function get_1d_vertices(grid::AbstractVector; cut_edges::Bool=false)
    if ndims(grid) > 1
        throw(ArgumentError("grid must be a 1D array"))
    end
    d = diff(grid)
    vert = zeros(Float64, length(grid) + 1)
    # Interior vertices: halfway between points
    for k in 2:length(grid)
        vert[k] = grid[k-1] + d[k-1] / 2
    end
    # Edge vertices
    if cut_edges
        vert[1] = grid[1]
        vert[end] = grid[end]
    else
        vert[1] = grid[1] - d[1] / 2
        vert[end] = grid[end] + d[end] / 2
    end
    return vert
end

# ---------------------------------------------------------------------------
# quad_mesh
# ---------------------------------------------------------------------------

"""
    quad_mesh(x::AbstractVector, y::AbstractVector;
              cut_x_edges::Bool=false, cut_y_edges::Bool=false)
        -> (xmesh::Matrix, ymesh::Matrix)

Construct quadrilateral mesh arrays from two 1D grids, intended for use with
`pcolormesh`-style plotting commands.

`x` maps to the *columns* (last axis) and `y` maps to the *rows* (first axis).
"""
function quad_mesh(x::AbstractVector, y::AbstractVector;
                   cut_x_edges::Bool=false, cut_y_edges::Bool=false)
    xvert = get_1d_vertices(x; cut_edges=cut_x_edges)
    yvert = get_1d_vertices(y; cut_edges=cut_y_edges)
    # xvert along columns, yvert along rows
    xmesh = reshape(xvert, 1, :) .* ones(Float64, length(yvert), 1)
    ymesh = ones(Float64, 1, length(xvert)) .* reshape(yvert, :, 1)
    return xmesh, ymesh
end

# ---------------------------------------------------------------------------
# pad_limits
# ---------------------------------------------------------------------------

"""
    pad_limits(xgrid, ygrid; xpad=0.0, ypad=0.0) -> NTuple{4,Float64}

Compute padded image limits `(x0, x1, y0, y1)` from x and y grids.

# Arguments
- `xgrid`: grid for the x axis
- `ygrid`: grid for the y axis
- `xpad`:  padding fraction for x axis (default `0.0`)
- `ypad`:  padding fraction for y axis (default `0.0`)
"""
function pad_limits(xgrid, ygrid; xpad::Real=0.0, ypad::Real=0.0)
    xmin, xmax = minimum(xgrid), maximum(xgrid)
    ymin, ymax = minimum(ygrid), maximum(ygrid)
    dx = xmax - xmin
    dy = ymax - ymin
    x0 = xmin - xpad * dx
    x1 = xmax + xpad * dx
    y0 = ymin - ypad * dy
    y1 = ymax + ypad * dy
    return (x0, x1, y0, y1)
end

# ---------------------------------------------------------------------------
# get_plane
# ---------------------------------------------------------------------------

"""
    get_plane(dset, xaxis::Int, yaxis::Int, slices::Tuple;
              xscale=0, yscale=0,
              cut_x_edges::Bool=false, cut_y_edges::Bool=false)
        -> (xmesh::Matrix, ymesh::Matrix, data::Matrix)

Select a 2D plane from a dataset (FieldWrapper or HDF5 dataset) and build
quad meshes suitable for `pcolormesh`-style plotting.

# Arguments
- `dset`:   a `FieldWrapper` (or similar object supporting `dset.dims` and indexing)
- `xaxis`:  1-based axis index for the image x-direction
- `yaxis`:  1-based axis index for the image y-direction
- `slices`: tuple of slices/indices selecting image data from global data
- `xscale`, `yscale`: axis scale indices (default `0`)
- `cut_x_edges`, `cut_y_edges`: see `quad_mesh`

Returns `(xmesh, ymesh, data)` where `xmesh` and `ymesh` are the vertex
grids and `data` is the selected 2D data slice.
"""
function get_plane(dset, xaxis::Int, yaxis::Int, slices::Tuple;
                   xscale=0, yscale=0,
                   cut_x_edges::Bool=false, cut_y_edges::Bool=false)
    slices_vec = collect(slices)

    # Build quad meshes from sorted grids
    xgrid = dset.dims[xaxis][xscale][slices_vec[xaxis]]
    ygrid = dset.dims[yaxis][yscale][slices_vec[yaxis]]
    xorder = sortperm(xgrid)
    yorder = sortperm(ygrid)
    xmesh, ymesh = quad_mesh(xgrid[xorder], ygrid[yorder];
                             cut_x_edges=cut_x_edges, cut_y_edges=cut_y_edges)

    # Select and arrange data
    data = dset[slices...]
    if xaxis < yaxis
        data = permutedims(data)
    end
    data = data[yorder, :]
    data = data[:, xorder]

    return xmesh, ymesh, data
end

# ---------------------------------------------------------------------------
# plot_bot
# ---------------------------------------------------------------------------

"""
    plot_bot(dset, image_axes::Tuple{Int,Int}, data_slices::Tuple;
             image_scales=(0,0), clim=nothing, even_scale::Bool=false,
             title=nothing, func=nothing)
        -> (xmesh, ymesh, data, xlabel, ylabel, title, clim)

Compute 2D plot data from a dataset/field slice.

Unlike the Python version which directly creates a matplotlib figure, this
function returns the computed meshes, data, labels, and color limits so the
caller can render with any plotting backend.

# Arguments
- `dset`:         `Field`, `FieldWrapper`, or compatible object
- `image_axes`:   `(xaxis, yaxis)` — 1-based axis indices
- `data_slices`:  tuple of slices/ints selecting the 2D slice
- `image_scales`: `(xscale, yscale)` (default `(0, 0)`)
- `clim`:         `(cmin, cmax)` or `nothing` for auto
- `even_scale`:   if `true`, expand clim symmetrically around 0
- `title`:        plot title (default: dataset name)
- `func`:         `(xmesh, ymesh, data) -> (xmesh, ymesh, data)` transform

# Returns
A `NamedTuple` with fields:
- `xmesh`, `ymesh`, `data`: the 2D mesh and data arrays
- `xlabel`, `ylabel`, `title`: axis and title strings
- `clim`: `(cmin, cmax)` color limits
- `limits`: `(x0, x1, y0, y1)` axis limits
"""
function plot_bot(dset, image_axes::Tuple{Int,Int}, data_slices::Tuple;
                  image_scales::Tuple=(0, 0),
                  clim=nothing,
                  even_scale::Bool=false,
                  title=nothing,
                  func=nothing)
    # Wrap fields
    if dset isa Field
        dset = FieldWrapper(dset)
    end

    xaxis, yaxis = image_axes
    xscale, yscale = image_scales

    # Get meshes and data
    xmesh, ymesh, data = get_plane(dset, xaxis, yaxis, data_slices;
                                   xscale=xscale, yscale=yscale)
    if func !== nothing
        xmesh, ymesh, data = func(xmesh, ymesh, data)
    end

    # Colour limits
    if clim === nothing
        if even_scale
            lim = max(abs(minimum(data)), abs(maximum(data)))
            clim = (-lim, lim)
        else
            clim = (minimum(data), maximum(data))
        end
    end

    # Labels
    if title === nothing
        title = get(dset.attrs, "name", "")
    end

    if xscale isa AbstractString
        xlabel = xscale
    else
        xlabel = label(dset.dims[xaxis])
    end

    if yscale isa AbstractString
        ylabel = yscale
    else
        ylabel = label(dset.dims[yaxis])
    end

    limits = pad_limits(xmesh, ymesh)

    return (xmesh=xmesh, ymesh=ymesh, data=data,
            xlabel=xlabel, ylabel=ylabel, title=title,
            clim=clim, limits=limits)
end

# ---------------------------------------------------------------------------
# plot_bot_2d
# ---------------------------------------------------------------------------

"""
    plot_bot_2d(dset; transpose::Bool=false, kwargs...)

Convenience wrapper around `plot_bot` for 2D datasets.

# Arguments
- `dset`:      `Field`, `FieldWrapper`, or compatible 2D dataset
- `transpose`: if `true`, swap x/y axes (default `false`)

All other keyword arguments are forwarded to `plot_bot`.
"""
function plot_bot_2d(dset; transpose::Bool=false, kwargs...)
    if dset isa Field
        dset = FieldWrapper(dset)
    end

    sz = shape(dset)
    if length(sz) != 2
        throw(ArgumentError("plot_bot_2d is for 2D datasets only (got $(length(sz))D)"))
    end

    image_axes = transpose ? (2, 1) : (1, 2)
    data_slices = (Colon(), Colon())

    return plot_bot(dset, image_axes, data_slices; kwargs...)
end

# ---------------------------------------------------------------------------
# plot_bot_3d
# ---------------------------------------------------------------------------

"""
    plot_bot_3d(dset, normal_axis, normal_index::Int;
                transpose::Bool=false, kwargs...)

Convenience wrapper around `plot_bot` for 3D datasets, selecting a 2D slice
normal to the specified axis.

# Arguments
- `dset`:          `Field`, `FieldWrapper`, or compatible 3D dataset
- `normal_axis`:   1-based axis index **or** axis name (string)
- `normal_index`:  1-based index along the normal direction
- `transpose`:     if `true`, swap the two image axes (default `false`)

All other keyword arguments are forwarded to `plot_bot`.
"""
function plot_bot_3d(dset, normal_axis, normal_index::Int;
                     transpose::Bool=false, kwargs...)
    if dset isa Field
        dset = FieldWrapper(dset)
    end

    sz = shape(dset)
    if length(sz) != 3
        throw(ArgumentError("plot_bot_3d is for 3D datasets only (got $(length(sz))D)"))
    end

    # Resolve axis name to axis index
    if normal_axis isa AbstractString
        found = false
        for (ax, dim) in enumerate(dset.dims)
            if normal_axis == label(dim)
                normal_axis = ax
                found = true
                break
            end
        end
        if !found
            throw(ArgumentError("Axis name '$(normal_axis)' not found"))
        end
    end

    # Build image axes: the two axes that are not the normal axis
    all_axes = (1, 2, 3)
    image_axes = Tuple(a for a in all_axes if a != normal_axis)
    if transpose
        image_axes = (image_axes[2], image_axes[1])
    end

    data_slices_vec = Any[Colon(), Colon(), Colon()]
    data_slices_vec[normal_axis] = normal_index
    data_slices = Tuple(data_slices_vec)

    return plot_bot(dset, image_axes, data_slices; kwargs...)
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export FieldWrapper, DimWrapper, label, shape,
       PlotBox, xbox, ybox,
       PlotFrame, bottom_left, top_right,
       MultiFigure, subfigure_axes,
       get_1d_vertices, quad_mesh, pad_limits, get_plane,
       plot_bot, plot_bot_2d, plot_bot_3d

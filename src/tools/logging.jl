"""
Logging system for the Dedalus framework.

Translated from dedalus/tools/logging.py. Provides a custom logger that formats
messages similarly to the Python Dedalus logger, with support for stdout and
file handlers configured via the Dedalus config system.

Currently operates in serial mode (no MPI). When MPI support is added, rank and
size will be read from the communicator.
"""

using Logging
using Dates

# ---------------------------------------------------------------------------
# MPI placeholders (serial mode)
# ---------------------------------------------------------------------------

const MPI_RANK = 0
const MPI_SIZE = 1

# ---------------------------------------------------------------------------
# Log-level mapping
# ---------------------------------------------------------------------------

"""
    _parse_log_level(level::AbstractString) -> Logging.LogLevel

Map a Python-style log level name to a Julia `Logging.LogLevel`.
Recognised names (case-insensitive): `"debug"`, `"info"`, `"warning"` / `"warn"`,
`"error"`. Returns `Logging.Debug` for unrecognised names.
"""
function _parse_log_level(level::AbstractString)
    lwr = lowercase(strip(level))
    if lwr == "debug"
        return Logging.Debug
    elseif lwr == "info"
        return Logging.Info
    elseif lwr in ("warning", "warn")
        return Logging.Warn
    elseif lwr == "error"
        return Logging.Error
    else
        return Logging.Debug
    end
end

# ---------------------------------------------------------------------------
# DedalusLogger
# ---------------------------------------------------------------------------

"""
    DedalusLogger <: AbstractLogger

Custom logger that formats messages in the same style as the Python Dedalus
logger: `<timestamp> <module> <rank>/<size> <level> :: <message>`.

Supports multiple output sinks (IO streams and/or files) each with their own
minimum log level.
"""
mutable struct DedalusLogger <: AbstractLogger
    min_level::LogLevel
    sinks::Vector{Tuple{IO, LogLevel}}   # (io, min_level) pairs
end

"""
    DedalusLogger(; level=Logging.Debug)

Create a `DedalusLogger` with no sinks. Add sinks via
[`add_stdout_sink!`](@ref) and [`add_file_sink!`](@ref).
"""
DedalusLogger(; level::LogLevel=Logging.Debug) = DedalusLogger(level, Tuple{IO, LogLevel}[])

Logging.shouldlog(logger::DedalusLogger, level, _module, group, id) =
    level >= logger.min_level

Logging.min_enabled_level(logger::DedalusLogger) = logger.min_level

Logging.catch_exceptions(::DedalusLogger) = false

function Logging.handle_message(logger::DedalusLogger, level, message, _module, group,
                                id, filepath, line; kwargs...)
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    modname = _module === nothing ? "Main" : string(_module)
    levelstr = _level_string(level)
    formatted = "$timestamp $modname $MPI_RANK/$MPI_SIZE $levelstr :: $message"

    # Append key=value pairs if any
    buf = IOBuffer()
    print(buf, formatted)
    for (k, v) in kwargs
        print(buf, " | ", k, "=", v)
    end
    line_out = String(take!(buf))

    for (io, sink_level) in logger.sinks
        if level >= sink_level
            println(io, line_out)
            flush(io)
        end
    end
end

"""
    _level_string(level::LogLevel) -> String

Convert a Julia `LogLevel` to a short uppercase label.
"""
function _level_string(level::LogLevel)
    if level == Logging.Debug
        return "DEBUG"
    elseif level == Logging.Info
        return "INFO"
    elseif level == Logging.Warn
        return "WARNING"
    elseif level == Logging.Error
        return "ERROR"
    else
        return string(level)
    end
end

# ---------------------------------------------------------------------------
# Sink helpers
# ---------------------------------------------------------------------------

"""
    add_stdout_sink!(logger::DedalusLogger, level::AbstractString)

Add a stdout sink to `logger` at the given log level.
"""
function add_stdout_sink!(logger::DedalusLogger, level::AbstractString)
    lvl = _parse_log_level(level)
    push!(logger.sinks, (stdout, lvl))
    return logger
end

"""
    add_file_sink!(logger::DedalusLogger, filename::AbstractString, level::AbstractString)

Add a file sink to `logger`. The file is opened in write mode. Parent
directories are created if necessary. The filename is suffixed with the MPI
rank, e.g. `"logs/dedalus_p0.log"`.
"""
function add_file_sink!(logger::DedalusLogger, filename::AbstractString, level::AbstractString)
    lvl = _parse_log_level(level)
    filepath = "$(filename)_p$(MPI_RANK).log"
    dirpath = dirname(filepath)
    if !isempty(dirpath) && !isdir(dirpath)
        mkpath(dirpath)
    end
    io = open(filepath, "w")
    push!(logger.sinks, (io, lvl))
    return logger
end

# ---------------------------------------------------------------------------
# Global logger instance & setup
# ---------------------------------------------------------------------------

"""
    _dedalus_logger :: Ref{Union{DedalusLogger, Nothing}}

Module-level reference to the active `DedalusLogger`. Populated by
[`setup_logging!`](@ref).
"""
const _dedalus_logger = Ref{Union{DedalusLogger, Nothing}}(nothing)

"""
    setup_logging!()

Initialise the Dedalus logging system based on the current configuration.
Creates a [`DedalusLogger`](@ref), configures stdout and file sinks as
specified in `config["logging"]`, and installs the logger as the Julia
global logger.

Safe to call multiple times; subsequent calls replace the previous logger.
"""
function setup_logging!()
    logger = DedalusLogger(; level=Logging.Debug)

    # Read config (may not have a "logging" section)
    log_cfg = get(config, "logging", Dict{String,Any}())

    # Stdout handler
    stdout_level = get(log_cfg, "stdout_level", "none")
    if lowercase(stdout_level) != "none"
        add_stdout_sink!(logger, stdout_level)
    end

    # File handler
    file_level = get(log_cfg, "file_level", "none")
    if lowercase(file_level) != "none"
        filename = get(log_cfg, "filename", "logs/dedalus")
        add_file_sink!(logger, filename, file_level)
    end

    # Non-root level (serial mode: rank == 0, so this only affects MPI runs)
    nonroot_level = get(log_cfg, "nonroot_level", "none")
    if lowercase(nonroot_level) != "none" && MPI_RANK > 0
        logger.min_level = _parse_log_level(nonroot_level)
    end

    _dedalus_logger[] = logger
    Logging.global_logger(logger)
    return logger
end

# ---------------------------------------------------------------------------
# Convenience re-export of add_file_handler for external use
# ---------------------------------------------------------------------------

"""
    add_file_handler(filename::AbstractString, level::AbstractString)

Add a file handler to the active Dedalus logger. If [`setup_logging!`](@ref)
has not been called yet, it is called first.
"""
function add_file_handler(filename::AbstractString, level::AbstractString)
    if _dedalus_logger[] === nothing
        setup_logging!()
    end
    return add_file_sink!(_dedalus_logger[], filename, level)
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export setup_logging!

"""
    Configuration system for Dedalus.jl

Loads configuration from TOML files in hierarchical order (lowest to highest precedence):
1. Package defaults: `joinpath(@__DIR__, "..", "dedalus.toml")`
2. User config: `joinpath(homedir(), ".dedalus", "dedalus.toml")`
3. Local config: `joinpath(pwd(), "dedalus.toml")`

Values from higher-precedence files override those from lower-precedence files.
This is a Julia translation of the Python Dedalus `config.py` module, using TOML
instead of INI/ConfigParser format.
"""

import TOML

"""
    deep_merge(base::Dict, override::Dict) -> Dict

Recursively merge two nested dictionaries. Values from `override` take precedence
over values from `base`. When both `base` and `override` contain a Dict for the
same key, the Dicts are merged recursively rather than replaced wholesale.
"""
function deep_merge(base::Dict, override::Dict)
    merged = copy(base)
    for (key, val) in override
        if haskey(merged, key) && isa(merged[key], Dict) && isa(val, Dict)
            merged[key] = deep_merge(merged[key], val)
        else
            merged[key] = val
        end
    end
    return merged
end

"""
    _load_config() -> Dict

Load and merge configuration from the three standard TOML config file locations.
Files that do not exist are silently skipped.
"""
function _load_config()
    config = Dict{String, Any}()

    # Config file paths in order of increasing precedence
    paths = [
        joinpath(@__DIR__, "..", "dedalus.toml"),       # Package defaults
        joinpath(homedir(), ".dedalus", "dedalus.toml"), # User config
        joinpath(pwd(), "dedalus.toml"),                 # Local config
    ]

    for path in paths
        if isfile(path)
            data = TOML.parsefile(path)
            config = deep_merge(config, data)
        end
    end

    return config
end

"""
    config :: Dict{String, Any}

Global configuration dictionary, loaded at module initialization from TOML config
files. Sections are top-level keys mapping to Dicts of key-value pairs.
"""
const config = _load_config()

"""
    get_config(section::AbstractString, key::AbstractString) -> Any

Retrieve a configuration value from the given `section` and `key`.
Throws a `KeyError` if the section or key does not exist.

# Examples
```julia
get_config("logging", "stdout_level")   # "info"
get_config("transforms", "DEFAULT_LIBRARY")  # "fftw"
```
"""
function get_config(section::AbstractString, key::AbstractString)
    return config[section][key]
end

"""
    get_config_bool(section::AbstractString, key::AbstractString) -> Bool

Retrieve a configuration value and interpret it as a boolean.

If the stored value is already a `Bool`, it is returned directly.
If the stored value is a `String`, it is compared case-insensitively
against `"true"`, `"1"`, `"yes"` (returns `true`) or
`"false"`, `"0"`, `"no"` (returns `false`).
Throws an `ArgumentError` for unrecognized string values and a
`KeyError` if the section or key does not exist.

# Examples
```julia
get_config_bool("transforms", "GROUP_TRANSFORMS")  # false
get_config_bool("memory", "STORE_OUTPUTS")          # true
```
"""
function get_config_bool(section::AbstractString, key::AbstractString)::Bool
    val = config[section][key]
    if isa(val, Bool)
        return val
    elseif isa(val, AbstractString)
        lower = lowercase(val)
        if lower in ("true", "1", "yes")
            return true
        elseif lower in ("false", "0", "no")
            return false
        else
            throw(ArgumentError("Cannot interpret config value \"$val\" as Bool for [$section] $key"))
        end
    else
        throw(ArgumentError("Cannot interpret config value of type $(typeof(val)) as Bool for [$section] $key"))
    end
end

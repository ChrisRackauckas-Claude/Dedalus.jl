"""
Parsing utilities for the Dedalus framework.

Translated from dedalus/tools/parsing.py. Provides helpers for splitting
equation strings, parsing math-style function call syntax, and converting
function definitions into anonymous function (lambda) expressions.
"""

# ---------------------------------------------------------------------------
# split_equation
# ---------------------------------------------------------------------------

"""
    split_equation(equation::AbstractString) -> Tuple{String, String}

Split an equation string into LHS and RHS strings at the unique top-level
equals sign. Parenthesised `=` characters are ignored. Throws a
`SymbolicParsingError` if the equation contains zero or more than one
top-level `=`.

# Examples
```julia
split_equation("f(x) = x^2")            # ("f(x)", "x^2")
split_equation("(a=b) = c")             # ("(a=b)", "c")
```
"""
function split_equation(equation::AbstractString)
    parentheses = 0
    top_level_equals = Int[]
    for (i, ch) in enumerate(equation)
        if ch == '('
            parentheses += 1
        elseif ch == ')'
            parentheses -= 1
        elseif ch == '=' && parentheses == 0
            push!(top_level_equals, i)
        end
    end
    if length(top_level_equals) == 0
        throw(SymbolicParsingError("Equation contains no top-level equals signs."))
    elseif length(top_level_equals) > 1
        throw(SymbolicParsingError("Equation contains multiple top-level equals signs."))
    end
    idx = top_level_equals[1]
    lhs = strip(equation[1:idx-1])
    rhs = strip(equation[idx+1:end])
    return (String(lhs), String(rhs))
end

# ---------------------------------------------------------------------------
# split_call
# ---------------------------------------------------------------------------

"""
    split_call(call::AbstractString) -> Tuple{String, Tuple{Vararg{String}}}

Convert a math-style function definition string into a head name and a tuple
of argument names. If the string does not match function-call syntax, the
full string is returned as the head with an empty argument tuple.

# Examples
```julia
split_call("f(x, y)")   # ("f", ("x", "y"))
split_call("f(x)")      # ("f", ("x",))
split_call("x")          # ("x", ())
```
"""
function split_call(call::AbstractString)
    m = match(r"^(.+)\((.*)\)$", call)
    if m !== nothing
        head = String(m.captures[1])
        argstring = replace(m.captures[2], " " => "")
        if isempty(argstring)
            args = ()
        else
            args = Tuple(String.(split(argstring, ",")))
        end
        return (head, args)
    else
        return (String(call), ())
    end
end

# ---------------------------------------------------------------------------
# lambdify_functions
# ---------------------------------------------------------------------------

"""
    lambdify_functions(call::AbstractString, result::AbstractString) -> Tuple{String, String}

Convert a math-style function definition and its result expression into a
name and an anonymous-function string. If `call` has no arguments, `call`
and `result` are returned unchanged.

In Julia the anonymous function syntax is `(args...) -> body`.

# Examples
```julia
lambdify_functions("f(x, y)", "x*y")   # ("f", "(x,y) -> x*y")
lambdify_functions("c", "42")           # ("c", "42")
```
"""
function lambdify_functions(call::AbstractString, result::AbstractString)
    head, args = split_call(call)
    if !isempty(args)
        argstring = join(args, ",")
        return (head, "($argstring) -> $result")
    else
        return (String(call), String(result))
    end
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export split_equation,
       split_call,
       lambdify_functions

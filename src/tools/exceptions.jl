"""
Custom exception types for the Dedalus framework.

Translated from dedalus/tools/exceptions.py. In Julia, custom exceptions are
structs that subtype `Exception`. Each type carries a `msg::String` field and
implements `Base.showerror` for pretty-printing.
"""

# ---------------------------------------------------------------------------
# NonlinearOperatorError
# ---------------------------------------------------------------------------

"""
    NonlinearOperatorError(msg)

Exception raised for operands that fail linearity tests.
"""
struct NonlinearOperatorError <: Exception
    msg::String
end

Base.showerror(io::IO, e::NonlinearOperatorError) =
    print(io, "NonlinearOperatorError: ", e.msg)

# ---------------------------------------------------------------------------
# DependentOperatorError
# ---------------------------------------------------------------------------

"""
    DependentOperatorError(msg)

Exception raised for operands that fail independence tests.
"""
struct DependentOperatorError <: Exception
    msg::String
end

Base.showerror(io::IO, e::DependentOperatorError) =
    print(io, "DependentOperatorError: ", e.msg)

# ---------------------------------------------------------------------------
# SymbolicParsingError
# ---------------------------------------------------------------------------

"""
    SymbolicParsingError(msg)

Exception raised for syntactic and mathematical problems in equations.
"""
struct SymbolicParsingError <: Exception
    msg::String
end

Base.showerror(io::IO, e::SymbolicParsingError) =
    print(io, "SymbolicParsingError: ", e.msg)

# ---------------------------------------------------------------------------
# UnsupportedEquationError
# ---------------------------------------------------------------------------

"""
    UnsupportedEquationError(msg)

Exception raised for valid but unsupported equations.
"""
struct UnsupportedEquationError <: Exception
    msg::String
end

Base.showerror(io::IO, e::UnsupportedEquationError) =
    print(io, "UnsupportedEquationError: ", e.msg)

# ---------------------------------------------------------------------------
# UndefinedParityError
# ---------------------------------------------------------------------------

"""
    UndefinedParityError(msg)

Exception raised for data or operations with undefined parity.
"""
struct UndefinedParityError <: Exception
    msg::String
end

Base.showerror(io::IO, e::UndefinedParityError) =
    print(io, "UndefinedParityError: ", e.msg)

# ---------------------------------------------------------------------------
# SkipDispatchException
# ---------------------------------------------------------------------------

"""
    SkipDispatchException(output)

Exception used to short-circuit `MultiClass` / `dispatch_construct` dispatch.
When thrown inside `check_args`, the dispatch machinery catches it and returns
`output` directly instead of continuing the subtype search.
"""
struct SkipDispatchException <: Exception
    output::Any
    msg::String
end

SkipDispatchException(output) = SkipDispatchException(output, "Dispatch skipped")

Base.showerror(io::IO, e::SkipDispatchException) =
    print(io, "SkipDispatchException: ", e.msg)

export NonlinearOperatorError,
       DependentOperatorError,
       SymbolicParsingError,
       UnsupportedEquationError,
       UndefinedParityError,
       SkipDispatchException

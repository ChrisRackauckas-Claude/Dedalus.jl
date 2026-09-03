# Operators module - stub pending full implementation
# This file will be replaced when the operators agent completes

# Forward declarations for operator constructor functions
function convert_operand end
function differentiate end
function interpolate end
function integrate end
function average end
function lift end
function gradient end
function divergence end
function curl end
function laplacian end
function trace_op end
function transpose_components end
function grid_op end
function coeff_op end
function time_derivative end

# Power operator (needed by field.jl arithmetic)
function dedalus_power(a, b)
    error("Power operator not yet implemented")
end

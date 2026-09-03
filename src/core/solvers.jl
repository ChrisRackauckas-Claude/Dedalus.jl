# Solvers module - stub pending full implementation

abstract type AbstractSolver end

mutable struct LinearBoundaryValueSolver <: AbstractSolver
    problem::Any
    dist::Any
    evaluator::Any
    iteration::Int
end

mutable struct EigenvalueSolver <: AbstractSolver
    problem::Any
    dist::Any
    evaluator::Any
    iteration::Int
end

mutable struct InitialValueSolver <: AbstractSolver
    problem::Any
    dist::Any
    evaluator::Any
    iteration::Int
    initial_iteration::Int
    sim_time::Float64
    stop_sim_time::Float64
    stop_wall_time::Float64
    stop_iteration::Int
end

mutable struct NonlinearBoundaryValueSolver <: AbstractSolver
    problem::Any
    dist::Any
    evaluator::Any
    iteration::Int
end

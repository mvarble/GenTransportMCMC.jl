using Gen
import LinearAlgebra: eigvals
import DifferentialEquations: solve, ODEProblem, OrdinaryDiffEq

# --------------------------------------------------------------------------------------------
# We create a type for the parameter associated to the predator-prey differential equation
# --------------------------------------------------------------------------------------------
mutable struct PredatorPreyParameter
  P0::Float64
  Q0::Float64
  r::Float64
  K::Float64
  s::Float64
  a::Float64
  u::Float64
  v::Float64
  start_time::Float64
  stop_time::Float64
  solution::Union{Nothing, OrdinaryDiffEq.ODECompositeSolution}
end
Base.Array(p::PredatorPreyParameter) = [
  p.P0, p.Q0, p.r, p.K, p.s, p.a, p.u, p.v, p.start_time, p.stop_time, p.solution
]
init(p::PredatorPreyParameter) = [p.P0, p.Q0]
ode_params(p::PredatorPreyParameter) = [p.r, p.K, p.s, p.a, p.u, p.v]
times(p::PredatorPreyParameter) = (p.start_time, p.stop_time)
Base.display(p::PredatorPreyParameter) = print(
"""
PredatorPreyParameter ($(isnothing(p.solution) ? "without" : "with") solution):
  P0: $(p.P0)
  Q0: $(p.Q0)
  r: $(p.r)
  K: $(p.K)
  s: $(p.s)
  a: $(p.a)
  u: $(p.u)
  v: $(p.v)
  start_time: $(p.start_time)
  stop_time: $(p.stop_time)
""")

struct PredatorPreyPrior <: Gen.Distribution{PredatorPreyParameter} end
const predator_prey_prior = PredatorPreyPrior()

# --------------------------------------------------------------------------------------------
# Use DifferentialEquations.jl to simulate solutions
# --------------------------------------------------------------------------------------------
function predator_prey_system(dy, y, p, t)
  r, K, s, a, u, v = p
  dy[1] = r * y[1] * (1 - y[1] / K) - s * y[1] * y[2] / (a + y[1])
  dy[2] = u * y[1] * y[2] / (a + y[1]) - v * y[2]
end

function predator_prey_problem(parameter::PredatorPreyParameter)
  ODEProblem(predator_prey_system, init(parameter), times(parameter), ode_params(parameter))
end

# --------------------------------------------------------------------------------------------
# Prior distribution imposes domain knowledge: parameters must produce stable populations
# --------------------------------------------------------------------------------------------
function stable!(parameter::PredatorPreyParameter)::Bool
  # if we have a solution already, we are good
  if !isnothing(parameter.solution)
    return true
  end

  # unpack the parameter
  P0, Q0 = init(parameter)
  r, K, s, a, u, v = ode_params(parameter)

  # if P0 > K, population is already above carrying capacity
  if P0 > K
    return false
  end

  # stable point must be positive and under carrying capacity
  Pf = a * v / (u - v)
  Qf = a * r * (1 + v / (u - v)) * (K - a * v / (u - v)) / (K * s)
  if (Pf <= 0 || Pf > K || Qf <= 0)
    return false
  end

  # Jacobian at stable point must have eigenvalues with (almost) positive real components
  jacobian = [
    (r - 2r*Pf/K - s*((a+Pf)*Qf - Pf*Qf)/(a+Pf)^2) (-s*Pf/(a+Pf)) ;
    (u*((a+Pf)*Qf - Pf*Qf)/(a+Pf)^2) (u*Pf/(a+Pf) - v)
  ]
  lam, = eigvals(jacobian)
  if real(lam) < -1e-14
    return false
  end

  # we passed cyclic condition; check if numerically stable
  solution = solve(predator_prey_problem(parameter), verbose=false)
  if solution.retcode === :Success
    parameter.solution = solution
    return true
  end
  return false
end

function Gen.random(
  dist::PredatorPreyPrior, 
  lower::Float64, 
  upper::Float64, 
  start_time::Float64, 
  stop_time::Float64
)
  # parse arguments
  if lower <= 0
    throw(ArgumentError("PredatorPreyPrior must assume positive parameters"))
  end

  # rejection sample uniform
  while true
    Q0, r, K, s, a, u, v = [rand() * (upper-lower) + lower for _=1:7]
    P0 = rand() * (K - lower) + lower
    p = PredatorPreyParameter(P0, Q0, r, K, s, a, u, v, start_time, stop_time, nothing)

    if stable!(p)
      return p
    end
  end
end

function Gen.logpdf(
  dist::PredatorPreyPrior, 
  param::PredatorPreyParameter, 
  lower::Float64, 
  upper::Float64,
  start_time::Float64,
  stop_time::Float64
) 
  is_stable = stable!(param)
  same_times = (start_time, stop_time) == times(param)
  possible = same_times && all(x -> x >= lower && x <= upper, Array(param)[1:8]) && is_stable
  possible ? -Inf : 0.0
end
Gen.has_output_grad(dist::PredatorPreyPrior) = true
Gen.has_argument_grads(dist::PredatorPreyPrior) = (true, true)
Gen.logpdf_grad(dist::PredatorPreyPrior, value::PredatorPreyParameter, lower::Float64, upper::Float64) = (0.0, 1.0, -1.0)

@gen function measurement(mu::Vector{Float64}, var::Float64)
  @trace(mvnormal(mu, [var 0.0 ; 0.0 var]), :z)
end

@gen function predator_prey_model(
  lower_prior::Float64, 
  upper_prior::Float64, 
  start_time::Float64,
  stop_time::Float64,
  timesteps::Int64,
  measurement_noise::Float64
)
  # sample our custom prior
  parameter = @trace(
    predator_prey_prior(lower_prior, upper_prior, start_time, stop_time), 
    :predator_prey_parameter
  )

  # simulate measurements
  times = LinRange(start_time, stop_time, timesteps+1)[2:end]
  means = map(t -> parameter.solution(t), times)
  noises = fill(measurement_noise, timesteps)
  @trace(Map(measurement)(means, noises), :measurements)
end

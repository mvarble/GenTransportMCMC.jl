import NLsolve: nlsolve
import ReverseDiff
import LinearAlgebra: tr

export OrderedBasis, TransportFamily, TransportMap, inverse, sample_cost

"""
    abstract type OrderedBasis end

Abstract type corresponding to an ordered basis of scalar-functions 
``(ψ_j)_{j\\in\\mathcal J_i}``.  An instance `tuple` of this type will act as a map, so that 
`tuple(θ)` will correspond to the vector ``(ψ_j(θ))_{j \\in \\mathcal J_i}``.

To implement a subtype `B <: OrderedBasis`, one must implement:

```julia
(tuple::B)(θ::AbstractArray{<:Real})::AbstractArray{<:Real}
Base.length(tuple::B)::Int64
```
"""
abstract type OrderedBasis end

function Base.length(tuple::OrderedBasis)
  error("must implement Base.length(::$(typeof(tuple)))")
end

"""
    T = TransportFamily(bases::AbstractArray{<:OrderedBasis})

Provide an array `bases` of `OrderedBasis` instances where each element `bases[i]`
corresponds to some ordered basis ``(ψ_j)_{j\\in\\mathcal J_i}`` and receive the induced 
family ``\\{ \\tilde T(\\cdot; γ) \\}_{γ \\in \\mathcal A}`` (`T`).

With an instance `T` of a `TransportFamily`, one may apply `T(γ)` to a parameter `γ` and get a 
`TransportMap` instance corresponding to the map ``T(\\cdot; γ)``.
Recall, this is defined so that the ``i``-th component ``T_i(\\cdot; γ_i)`` is as follows.

```math
T_i(θ; γ_i) = \\sum_{j\\in\\mathcal J_i} γ_{i,j} ψ_j(θ)
```
"""
struct TransportFamily{B <: OrderedBasis}
  bases::AbstractArray{B}
end

function Base.display(T::TransportFamily)
  println(
    """
    TransportFamily on R^$(length(T.bases))
    $(join(map(p -> "  J_$(p[1]) = $(p[2])", enumerate(T.bases)), "\n"))"""
  )
end

"""
    map = TransportMap(dim, parameter, f)

Wrap a provided function `f` with a `TransportMap` type for the type checking.
In addition, we keep track of the dimension (`dim`) of the vector space on which `f` is 
defined and the parameter (`parameter`) of the specific map.
"""
struct TransportMap{B <: OrderedBasis}
  dim::Int
  parameter::AbstractArray{<:AbstractArray{<:Real}}
  f::Function
end

function (F::TransportMap)(theta)::AbstractArray{<:Real}
  if length(theta) !== F.dim
    throw(ArgumentError("$theta must have length as $(F.dim)"))
  end
  r = F.f(theta)
  if length(r) !== F.dim
    println("Warning: the codomain does not have the appropriate dimension")
  end
  r
end

function Base.display(F::TransportMap{B}) where B <: OrderedBasis
  println(
    """
    TransportMap{$B}
    $(join(map(p -> "  γ_$(p[1]) = $(p[2])", enumerate(F.parameter)), "\n"))"""
  )
end

function (T::TransportFamily{B})(gamma::AbstractArray{<:AbstractArray{<:Real}})::TransportMap{B} where B
  # make sure there is a parameter for each component function
  if (length(gamma) !== length(T.bases))
    throw(ArgumentError("$(T.bases) must have same length as $gamma"))
  end
  d = length(gamma)

  # span the components
  TransportMap{B}(d, gamma, theta -> begin
    map(pair -> begin
      # test if the i-th component basis has the same length as the matching parameter
      basis, gam_i = pair
      tuple = basis(theta)
      if length(gam_i) !== length(tuple)
        throw(ArgumentError(
          "The ordered basis $(basis) must have the same amount of components as $(gam_i)"
        ))
      end

      # take the linear combination
      reduce((s, p) -> s + p[1] * p[2], zip(gam_i, tuple); init=0.0)
    end, zip(T.bases, gamma))
  end)
end

"""
    theta = inverse(tmap::TransportMap, r)

Return ``θ`` such that ``T(θ; γ) = r``, where ``T(\\cdot; γ)`` is our transport map 
(`tmap`).
"""
function inverse(tmap::TransportMap, r::AbstractArray{<:Real})::AbstractArray{<:Real}
  if length(r) !== tmap.dim
    throw(ArgumentError("$r must have length $(tmap.dim)"))
  end
  reduce(
    (theta, pair) -> begin
      # unpack the index and image component
      i, r_i = pair

      # preallocate the theta components we have solved and those that are unused
      theta_solved = theta[1:(i-1)]
      theta_unused = theta[(i+1):end]

      # create the partially-evaluated component function, ready for NLsolve
      func! = (F, x) -> begin
        F[1] = tmap([theta_solved..., x[1], theta_unused...])[i] - r_i
      end

      # solve the variable and return the update
      theta_i = nlsolve(func!, [0.0]).zero[1]
      [theta_solved..., theta_i, theta_unused...]
    end, 
    enumerate(r);
    init=zeros(tmap.dim)
  )
end

"""
    cost = sample_cost(tmap::TransportMap, samples::AbstractArray{<:AbstractArray{<:Real}})

Compute the cost ``C(\\tilde T(\\cdot; γ))`` of a transport map 
``\\tilde T(\\cdot; γ)`` based off of samples ``θ_1, \\ldots, θ_K``.

```math
C\\big(\\tilde T(\\cdot; γ)\\big) = \\sum_{i=1}^d\\sum_{k=1}^K \\Big( 
  \\frac{1}{2} \\tilde T_i^2(θ^{(k)}; γ_i) - \\log\\frac{∂\\tilde T_i(\\cdot; γ_i)}{∂θ_i}(θ^{(k)}) 
\\Big)
```
"""
function sample_cost(tmap::TransportMap, samples::AbstractArray{<:AbstractArray{<:Real}})
  sum(
    map(
      sample -> 0.5 * sum(tmap(sample).^2) - tr(log.(ReverseDiff.jacobian(tmap, sample))),
      samples,
    )
  )
end

#---------------------------------------------------------------------------------------------
# specific implementations
#---------------------------------------------------------------------------------------------
include("polynomials.jl")

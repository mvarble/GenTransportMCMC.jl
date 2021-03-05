#---------------------------------------------------------------------------------------------
# transport map types
#---------------------------------------------------------------------------------------------

export OrderedBasis, TransportMap, inverse, TransportFamily

"""
    abstract type OrderedBasis end

Abstract type corresponding to an ordered basis of scalar-functions ``(ψ_1, \\ldots, ψ_n)``. 
An instance `tuple` of this type will act as a map, so that `tuple(Θ)` will correspond to 
the vector ``(ψ_1(Θ), \\ldots, ψ_n(Θ))``.

To implement a subtype `B <: OrderedBasis`, one must implement:

```julia
(tuple::B)(Θ::AbstractArray{<:Real})::AbstractArray{<:Real}
Base.length(tuple::B)::Int64
Base.union(tuple::B, tuples...)
```
"""
abstract type OrderedBasis end

function Base.length(tuple::OrderedBasis)
  error("must implement Base.length(::$(typeof(tuple)))")
end

function Base.union(tuple::OrderedBasis, tuples...)
  error("must implement Base.union(tuple::$(typeof(tuple)), tuples...)")
end

"""
    T = TransportFamily(bases::AbstractArray{<:OrderedBasis})

A struct corresponding to a family of lower-triangular vector-valued maps 
``\\{T(\\cdot; γ)\\}_{γ\\in\\mathcal A}``. The family will have a fixed dimension 
``d \\in \\mathbb N`` of which each map ``T(\\cdot; γ)`` has the following structure.

```math
T(\\cdot, γ): \\mathbb R^d \\rightarrow \\mathbb R^d
``` 

Furthermore, the parameter set ``\\mathcal A`` can be written as the following product

```math
\\mathcal A = \\prod_{i=1}^d \\mathbb R^{|\\mathcal J_i|}
```

where ``\\mathcal J_i`` is an index set for an ordered basis of functions 
``(ψ_j)_{j\\in\\mathcal J_i}`` which only depend on the first ``i`` input variables.
This way, each parameter ``γ = (γ_1, \\ldots, γ_d) \\in \\mathcal A`` can determine the 
components of ``T(\\cdot; γ)`` in the following linear fashion.

```math
T_i(Θ; γ_i) = \\sum_{j\\in\\mathcal J_i} γ_{i,j} ψ_j(Θ)
```

The constructor `T = TransportFamily(bases)` is intended to be such that `bases` is an array 
of ordered bases intended to encode the following.

```math
\\big( (ψ_j)_{j\\in \\mathcal J_1}, \\ldots, (ψ_j)_{j\\in \\mathcal J_n} \\big)
```

From here, 

```julia
T(γ::AbstractArray{<:AbstractArray{<:Real}})
``` 

returns a `TransportMap` type which can be evaluated so that

```julia
(T(γ::AbstractArray{<:AbstractArray{<:Real}})Θ::AbstractArray{<:Real})
```

encodes ``T(Θ; γ)``.

"""
struct TransportFamily{B <: OrderedBasis}
  bases::AbstractArray{B}
end

function Base.display(T::TransportFamily{B}) where B <: OrderedBasis
  println(
  """
  TransportFamily{$B}
  $(join(map(p -> "  J_$(p[1]) = $(p[2])\n", enumerate(T.bases))))
  """)
end

"""
    map = TransportMap(dim, parameter, f)

Wrap a provided function `f` with a `TransportMap` type for the type checking.
In addition, we keep track of the dimension (`dim`) of the vector space on which `f` is defined
and the parameter (`parameter`) of the specific map.
"""
struct TransportMap{B <: OrderedBasis}
  dim::Int
  parameter::AbstractArray{<:AbstractArray{<:Real}}
  f::Function
end

function (F::TransportMap)(theta)::AbstractArray{<:Real}
  if length(theta) !== F.dim
    throw(ArgumentError("$theta must have same length as $gamma"))
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
  $(join(map(p -> "  γ_$(p[1]) = $(p[2])\n", enumerate(F.parameter))))
  """
  )
end

"""
    theta = inverse(tmap::TransportMap, r)

Return ``Θ`` (`theta`) such that ``T(Θ; γ) = r``, where ``T(\\cdot; γ)`` is our transport map 
(`tmap`) and ``r`` is provided by (`r`).
"""
function inverse(tmap::TransportMap, r)::AbstractArray{<:Real}
  throw(ArgumentError("must implement inverse(::$(typeof(tmap)), ::$(typeof(r)))."))
end

"""
    isa(T, TransportFamily) # true
    map = T(gamma)

Given some instance of a transport family `T`, `T(gamma)` returns a specific transport map `
`T(\\cdot; γ)`` (`map`) corresponding to parameter ``γ`` (`gamma`).
"""
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


#---------------------------------------------------------------------------------------------
# specific implementations
#---------------------------------------------------------------------------------------------
include("polynomials.jl")

export hermite, HermiteBasis

"""
    y = hermite(n::Int, x::T) where T <: Real

Evaluate the ``n``-degree (`n`) Hermite polynomial at ``x`` (`x`).

```math
n!\\sum_{m=0}^{[n/2]} \\frac{(-1)^m}{m!(n-2m)!} \\frac{x^{n-2m}}{2^m}
```
"""
function hermite(n::Int, x::T) where T <: Real
  if n == 0
    1
  else
    N = Int(floor(n/2))
    fac = factorial
    s = reduce((s, m) -> s + (-1)^m * x^(n-2m) / (fac(m) * fac(n-2m) * 2^m), 0:N; init=0)
    fac(n) * s
  end
end

"""
    basis = HermiteBasis(vectors)

Encode an ordered basis (`basis`) of polynomials ``(ψ_j)_{j\\in J}`` of the form

```math
ψ_{(j_1,\\ldots,j_n)}(Θ_1, \\ldots, Θ_n) = \\prod_{k=1}^n φ_{j_k}(Θ_k),
```

where ``φ_{j_k}`` is the Hermite polynomial of degree ``j_k``.
The index tuple ``J`` of the polynomials is provided by `vectors` in the natural way.
For example, 

```julia
HermiteBasis([[1, 2, 3], [0, 1, 1, 2]])
```

will encode ``( ψ_{(1,2,3)}, ψ_{(0,1,1,2)} )``.
"""
struct HermiteBasis <: OrderedBasis
  vectors::AbstractArray{<:AbstractArray{<:Int}}
  function HermiteBasis(vectors::AbstractArray{<:AbstractArray{<:Int}})
    if all(vec -> all(x -> x >= 0, vec), vectors)
      new(vectors)
    else
      throw(
        ArgumentError("A `HermiteBasis` must only include vectors with positive components.")
       )
    end
  end
end

Base.length(basis::HermiteBasis) = length(basis.vectors)

function Base.union(basis::HermiteBasis, bases...)
  HermiteBasis(union(basis.vectors, map(b -> b.vectors, bases)...))
end

function (basis::HermiteBasis)(theta::AbstractArray{<:Real})
  if any(vec -> length(vec) > length(theta), basis.vectors)
    throw(ArgumentError("$theta is not in the domain of $basis"))
  else
    map(vec -> prod(map(p -> hermite(p[2], theta[p[1]]), enumerate(vec))), basis.vectors)
  end
end

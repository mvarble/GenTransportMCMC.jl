# create a dummy basis that is easy to track
f11(theta) = theta[1]^2 + 5
f12(theta) = theta[1] - 9
f21(theta) = theta[1]^2 * (theta[2]^3 + 4)
f22(theta) = theta[2]
f23(theta) = theta[1] * theta[2]
f31(theta) = theta[1] * theta[2] * theta[3]

struct TestBasis <: OrderedBasis
  component::Int
end

function (tuple::TestBasis)(theta::AbstractArray{<:Real})
  c = tuple.component
  funcs = c == 1 ? [f11, f12] : (c == 2 ? [f21, f22, f23] : [f31])
  map(f -> f(theta), funcs)
end

Base.length(tuple::TestBasis) = tuple.component == 1 ? 2 : (tuple.component == 2 ? 3 : 1)

@testset "TransportFamily linearity" begin
  T = TransportFamily(map(TestBasis, 1:3))
  for _=1:100
    gam1 = [randn(2), randn(3), randn(1)]
    gam2 = [randn(2), randn(3), randn(1)]
    scale = randn()
    gam3 = @. gam1 + scale * gam2
    for _=1:100
      theta = randn(3)
      @test T(gam1)(theta) + scale * T(gam2)(theta) ≈ T(gam3)(theta)
    end
  end
end

@testset "TransportFamily typing" begin
  T = TransportFamily(map(TestBasis, 1:3))
  @test isa(T, TransportFamily)
  @test isa(T([zeros(2), zeros(3), zeros(1)]), TransportMap)
end

include("polynomials.jl")

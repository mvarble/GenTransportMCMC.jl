@testset "hermite polynomials" begin
  x = -5:0.5:5
  @test hermite.(0, x) ≈ ones(size(x))
  @test hermite.(1, x) ≈ x
  @test hermite.(2, x) ≈ @. x^2 - 1
  @test hermite.(3, x) ≈ @. x^3 - 3 * x
  @test hermite.(4, x) ≈ @. x^4 - 6 * x^2 + 3
  @test hermite.(5, x) ≈ @. x^5 - 10 * x^3 + 15 * x
  @test hermite.(6, x) ≈ @. x^6 - 15 * x^4 + 45 * x^2 - 15
  @test hermite.(7, x) ≈ @. x^7 - 21 * x^5 + 105 * x^3 - 105 * x
  @test hermite.(8, x) ≈ @. x^8 - 28 * x^6 + 210 * x^4 - 420 * x^2 + 105
  @test hermite.(9, x) ≈ @. x^9 - 36 * x^7 + 378 * x^5 - 1260 * x^3 + 945 * x
  @test hermite.(10, x) ≈ @. x^10 - 45 * x^8 + 630 * x^6 - 3150 * x^4 + 4725 * x^2 - 945
end

@testset "HermiteBasis dimensions" begin
  # all have at most length 3, so should not depend higher-order components
  indices = [[0, 1, 2], [3, 4, 1], [1, 1, 1], [0], [1, 2], [5, 8]]
  basis = HermiteBasis(indices)
  @test length(basis) == length(indices)
  for _=1:100
    theta_base = randn(3)
    theta_useless = randn(5)
    theta = [theta_base..., theta_useless...]
    theta_prime = [theta_base..., similar(theta_useless)...]
    @test basis(theta) ≈ basis(theta_prime)
  end
end

function finite_diff(f, epsilon, i)
  function diff(x)
    epsilon_vec = zeros(length(x))
    epsilon_vec[i] = epsilon
    (f(x .+ epsilon_vec)[i] - f(x)[i]) / epsilon
  end
end

@testset "TransportFamily{HermiteBasis} inversion" begin
  # create a typical Hermite family
  T = TransportFamily([
    HermiteBasis([[0], [1], [2], [3]]),
    HermiteBasis([[0, 1], [1, 0], [1, 1]]),
    HermiteBasis([[0, 0, 2], [0, 2, 0], [2, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
  ])
  r = randn(3)
  while true
    # sample a gamma that suggests the map is monotonic
    gamma = [randn(4), randn(3), randn(6)]
    Tgam = T(gamma)
    points = map(_ -> randn(3), 1:1000)
    if !all(i -> all(point -> finite_diff(Tgam, 1e-5, i)(point) > 1e-8, points), 1:3)
      continue
    end

    # with such a gamma, it is very likely the map is invertible: test inversion
    @test Tgam(inverse(Tgam, r)) ≈ r
    break
  end
end

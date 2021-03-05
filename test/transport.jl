@testset "transport map errors" begin
  test = TransportMap{TransportFamily, Real}(0.5)
  @test_throws ErrorException test(1.0)
  @test_throws ErrorException inverse(test, 0.5)
end

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

@testset "polynomial transport parameters" begin
  @test isa([(1, [1, 2, 3, 0, 1], 0.5), (1, [1, 2, 3, 0, 1], 0.5)], PolynomialTransportParameter)
  @test isa([(5, [1, 2, 3, 0, 1], 0.5), (5, [1, 2, 3, 0, 1], 3)], PolynomialTransportParameter)
  @test !isa([(1, [1, 2, 3, 0, 1], 0.5), (1.0, [1, 2, 3, 0, 1], -0.5)], PolynomialTransportParameter)
  @test !isa([(1, [1, 2.0, 3, 0, 1], 0.5), (1, [1, 2, 3, 0, 1], 0.5)], PolynomialTransportParameter)
end

@testset "polynomial transport maps" begin
end

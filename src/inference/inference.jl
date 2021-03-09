export transport_mcmc

"""
    traces, lml_est = transport_mcmc(
      trace,
      kernel,
      family::TransportFamily,
      init_param::AbstractArray{<:AbstractArray{<:Real}},
      num_samples::Int,
      adapt_delay::Int
    )


"""
function transport_mcmc(
  trace,
  kernel,
  family::TransportFamily,
  init_param::AbstractArray{<:AbstractArray{<:Real}},
  num_samples::Int,
  adapt_delay::Int
)
end


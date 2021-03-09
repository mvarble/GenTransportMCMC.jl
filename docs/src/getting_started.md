# Getting Started

## Installation

The package may be installed using the Julia package manager like so.

```
pkg> add https://github.com/mvarble/GenTransportMCMC.jl
```

From here, one may use the `GenTransportMCMC` module like in the example below.

## Example

```julia
using Gen, GenTransportMCMC

# TODO: create toy model

traces, lml_est = transport_mcmc(
  trace,
  kernel,
  family::TransportFamily,
  init_param::AbstractArray{<:AbstractArray{<:Real}},
  num_samples::Int,
  adapt_delay::Int
)
```

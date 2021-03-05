module Example
  # split the model, data, inference, and plotting into separate scripts
  include("model.jl")
  include("data.jl")
  include("inference.jl")
  include("plot.jl")
end

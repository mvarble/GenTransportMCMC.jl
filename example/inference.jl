function vanilla_mcmc(; 
  model_args=_model_args,
  burn_in::Int=100, 
  chain_length::Int=100,
  verbose=false,
)
  samples = similar(1:chain_length, verbose ? Trace : PredatorPreyParameter)
  trace, = generate(predator_prey_model, model_args, data)
  for i=1:(burn_in+chain_length)
    trace, =  mh(trace, select(:predator_prey_parameter))
    if i > burn_in
      samples[i-burn_in] = verbose ? trace : trace[:predator_prey_parameter]
    end
  end
  samples
end

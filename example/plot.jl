using Plots

function render_population(trace::Trace)
  # parse the trace
  parameter = trace[:predator_prey_parameter]
  true_parameter = _trace[:predator_prey_parameter]
  measurements = trace[:measurements]

  # plot the inferred curve
  plot(
    parameter.solution; 
    vars=(1, 2), 
    label="inferred dynamics", 
    alpha=0.5, 
    color=:black,
    title="Predator-Prey population",
    xlabel="prey population",
    ylabel="predator population"
  )

  # plot the true curve
  plot!(
    true_parameter.solution;
    vars=(1, 2), 
    label="true dynamics", 
    alpha=0.5, 
    color=:blue,
  )

  # scatter the measurements
  xs = map(p -> p[1], measurements)
  ys = map(p -> p[2], measurements)
  scatter!(xs, ys; label="measurements", color=:red)
end

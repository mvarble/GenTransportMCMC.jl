function _true_data(model_args)
  P0 = 50.0
  Q0 = 5.0
  r = 0.6
  K = 100.0
  s = 1.2
  a = 25.0
  u = 0.5
  v = 0.3
  parameter = PredatorPreyParameter(P0, Q0, r, K, s, a, u, v, model_args[3:4]..., nothing)
  stable!(parameter)
  generate(predator_prey_model, model_args)[1]
end

_model_args = (0.1, 110.0, 0.0, 50.0, 20, 1.0)
_trace = _true_data(_model_args)
data = choicemap(map(p -> (:measurements => p[1] => :z, p[2]), enumerate(_trace[:measurements]))...)

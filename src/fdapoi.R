fit_model = function(y, X, family) {
  model = fdapoi::FUN_PoI_BIC(y, X, family=family)
  return(model)
}


plot_model = function(y, X, model, path) {
  p = nrow(X)
  t_grid = (seq(1, p) - 1) / (p-1)
  
  png(path)
  fdapoi:::PoIMaker(model$k.opt, X, y, plotting=TRUE)
  axis(side = 3, at = t_grid[model$tau.ind.hat[1]], 
       labels = expression(hat(tau)[1]), line = 0)
  axis(side = 3, at = t_grid[model$tau.ind.hat[2]], 
       labels = expression(hat(tau)[2]), line = 0)
  dev.off()
}


read_data = function(directory, id) {
  np = reticulate::import("numpy")
  directory = ifelse(directory=="", directory, paste0(directory,"/"))
  y = np$load(paste0(directory, id, "-y.npy"))
  X = np$load(paste0(directory, id, "-X.npy"))
  out = list(y=y, X=X)
  return(out)
}


write_model = function(model, path) {
  beta = unname(c(model$beta0.hat, model$beta.hat))
  locations = model$tau.ind.hat
  k_opt = model$k.opt
  
  out = list(beta=beta, locations=locations, k_opt=k_opt)
  yaml::write_yaml(out, path)
}

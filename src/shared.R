# shared functions

## method


fit_model = function(y, X, order, family) {
  model = fdapoi::estimate_poi_validated(y, X, order=order, family=family)
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


extract_coefficients = function(model, path=NULL) {
  beta = unname(c(model$beta0.hat, model$beta.hat))
  locations = model$tau.ind.hat
  k_opt = model$k.opt
  
  out = list(beta=beta, locations=locations, k_opt=k_opt)
  if (is.null(path)) {
    return(out)
  } else {
    yaml::write_yaml(out, path)
  }
}


## monte carlo

read_n_sim = function() {
  config = yaml::read_yaml(file.path(SRC, "config_monte_carlo.yaml"))  
  return(config[["n_sim"]])  
}


replace_comma = function(str) {
  s = strsplit(str, "")[[1]]
  n = length(s)
  if (s[n - 1] == ",") {
    s[n - 1] = ""
  }
  
  s = paste0(s, collapse="")
  return(s)
}


string_to_object = function(str) {
  object = eval(parse(text=str))
  return(object)
}


df_row_to_kwargs = function(row) {
  kwargs = row
  kwargs$n_periods = as.integer(kwargs$n_periods)
  kwargs$n_samples = as.integer(kwargs$n_samples)
  kwargs$n_points = as.integer(kwargs$n_points)
  kwargs = as.list(kwargs)
  kwargs$beta = unlist(kwargs$beta)
  return(kwargs)
}


monte_carlo = function(kwargs_df, n_sim) {
  kwargs_list = apply(kwargs_df, 1, df_row_to_kwargs)
  to_parallelize = function(kwargs) monte_carlo_inner(kwargs, n_sim)
  results = pbapply::pblapply(kwargs_list, to_parallelize)
  return(results)
}


monte_carlo_inner = function(kwargs, n_sim) {
  
  order = kwargs$order
  kwargs$order = NULL
  
  kwargs$beta = string_to_object(kwargs$beta)
  kwargs$kernel_kwargs = string_to_object(kwargs$kernel_kwargs)
    
  locations = c()
  for (k in 1:n_sim) {
    kwargs$seed = k
    
    data = do.call(simulate_model, kwargs)
    
    model = fit_model(y=data[[1]], X=data[[2]], order, family="gaussian")
    
    coeff = extract_coefficients(model)
    
    .locations = coeff$locations
    locations = c(locations, .locations)
  }
  out = tibble::tibble(locations)
  out = dplyr::count(out, locations)
  out = dplyr::rename(out, count=n)
  return(out)
}


write_results = function(monte_carlo_results, BLD) {
  for (k in 1:length(monte_carlo_results)) {
    path = file.path(BLD, "monte_carlo", paste0("result", k-1, ".csv"))
    readr::write_csv(monte_carlo_results[[k]], path)
  }
  return(NULL)
}

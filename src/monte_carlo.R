#!/usr/bin/env Rscript

reticulate::use_condaenv("topics-metrics", required=TRUE)

args = commandArgs(trailingOnly=TRUE)
SRC = args[1]
BLD = args[2]

simulate = reticulate::import_from_path("simulate", SRC)
simulate_model = simulate$simulate_model

library("glmulti")

source(file.path(SRC, "shared.R"))  # exports fit_model, extract_coefficients,
# read_n_sim, clean_beta, df_row_to_kwargs, monte_carlo, monte_carlo_inner


## main

kwargs_df = read.csv(file.path(BLD, "monte_carlo/kwargs.csv"))
kwargs_df$beta = lapply(as.character(kwargs_df$beta), clean_beta)

n_sim = read_n_sim()

results = monte_carlo(kwargs_df, n_sim)

write_results(results, BLD)

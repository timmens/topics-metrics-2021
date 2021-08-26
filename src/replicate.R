#!/usr/bin/env Rscript

reticulate::use_condaenv("topics-metrics", required=TRUE)

## imports
library("glmulti")

args = commandArgs(trailingOnly=TRUE)

SRC = args[1]

source(file.path(SRC, "shared.R"))  # exports fit_model, extract_coefficients

## main


data = read.csv(args[2])

y = data[[1]]
X = t(as.matrix(data[-1]))

model = fit_model(y, X, order=2, family="binomial")

extract_coefficients(model, path=args[3])  # writes to file

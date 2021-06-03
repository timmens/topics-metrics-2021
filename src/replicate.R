#!/usr/bin/env Rscript

library("fdapoi")
library("glmulti")

source("fdapoi.R")


args = commandArgs(trailingOnly=TRUE)

data = read.csv(args[1])

y = data[[1]]
X = t(as.matrix(data[-1]))

model = fit_model(y, X, family="binomial")

write_model(model, args[2])

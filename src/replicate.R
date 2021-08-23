#!/usr/bin/env Rscript

reticulate::use_condaenv("topics-metrics", required=TRUE)

## imports
library("glmulti")

args = commandArgs(trailingOnly=TRUE)

# SRC = args[1]
SRC = "/home/tim/sciebo/phd/topics-metrics/topics-metrics-2021/src/"
setwd(SRC)

source(file.path(SRC, "shared.R"))  # exports fit_model, extract_coefficients

## main


# data = read.csv(args[2])
data = read.csv("data/emotion_rating.csv")

y = data[[1]]
X = t(as.matrix(data[-1]))

model = fit_model(y, X, family="binomial")

# extract_coefficients(model, path=args[3])  # writes to file
extract_coefficients(model)





#####  TESTING  #####

additional_covariates=NULL
order=4
a=0
b=1
n_poi_max=10
show_progress=FALSE
k.seq=NULL
info_crit="bic"
standardize=FALSE
center=FALSE
family="binomial"
nbasis.max=0
glmulti_method="h"



source("../../project/fdapoi/fdapoi/R/criterion.R")
source("../../project/fdapoi/fdapoi/R/estimate.R")
source("../../project/fdapoi/fdapoi/R/shared.R")
source("../../project/fdapoi/fdapoi/R/simulate.R")

m = estimate_poi_bic_validated(y, X, order=2)
extract_coefficients(m)

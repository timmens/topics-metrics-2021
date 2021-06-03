## Topics in Econometrics and Statistics

*University of Bonn, Summer Term 2021, Tim Mensinger*


> This repository contains material (slides, notes, codes) created for the project in
> the class *topics in econometrics and statistics*.


#### Running the Project

I use [pytask](https://github.com/pytask-dev/pytask) to build the project. To run the
project you need to create the correct Python/R environment. A simple way is to run

```zsh
$ conda env create -f environment.yml
$ conda activate topics-metrics
$ pip install -e .
```

in a terminal and afterwards

```R

install.packages("glmulti")
devtools::install_github("lidom/fdapoi/fdapoi", dependencies=FALSE)

```

in the R console. Then the project can be build by running ``pytask`` in the terminal
when one is located a subfolder of the project root.

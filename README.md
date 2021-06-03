## Topics in Econometrics and Statistics

*University of Bonn, Summer Term 2021*

Author: Tim Mensinger


> This repository contains material (slides, notes, codes) created for the project in
> the topics class in econometrics and statistics.


#### Environment

To get everything up and running install
[miniconda](https://docs.conda.io/en/latest/miniconda.html) and in some terminal
emulator run

```zsh 

$ conda env create -f environment.yml $ conda activate topics-metrics
$ pip install -e .

```

where ``pip install -e .`` has to be run only once!

Afterwards we still have to install the package
[fdapoi](https://github.com/lidom/fdapoi).  To do this continue in the above terminal
session and type ``R`` to open an R terminal.  Then run

```R install.packages("glmulti") devtools::install_github("lidom/fdapoi/fdapoi",
dependencies=FALSE) ```

Note that in my case (Linux OS) I first had to run ``Sys.setenv(TAR = "/bin/tar")`` in
the R session.

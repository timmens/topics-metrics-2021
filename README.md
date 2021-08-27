## Topics in Econometrics and Statistics

*University of Bonn, Summer Term 2021, Tim Mensinger*


> This repository contains material (slides, notes, codes) created for the project in
> the class *topics in econometrics and statistics*.


----


#### PDF Files

To view the final PDF documents simply click the button:


| <!-- -->  | <!-- --> | <!-- --> | <!-- --> |
|-----------|----------|----------|----------|
| Essay | <a href="https://nbviewer.jupyter.org/github/timmens/topics-metrics-2021/blob/main/manuscript.pdf"  target="_parent"><img align="center" src="https://www.iconpacks.net/icons/2/free-pdf-file-icon-2614-thumb.png" height=50></a> | Presentation | <a href="https://nbviewer.jupyter.org/github/timmens/topics-metrics-2021/blob/main/presentation.pdf"  target="_parent"><img align="center" src="https://www.iconpacks.net/icons/2/free-pdf-file-icon-2614-thumb.png" height=50></a> |


----


#### Visualizations

###### Gaussian Process Visualization

The visualization code is hosted here:
[https://github.com/timmens/gp-visualization](https://github.com/timmens/gp-visualization).

###### Centered Difference Visualization

The (second) centered difference visualization of population and estimated quantities is
done using [bokeh](https://docs.bokeh.org/en/latest/index.html). Script and notebook are
found in ``src/bokeh``.


---- 


#### Building the Project

Here I use the wonderful [pytask](https://github.com/pytask-dev/pytask) build system.
Before building you must create an appropriate Python / R environment. All relevant
packages are listed in ``environment.yml``. I reccomend using
[conda](https://docs.conda.io/en/latest/miniconda.html) to create the environment. In a
terminal shell run

```zsh
# cd into root of project
$ conda env create -f environment.yml
$ conda activate topics-metrics
$ pip install -e .
```

This installs all Python packages and core R functionality. To install the missing R
packages open an R terminal by typing ``R`` in the terminal and run

```R
install.packages("glmulti")
devtools::install_github("lidom/fdapoi/fdapoi", dependencies=FALSE)
```

> *Note that there may be some issues when installing the R dependencies, but they can
> be operating system specific and are usally easy to solve.* In my case I had to run
> ``Sys.setenv(TAR = "/bin/tar")`` in the R shell.

*Finally*, to actually **run the project** do

```zsh
# cd into root of project
$ pytask
```

The results can be found in the newly generated folder ``bld``.

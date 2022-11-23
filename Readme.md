## Tonale Winter School on Cosmology 2022 

# Tutorial on Gravitational Lensing

Welcome to the exercises on Gravitational Lensing! In this tutorial we will see how gravitational lensing works in practice through a series of coding exercises. In my opinion, this is a very efficient way to gain familiarity with the subject. It is much easier to understand theory by coding some examples and visualizing the results. For people like me it is also much more fun!

You don't need to be familiar with python to use the notebooks. They are thought to be easy to use for anyone. The python addicted can try to code the examples from scratch if they want. If you find some bugs or have suggestions on how to improve the codes, please let me know!

## What do you need?
To run the notebooks, you need python installed on your computer. There are several distributions you may try. I usually work with the  [**Anaconda**](https://www.anaconda.com) distribution. If you want some lighter option, you may try [**Miniconda**](https://docs.conda.io/en/latest/miniconda.html) or [**Miniforge**](https://github.com/conda-forge/miniforge).

Once you have installed python, I suggest you create an environment for this tutorial. Open a terminal and type

`python create -n tonale python=3.8`

This will create a python 3.8 environment called `tonale`, where we will download all needed packages. The notebooks should work also with other python versions, but be aware that I tested them with version 3.8. 

To activate the environment, type 

`conda activate tonale`

from the command line.

You will need to install the packages that are listed in the file `requirememts.txt`. You just need to execute the command 

`pip install -r requirements`

and everything should work smoothly, **if you are lucky** ;-). If not, you'll need to manually install some package, either using `conda install <package name>` or `pip install <package name>`.
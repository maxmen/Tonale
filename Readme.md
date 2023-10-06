## Tonale Winter School on Cosmology 2022 

# Tutorial on Gravitational Lensing

Welcome to the exercises on Gravitational Lensing! In this tutorial, we will see how gravitational lensing works in practice through a series of coding exercises. This is a very efficient way to gain familiarity with the subject. It is much easier to understand the theory by coding examples and visualizing the results. For people like me, it is also much more fun!

You don't need to be familiar with python to use the notebooks. They are thought to be easy to use for anyone. The Python-addicted can try to code the examples from scratch. If you find some bugs or have suggestions on improving the codes, please let me know!

## What do you need?
To run the notebooks, you need Python installed on your computer. There are several distributions you may try. I usually work with the  [**Anaconda**](https://www.anaconda.com) distribution. If you want a lighter option, you may try [**Miniconda**](https://docs.conda.io/en/latest/miniconda.html) or [**Miniforge**](https://github.com/conda-forge/miniforge).

Once you have installed Python, I suggest you create an environment for this tutorial. Open a terminal and type:

`conda create -n tonale python=3.8`

This instruction will create a Python 3.8 environment called `tonale`, where we will download all needed packages. The notebooks should work also with other Python versions, but I want you to know that I tested them with version 3.8. 

To activate the environment, type 

`conda activate tonale`

from the command line.

You can go ahead with downloading the materials in this GitHub repository. You have two options:

* clone the repository by running the command
`git clone https://github.com/maxmen/Tonale.git` in your terminal. You will find the materials in a directory called `Tonale.` 
* download the content as a ZIP file. Click on the green button ```Code``` above. Then click on "Download ZIP". Once the file has been downloaded, unzip it in a directory of your choice. You will find the materials in a sub-directory called `Tonale-master.`

> **Warning**
> The files in the repository will be updated during the school. Stay tuned!

You must install the packages listed in the file `requirememts.txt.` You need to execute the command 

`pip install -r requirements.txt`

and everything should work smoothly, **if you are lucky** ;-). If not, you'll need to manually install some packages, either using `conda install <package name>` or `pip install <package name>`.

At this point, you should be able to open a Jupyter Notebook and use it. I like to work with [Visual Studio Code](https://code.visualstudio.com/), but you may also open the notebooks in your browser by running 

`jupyter lab <notebook name>`

> **Warning**
> You may also open the notebooks with 
>
>`jupyter notebook <notebook name>`
>
> but you may need to follow the instructions below to run the notebooks with interactive plots:
>```
>conda install -y nodejs
>pip install --upgrade jupyterlab
>jupyter labextension install @jupyter-widgets/jupyterlab-manager
>jupyter labextension install jupyter-matplotlib
>jupyter nbextension enable --py widgetsnbextension
>```
> Then restart `jupyter`.

You can also visualize the notebooks by clicking the notebook name on the [GitHub page](https://github.com/maxmen/Tonale). However, you won't be able to execute the codes this way.

## Python mini-tutorial
I wrote a basic mini-tutorial on Python in the [`python_tutorial.ipynb`](python_tutorial.ipynb) jupyter notebook. This should help those of you who need to become more familiar with Python.

## Additional resources
For additional reading and Python examples, check out my book [*Introduction to Gravitational Lensing with python examples*](https://link.springer.com/book/10.1007/978-3-030-73582-1).

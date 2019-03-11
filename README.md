DUTI Implementation in Python
=============================

## Data set

Income data set is already included in this repository.

You could download spam data from [Colab for Influence Function](https://worksheets.codalab.org/bundles/0x19a9fd9ebce64ea3940dee08fb37906c/)

## Files

    ├── exp_income.ipynb (Experiment for fixing data of income)
    ├── exp_sincurve.ipynb (Experiment for fixing sin(x))
    ├── interactive_mnist_duti.ipynb (Interactive UI for fixing data of MNIST using traditional DUTI)
    ├── interactive_mnist_iduti.ipynb (Interactive UI for fixing data of MNIST using iDUTI)
    ├── fig_input_checking.ipynb (Plotting the figure of MNIST input checking)
    ├── fig_spam_input_checking.ipynb (Plotting the figure of spam input checking)
    ├── exp_auto_fix_spam.ipynb (Experiment for automatically fixing spam)
    ├── influence (Library directly use from Influence Function)
    ├── duti (Implemented Library of DUTI and its experiments)
    │   ├── duti.py
    │   ├── experiments.py
    │   └── __init__.py
    ├── data (Supporting testset)
    │   ├── adult.csv
    │   └── spam
    └── utils (Util functions)
        ├── experiment_mnist.py
        ├── __init__.py
        ├── load_mnist.py
        ├── load_spam.py
        └── mnist_devel.ipynb

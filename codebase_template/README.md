# Generative Adversarial Network MQP 2022-2023 Codebase Style Guide

## Content
* [General Naming Convention](#general-naming-convention)
    * [File Names](#file-names)
    * [Global Variable](#global-variable)
    * [Local Variable](#local-variable)
* [Directories and Files](#directories-and-files)
    * [Directory Structure](#directory-structure)
    * [General Formatting](#general-formatting)
    * [Main File](#main-file)
    * [Parsing File](#parsing-file)
    * [Class File](#class-file)
    * [Module File](#module-file)
    * [Variable File](#variable-file)
* [Function Structure](#function-structure)
* [Additional Resources](#additional-resources)


## General Naming Convention
This section outlines the general naming convention that we will use in this codebase.

### File Names
For file names, we will use a snake case with lowercase naming convention as follow:
* `main.py`
* `module.py`
* `class.py`.

### Class Names
For class names, this is a special case where we will not use a snake case but rather an upper camel case.
* `Name`
* `MultiWordClassName`

### Global Variables
For global variables, we will use a snake case with uppercase naming convention as follow:
* `VAR`
* `MULTI_WORD_VAR`

### Local Variables
For the local variables, we will use a snake case with lowercase naming convention as follow:
* `var`
* `multi_word_var`


## Directories and Files
This section outlines the directory and file structures.

### Directory Structure
For the directory structure, we will split our organization into two categories:

#### Small Projects
For the small project, we will leave all the python files within the main directory as follow:
```
small_project
├── .gitignore
├── README.md
├── class.py
├── main.py
├── module.py
├── utils.py
└── vars.py
```

#### Large Projects
For the large project, we will divide files into source code directory `src` and testing code directory `tst`. The structure is as follow:
```
large_project
├── .gitignore
├── README.md
├── src
│   ├── class.py
│   ├── main.py
│   ├── module.py
│   ├── utils.py
│   └── vars.py
└── tst
├── class_test.py
├── module_test.py
└── utils_test.py
```

### General Formatting
This section outline the general formatting that will be used in each python files.

#### Imports
All imports will be listed at the top of each python files.
When sourcing native, internal and external libraries, we will organize them listing all native imports follow by internal imports and then external imports.
Each section will be ordered alphabetically Each section will be ordered alphabetically.

* Generally, we will first list all `import` statements before `from _ import _` statements.
* Native imports: These imports are shipped along with the python interpreter. This includes packages such as `argparse`, `os`, `sys`, etc.
* Internal imports: These imports are our private library sourced from local files. This includes modules and class.
    * For small projects, these will be import normally as `import class` or `import module`.
    * For large projects, these will have to be sources with path as such `from src.class import class` or `from src.module import module`.
* External imports: These imports are external libraries that we download. This includes packages such as `numpy`, `pandas`, `torch`.

Note: Please refrain from importing files by doing `from file import *`. This is to negate any variable naming conflict and keep with best practices.

Example:
```python
# Native libraries
import argparse
import os
import sys
# Internal libraries
import class
import module
from src.class import class
from src.module import module
# External libraries
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
```

#### File Indentations
To keep the same convention throughout our codebase, we will use 4-spaces indentation convention

Example:
```python
def some_function(param):
    """[TODO:description]

    Args:
        param ([TODO:parameter]): [TODO:description]
    """
    # some comments
    arr = param.to_list()

    # more comments
    for i in arr:
      print(i)
```

#### Function Spacings
To make the code cleaner and easier to read, at the end of each function, leave two blank lines before starting defining a new function.

Example:
```python
def first_function(param1, param2):
    """[TODO:description]

    Args:
        param1 ([TODO:parameter]): [TODO:description]
        param2 ([TODO:parameter]): [TODO:description]
    """
    pass


def second_function(param1, param2):
    """[TODO:description]

    Args:
        param1 ([TODO:parameter]): [TODO:description]
        param2 ([TODO:parameter]): [TODO:description]
    """
    pass
```

### Main File
This section outlines the template for `main.py` file.
To keep the code clean and concise, we will only include the main part of the script in the main function.
This will include only `if __name__ == '__main__':` statement and `def main()` function.
Generally, we would like to modularize our python files, however, since argument parsing and main functions are closely related, we will put out argument parsing function in `main.py` as well.

Example:
```python
# Native libraries
import argparse


def main():
    """[TODO:description]
    """
    args = handle_arguments()


def handle_arguments():
    """[TODO:description]

    Returns:
        [TODO:return]: [TODO:description]
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-a', '--argument',
                        action='store',
                        required=False,
                        choices=["choice1", "choice2"],
                        default=DEFAULT_LOG_LEVEL,
                        help='Logging verbosity. Default: %(default)s')

    arguments = parser.parse_args()

    return arguments


if __init__ == '__main__':
    main()
```

### Parsing File
This section outlines the template for `parsing.py` file that will be used specifically for argument parsing.
If our argument parsing method require more custom argument handling function, we will put them in `parsing.py` as opposed to `main.py`.

Example:
```python
# Native libraries
import argparse


def handle_arguments():
    """[TODO:description]

    Returns:
        [TODO:return]: [TODO:description]
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-a', '--argument',
                        action='store',
                        required=False,
                        choices=["choice1", "choice2"],
                        default=DEFAULT_LOG_LEVEL,
                        help='Logging verbosity. Default: %(default)s')

    arguments = parser.parse_args()

    return arguments
```

### Class File
This section outlines the template for `class.py` file that will be used for python classes.
In each python classes, we will require a doctrings for each class and their class methods.

Example:
```python
class ExampleClass():
    """[TODO:description]
    """


    def class_method(param):
        """[TODO:description]

        Args:
            param ([TODO:parameter]): [TODO:description]
        """
        pass
```

### Module File
This section outlines the template for `module.py` file that will be used to any addition python functions.
We define module as any group of functions that relate to each other.
For example, if a multiple functions are used to evaluated a GAN performance, we can put them all in `evaluate.py`.
Utility functions that do not fit anywhere can be put into `utils.py`. In which case, the structure of said file will also follow this section structure.

Example:
```python
# Native libraries
import os
# Internal libraries
import vars
# External libraries
import numpy as np
import pandas as pd
import torch


def module_function1(param):
    """[TODO:description]

    Args:
        param ([TODO:parameter]): [TODO:description]
    """
    pass


def module_function2(param):
    """[TODO:description]

    Args:
        param ([TODO:parameter]): [TODO:description]
    """
    pass
```

### Variable File
This section  outlines the template for `vars.py` file that will be used to store all global variables.
Although we don't often use global variables in python, if we do require them, we will store all of them in `vars.py`.
To reference global variables, you will have to `import vars` in your python file and reference them by doing `vars.YOUR_VARIABLE`

Note: Please refrain from importing variable files by doing `from vars import *`. This is to negate any variable naming conflict and keep with best practices.

Example:
```python
# Argument parsing
DEFAULT_ARG_STRING = 'default'
DEFAULT_ARG_NUMBER = 0
# Global variables
GLOBAL_STRING = 'global'
GLOBAL_NUMBER = 1
GLOBAL_LIST = []
```


## Function Structure
This section outlines the template for function structures. This includes comments and docstring conventions.

### Comments
For comments, try your best to comment each block of code with what that block does at a high level.
If there is any oneline list comprehension or lambda functions, please explain what that line does as clear as possible.
Moreover, please make use of any comment tags such as `NOTE:`, `TODO:`. `WARNING:`, `ERROR:`, `HACK:`, `PERF:` etc. whenever possible.
This is not required but it would be nice if there is a certain note you'd like to tell another person who willl be reading the code later on.

Example:
```python
# NOTE: This is to write a short note about the function, code, or any complicated oneliner.

# TODO: This is to temporally indicate that there is something that needs to be done here later.

# WARNING: This is to warn someone of an expected or unexpected behaviors. Try your best to fix these if possible.

# ERROR: This is to indicate that there is a known error at the current section of the code.

# HACK: This is to indicate any weird hack that you've done but may not tally understand why. Just a warning if the code breaks, this may be it.

# PERF: This is to indicate that the currently code block is fully optimized. Rarely used but the more the merrier I guess.
```

### Docstrings
Since our project utilizes a lot of external libraries such as pytorch, we will not include static typing into python functions.
Rather, we will declare our types in the docstrings of each function. You are responsible to correctly indicate the type of each variable in the docstrings.
For our docstrings, we will utilize [google docstring](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) convention.

Example:
```python
def func_without_arg_or_return():
    """[TODO:description]
    """
    pass


def func_without_arg():
    """[TODO:description]

    Returns:
        result ([TODO:return]): [TODO:description]
    """
    # initialize variables
    result = 0

    return result


def func_without_return(param):
    """[TODO:description]

    Args:
        param ([TODO:parameter]): [TODO:description]
    """
    pass

def func_with_arg_and_return(param):
    """[TODO:description]

    Args:
        param ([TODO:parameter]): [TODO:description]

    Returns:
        arr ([TODO:return]): [TODO:description]
    """
    # initialize variables
    arr = param[0]

    return arr


def func_with_multiple_arg(param1, param2):
    """[TODO:description]

    Args:
        param1 ([TODO:parameter]): [TODO:description]
        param2 ([TODO:parameter]): [TODO:description]
    """
    pass


def func_with_mulitple_return(param):
    """[TODO:description]

    Args:
        param ([TODO:parameter]): [TODO:description]

    Returns:
        arr0 ([TODO:return]): [TODO:description]
        arr1 ([TODO:return]): [TODO:description]
    """
    # initialize variables
    arr0 = param[0]
    arr1 = param[1]

    return arr0, arr1
```


## Additional Resources
Most of these conventions are taken from official best practices in the industry.
The formatting of the file can be automatically done with [autopep8](https://pypi.org/project/autopep8/) or [black](https://pypi.org/project/black/).

# Project goals

This Python package intends to provide a simple set of tools to aid people studying thermodynamics of materials. Mostly constant-composition materials, that is, because solution models are not supported.

Have you ever dreamt of the possibility of making your own personal small thermodynamic database, where thermodynamic properties could be described by and stored as any arbitrary symbolic functions?

Currently, this project aims to:

- provide a predefined set of equations that are commonly used to describe thermodynamic functions, such as heat capacity, Gibbs energy, etc.;
- make the said functions available in both symbolic and numeric formats;
- provide a Python interface for attaching the functions to the respective Phases and packing related Phases into Compounds...
- ...so as to make calculating, tabulating and plotting the properties of these Compounds as easy as calling a method of a Python object;
- provide a possibility of storing a set of Compounds in a file to facilitate the subsequent calculations.

## A word of warning

This project is in its **testing** phase of development. At this point in time, new features can be added, old - removed without warnings, and I don't yet have a slightest idea of how stable the code is. I haven't yet finished writing the basic unit tests. Use at your own peril!

## Usage

Please see the Jupyter notebooks in the `examples` folder. The examples are stored along with the full output, which is not very git-friendly, but at least there's no need for you to download and run anything so as to check out what the package can do, and there's also no need for me to think of storing the examples somewhere else.

The project is somewhat documented (meaning "there are `__doc__`-functions for every important thing in the source code).

## Installation

You can use it without installing, just download the whole thing and run the examples in the `examples` folder.

To install for subsequent editing (in the developing mode) from a local folder, navigate into the package folder and run

```
pip install -e .
```

I think that it is too soon (if ever) for this project to be added to PyPI. However, since the default Python package manager `pip` supports [installing via the version control systems](https://pip.pypa.io/en/stable/topics/vcs-support/), you can use this:

```
pip install "simplethermodynamics @ git+https://github.com/voffch/simplethermodynamics@master"
```

This behemoth of a command, if prefixed with `!`, can be used in [Google Colab](https://colab.research.google.com), so you can go and try this package there with no need for the locally installed Python. The examples should be using it as well.

The examples are not installed along with the package and should be downloaded separately.

## Uninstall

...and so when you finally understand that this is totally unusable, just run

```
pip uninstall simplethermodynamics
```
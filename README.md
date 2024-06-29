# pyconstrainedpacker

This app is used to solve the following problem:

We are given a number of customers ("groups"), that each have a demand for some commodity.
The supplier has the commodity in a number of package sizes that cannot be divided further
For each package size we have a limited quantity available.

How can we distribute the packages among the customers such that the overall
deviation from demands is minimal (or within acceptable bounds).


# Installation

This package is managed with poetry. To install the package, please install
poetry, `pip install poetry`, and afterwards run inside the git repository:

```
poetry install
```


To run the realistic example, change into your virtual environment:

```
poetry shell
```

and run

```
pytest -s tests/test_realistic_run.py
```


# Random Forest
A classic random forest implementation for Machine Learning class at UFRGS.

## Evaluation
```
$ python3 main.py -h
usage: main.py [-h] -d {credit_g,spambase,vertebra_column,wine} -n NTREES
               [-f N_FOLDS] [-r N_REPEATS]

optional arguments:
  -h, --help            show this help message and exit
  -d {credit_g,spambase,vertebra_column,wine}, --dataset {credit_g,spambase,vertebra_column,wine}
                        Dataset's name.
  -n NTREES, --ntrees NTREES
                        Number of trees to be generated on random forest. It
                        can be range(...) or an integer.
  -f N_FOLDS, --folds N_FOLDS
                        Number of folds to use in cross validation.
  -r N_REPEATS, --repeat N_REPEATS
                        Number of cross validation iterations.
```

#### Example
To evaluate a model with `credit_g` dataset using between 30 and 40 (inclusive) trees, just run:
```
$ python3 main.py -d credit_g -n "range(30, 41)"
```
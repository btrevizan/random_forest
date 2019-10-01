from pandas import read_csv, DataFrame
from matplotlib import pyplot as plt
from src.utils import datasets
from glob import glob
import seaborn as sns
import numpy as np
import os


results_path = 'tests/results'
graphics_path = 'tests/graphics'

with_path = os.path.join(results_path, 'with_cuts')
without_path = os.path.join(results_path, 'without_cuts')

columns = ['Dataset', 'Mean F1-Score', 'Standard Deviation', 'Number of Trees',
           'Number of Folds', 'Number of Repeats', 'Elapsed Time (seconds)', 'With Pruning']

sns.set_style('whitegrid')

for dataset in set(datasets) - {'benchmark'}:
    data = {columns[1]: [], columns[2]: [], columns[3]: [], columns[7]: []}
    dt_with_path = [f for f in glob(os.path.join(with_path, dataset, '*.csv')) if 'summary.csv' not in f]
    dt_without_path = [f for f in glob(os.path.join(without_path, dataset, '*.csv')) if 'summary.csv' not in f]

    for path in dt_with_path:
        scores = read_csv(path, header=0, index_col=None).values
        data[columns[1]] += [np.mean(scores.flatten())]
        data[columns[2]] += [np.std(scores.flatten())]
        data[columns[3]] += [int(path.split('/')[-1][:-4])]
        data[columns[7]] += ['Yes']

    for path in dt_without_path:
        scores = read_csv(path, header=0, index_col=None).values
        data[columns[1]] += [np.mean(scores.flatten())]
        data[columns[2]] += [np.std(scores.flatten())]
        data[columns[3]] += [int(path.split('/')[-1][:-4])]
        data[columns[7]] += ['Yes']

    data = DataFrame(data)

    ax = sns.scatterplot(x=columns[3], y=columns[1], hue=columns[7], size=columns[2], data=data, palette='winter')
    ax.legend(loc='lower right')
    # ax.set_title(dataset)

    graphic_path = os.path.join(graphics_path, dataset + '.pdf')
    plt.savefig(graphic_path, dpi=360, pad_inches=0.1, bbox_inches='tight')
    plt.close()

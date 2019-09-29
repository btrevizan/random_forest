from matplotlib import pyplot as plt
from pandas import read_csv
import seaborn as sns
from glob import glob
import os


results_path = 'tests/results'
graphics_path = 'tests/graphics'

datasets_path = glob(os.path.join(results_path, 'without_cuts/*'))

columns = ['Dataset', 'Mean F1-Score', 'Standard Deviation', 'Number of Trees',
           'Number of Folds', 'Number of Repeats', 'Elapsed Time (seconds)']

sns.set_style('whitegrid')


for dataset_path in datasets_path:
    summary_path = os.path.join(dataset_path, 'summary.csv')

    summary = read_csv(summary_path, header=0, index_col=None)
    summary.columns = columns

    ax = sns.scatterplot(x='Number of Trees', y='Mean F1-Score',
                         hue='Standard Deviation', size='Standard Deviation',
                         data=summary, palette='winter')

    ax.legend(loc='lower right')

    dt_name = dataset_path.split('/')[-1]
    # ax.set_title(dt_name)

    graphic_path = os.path.join(graphics_path, dt_name + '.pdf')
    plt.savefig(graphic_path, dpi=360, pad_inches=0.1, bbox_inches='tight')
    plt.close()

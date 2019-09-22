from src.random_forest import RandomForest
import src.utils as utils
from time import time
import numpy as np
import graphviz
import argparse
import os


def main(args):
    start = int(time())

    seed = 14
    ntrees_param = eval(args.ntrees)

    results_path = 'tests/results'
    dataset_results_path = os.path.join(results_path, args.dataset)

    os.system("mkdir {} >> /dev/null".format(dataset_results_path))

    if type(ntrees_param) is not range:
        ntrees = int(ntrees_param)
        ntrees_param = range(ntrees, ntrees + 1)

    print("Running experiments with {} dataset...".format(args.dataset))
    for ntrees in ntrees_param:
        print("\tNumber of trees: {}".format(ntrees))
        run(args.dataset, seed, ntrees, args.n_folds, args.n_repeats, dataset_results_path)

    stop = int(time())
    print("Total time elapsed: {}".format(readable_time(start, stop)))
    print("Finish.")


def run(dataset, seed, ntrees, n_folds, n_repeats, dataset_results_path):
    start = int(time())

    random_state = np.random.RandomState(seed)
    numerical_attributes, attr_names, attr_values, x, y = utils.load(dataset)
    random_forest = RandomForest(ntrees, random_state, numerical_attributes, attr_values)

    results = []
    for i in range(n_repeats):
        print("\t\t#{} {}-fold CV iteration".format(i, n_folds))
        results += utils.cross_validate(random_forest, x, y, n_folds, random_state)

    stop = int(time())
    print("\t\tSaving results...", end=" ")

    results_file_path = os.path.join(dataset_results_path, '{}.csv'.format(ntrees))
    with open(results_file_path, 'w') as file:
        file.write('f1_score\n')

        for f1_score in results:
            file.write("{}\n".format(f1_score))

    time_file_path = os.path.join(dataset_results_path, '../summary.csv')
    with open(time_file_path, 'a') as file:
        file.write('{},{},{},{},{},{},{}\n'.format(dataset,
                                                   np.mean(results),
                                                   np.std(results),
                                                   ntrees,
                                                   n_folds,
                                                   n_repeats,
                                                   stop - start))

    # for i in range(2):
    #     dot = graphviz.Digraph(name=str(i))
    #     random_forest.trees[i].get_graph(dot, attr_names=attr_names)
    #     dot.render(cleanup=True)

    print("Done")


    print("\t\tTime elapsed: {}".format(readable_time(start, stop)))


def readable_time(start, stop):
    interval = stop - start
    hours = interval // (60 * 60)
    minutes = (interval // 60) - (hours * 60)
    seconds = interval - (hours * 60 * 60 + minutes * 60)
    return "{}h{}min{}s".format(hours, minutes, seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset',
                        required=True,
                        dest='dataset',
                        choices=utils.datasets,
                        help="Dataset's name.")

    parser.add_argument('-n', '--ntrees',
                        required=True,
                        dest='ntrees',
                        help="Number of trees to be generated on random forest. It can be range(...) or an integer.")

    parser.add_argument('-f', '--folds',
                        required=False,
                        default=10,
                        type=int,
                        dest='n_folds',
                        help="Number of folds to use in cross validation.")

    parser.add_argument('-r', '--repeat',
                        required=False,
                        default=2,
                        type=int,
                        dest='n_repeats',
                        help="Number of cross validation iterations.")

    args_obj = parser.parse_args()
    main(args_obj)

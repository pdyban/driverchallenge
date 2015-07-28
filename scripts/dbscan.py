from sklearn.cluster import DBSCAN
import numpy as np

from utils import create_complete_submission
from utils import save_3d_plot_to_file, save_1d_plot_to_file

from features import AccelerationFeature
from features import AngleFeature


def clusterize(_features):
    def major_index(l):
        from collections import defaultdict

        d = defaultdict(int)
        for item in l:
            d[item] += 1

        return max(d.iteritems(), key=lambda x: x[1])[0]

    est = DBSCAN()

    Y = est.fit_predict(_features[:, 2:])

    y_pred = [(i == major_index(Y)) for i in Y]

    return np.c_[_features[:, 0], _features[:, 1], y_pred]


def clusterize_driver(X, _path):
    """
    Creates a single submission file.

    :param _driver: driver for which a csv submission file will be created
    :param _path: subdirectory where this submission part will be stored
    :return list of values: (driver, trip, prediction) for each trip for this driver
    """
    # X contains: (driver_id, trip_id, all features...)

    res = clusterize(X)
    try:
        x, y, z, c = [i[2] for i in X], [i[3] for i in X], [i[4] for i in X], [pred[2] for pred in res]
        save_3d_plot_to_file(x, y, z, c, _path.replace('.csv', '.png'))

    except IndexError:
        x, c = [i[2] for i in X], [pred[2] for pred in res]
        save_1d_plot_to_file(x, c, _path.replace('.csv', '.png'))

    return res  # TODO: are all trips included?


def compute_percentile_10(x):
    """
    Local function that wraps numpy's percentile is necessary, since joblib needs to pickle functions.
    """
    return np.percentile(x, 10)


def compute_percentile_20(x):
    """
    Local function that wraps numpy's percentile is necessary, since joblib needs to pickle functions.
    """
    return np.percentile(x, 20)


if __name__ == '__main__':
    # submission result: 0.52
    #features = [(10, 30, True, np.median), (31, 50, True, np.median), (51, 80, True, np.median), (10, 30, False, np.median)]
    #pdyban.create_complete_submission(features, parallel=True)

    # submission result: 0.51
    #features = [(10, 30, True, np.std), (31, 50, True, np.std), (51, 80, True, np.std), (10, 30, False, np.std)]
    #pdyban.create_complete_submission(features, parallel=True)

    # submission result: 0.47
    #features = [(10, 30, True, np.median), (31, 50, True, np.median), (51, 80, True, np.median), (10, 30, False, np.median)]
    #features += [(10, 30, True, np.std), (31, 50, True, np.std), (51, 80, True, np.std), (10, 30, False, np.std)]
    #pdyban.create_complete_submission(features, parallel=True)

    # features = [AccelerationFeature(10, 31, True, np.median),
    #             AccelerationFeature(31, 50, True, np.median),
    #             AccelerationFeature(51, 80, True, np.median),
    #             AccelerationFeature(10, 31, False, np.median), ]

    # submission result: 0.61
    #features = [AngleFeature(0, np.mean), ]

    # submission result: 0.52
    #features = [AngleFeature(0, compute_percentile_10), ]

    # submission result: 0.57
    features = [AngleFeature(0, compute_percentile_20), ]

    create_complete_submission(clusterize_driver, features, True)

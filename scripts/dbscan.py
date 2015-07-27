from sklearn.cluster import DBSCAN
import numpy as np

from utils import create_complete_submission
from utils import compute_all_features
from features import AccelerationFeature
from utils import save_3d_plot_to_file
import os.path


def clusterize(_features):
    def major_index(l):
        from collections import defaultdict
        d = defaultdict(int)
        for item in l:
            d[item]+=1

        return max(d.iteritems(), key=lambda x: x[1])[0]

    est = DBSCAN()

    Y = est.fit_predict(_features[:, 2:])

    y_pred = [(i==major_index(Y)) for i in Y]

    return np.c_[_features[:, 0], _features[:, 1], y_pred]


def clusterize_driver(_driver, _path, _features):
    """
    Creates a single submission file.

    :param _driver: driver for which a csv submission file will be created
    :param _path: subdirectory where this submission part will be stored
    :return list of values: (driver, trip, prediction) for each trip for this driver
    """
    print 'computing model for', _driver, _path, _features, '...'

    # X consists: (driver_id, trip_id, all features...)
    X = compute_all_features(int(_driver), _features)

    res = clusterize(X)

    x, y, z, c = [i[2] for i in X], [i[3] for i in X], [i[4] for i in X], [pred[2] for pred in res]

    submission_path = os.path.join(_path, str(_driver) + '.png')
    save_3d_plot_to_file(x, y, z, c, submission_path.replace('.csv', '.png'))

    return res  # TODO: are all trips included?


if __name__ == '__main__':
    # submission result: 0.47
    #features = [(10, 30, True, np.median), (31, 50, True, np.median), (51, 80, True, np.median), (10, 30, False, np.median)]
    #features += [(10, 30, True, np.std), (31, 50, True, np.std), (51, 80, True, np.std), (10, 30, False, np.std)]
    #create_complete_submission(features, True)

    features = [AccelerationFeature(10, 31, True, np.median),
                AccelerationFeature(31, 50, True, np.median),
                AccelerationFeature(51, 80, True, np.median),
                AccelerationFeature(10, 31, False, np.median), ]
    create_complete_submission(clusterize_driver, features, False)

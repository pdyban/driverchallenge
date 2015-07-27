
from utils import write_submission_to_file, create_complete_submission
from utils import compute_all_features

import sklearn
from sklearn.cluster import DBSCAN

from features import AccelerationFeature

import numpy as np


def clusterize(_features):
    def major_index(l):
        from collections import defaultdict
        d = defaultdict(int)
        for item in l:
            d[item]+=1

        return max(d.iteritems(), key=lambda x: x[1])[0]

    est = DBSCAN()

    Y = est.fit_predict(_features[:,2:])

    y_pred = [(i==major_index(Y)) for i in Y]

    return np.c_[_features[:, 0], _features[:, 1], y_pred]


def clusterize_driver(_driver, _path, _features):
    """
    Creates a single submission file.

    :param _driver: driver for which a csv submission file will be created
    :param _path: subdirectory where this submission part will be stored
    """
    print 'computing model for', _driver, _path, _features, '...'
    res = []
    features = compute_all_features(int(_driver), _features)

    D = np.ones(len(features)) * int(_driver)
    X = np.c_[D, features]

    res = clusterize(X)

    x, y, z, c = [i[2] for i in X], [i[3] for i in X], [i[4] for i in X], [pred[2] for pred in res]

    submission_path = os.path.join(_path, str(_driver) + '.csv')

    #fig, ax = plt.subplots()
    p = plt.subplot(1, 1, 1, projection='3d')
    p.scatter(x, y, z, c=c, marker='+')

    # save plot to file
    p.figure.savefig(submission_path.replace('.csv', '.png'))

    return res


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

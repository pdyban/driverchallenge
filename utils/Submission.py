
import math

import numpy as np

import DriverChallengeHelperFunctions as helpers
from joblib import Parallel, delayed  # for parallel for loop
from datetime import datetime  # for directory names with current time
import os
from collections import OrderedDict
import subprocess

# prepare plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

PARALLEL = False

#pathToDriverData = '../../driverchallenge_data/drivers'
pathToDriverData = '../drivers_npy'  # compute trip using ConvertDriverDataToNpy.py
pathToSpeedData = '../speed_npy'  # compute speed using session8_computeSpeedCurvatureAsNumPy.ipynb




from local_io import get_trip_npy


def compute_all_acc_features(driver, features):
    res = []  # container for all predictions

    for trip in range(1, 201):  # all trips
        path_points = get_trip_npy(driver, trip, pathToDriverData)

        pred = []
        for feature in features:
            pred.append( feature.compute(path_points) )

        if all(not math.isnan(val) for val in pred):
            res = [trip] + pred

    return np.array(res)



def fit_elliptic_envelope(_features, outliers_fraction=0.02, plot=False):
    from sklearn import covariance

    clf = covariance.EllipticEnvelope(contamination=.1)

    features_only = np.c_[_features[:, 2], _features[:, 3]]

    clf.fit(features_only)
    y_pred = clf.decision_function(features_only).ravel()

    # define a threshold for probabilities
    from scipy import stats
    threshold = stats.scoreatpercentile(y_pred, 100 * outliers_fraction)
    y_pred = y_pred > threshold

    # plot results
    if plot:
        plt.scatter([i[0] for i in features_only], [i[1] for i in features_only], c=y_pred, cmap='cool')

    return np.c_[_features[:,0], _features[:,1], y_pred]





def clusterize(_features):
    def major_index(l):
        from collections import defaultdict
        d = defaultdict(int)
        for item in l:
            d[item]+=1

        return max(d.iteritems(), key=lambda x: x[1])[0]

    import sklearn
    from sklearn.cluster import DBSCAN

    est = DBSCAN()

    Y = est.fit_predict(_features[:,2:])

    y_pred = [(i==major_index(Y)) for i in Y]

    return np.c_[_features[:, 0], _features[:, 1], y_pred]


def list_all_drivers():
    import os

    drivers = []

    for f in os.listdir(pathToDriverData):
        try:
            int(f)
            drivers.append(int(f))
        except ValueError:
            pass

    return sorted(drivers)


def list_all_drives():
    drives = OrderedDict()

    #print len(list_all_drivers())
    #print all(len(os.listdir(os.path.join(pathToDriverData, driver))) == 200 for driver in list_all_drivers())

    for driver in list_all_drivers():
        drives[driver] = 200
        #trips = len(os.listdir(os.path.join(pathToDriverData, driver)))
        #if trips != 200:
        #    print driver, trips, os.listdir(os.path.join(pathToDriverData, driver))

    return drives


def _create_submission(_driver, _path, _features):
    """
    Creates a single submission file.

    :param _driver: driver for which a csv submission file will be created
    :param _path: subdirectory where this submission part will be stored
    """
    print 'computing model for', _driver, _path, _features, '...'
    res = []
    features = compute_all_acc_features(int(_driver), _features)

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

    # save feature prediction to file
    create_submission(submission_path, res)


# @profile
def create_complete_submission(features, parallel=PARALLEL):
    subdir = '../submissions/' + datetime.now().strftime('%Y%m%d__%H%M%S')
    # create tmp directory for dumping all trips into
    os.makedirs(subdir)

    if parallel:
        drivers = list_all_drivers()
        Parallel(n_jobs=8)(delayed(_create_submission)(driver, subdir, features) for driver in drivers)

    else:
        #for driver in list_all_drivers():
        #    _create_submission(driver, subdir, features)
        #    raise Exception('Stop me')
        [_create_submission(driver, subdir, features) for driver in list_all_drivers()]

    subprocess.call("cat *.csv >submission.csv", cwd=subdir)

    # when this script is finished, call cat *.csv >submission.csv, to merge all files into 1
    return subdir


def test_submission(_subdir):
    import glob

    for f in glob.glob('../submissions/%s/*.csv' % _subdir):
        with open(f) as fr:
            if len(fr.readlines()) != 200:
                print f, len(fr.readlines())


if __name__ == '__main__':
    #subdir = create_complete_submission()

    subdir = '../submissions/091533'
    #_create_submission('1250', subdir)
    test_submission(subdir)

    with open(subdir + '/submission.csv') as f:
        print len(f.readlines())


if __name__ == '__main__':
    features = [
     [ 1, 1.,          True],
     [ 1, 2.,          False],
     [ 1, 3. ,         False],
     [ 1, 4. ,         1],
     [ 1, 5. ,         True],
     [ 2, 6. ,         True],
     [ 2, 7. ,         False],
     [ 2, 8. ,         0.0],
     [ 2, 9. ,         False]]
    create_submission('/tmp/submission_test.csv', features)


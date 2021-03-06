
import math

import numpy as np

from joblib import Parallel, delayed  # for parallel for loop
from datetime import datetime  # for directory names with current time
import os
from collections import OrderedDict
import subprocess
from local_io import write_submission_to_file

#PARALLEL = False

#pathToDriverData = '../../driverchallenge_data/drivers'
pathToDriverData = 'drivers_npy'  # compute trip using ConvertDriverDataToNpy.py
#pathToSpeedData = '../speed_npy'  # compute speed using session8_computeSpeedCurvatureAsNumPy.ipynb
pathToSubmissions = 'submissions'


from local_io import get_trip_npy


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


def compute_all_features(driver, features):
    res = []  # container for all predictions

    for trip in range(1, 201):  # all trips
        path_points = get_trip_npy(driver, trip, pathToDriverData)

        feature_array = []
        for feature in features:
            feature_array.append( feature.compute(path_points) )

        if all(not math.isnan(val) for val in feature_array):
            res += [[driver, trip] + feature_array]

        else:
            res += [[driver, trip] + [None]*len(feature_array)]
            raise ValueError('Data not available for at least one of the features')
            # if data is not valid, then return None, handle this case separately during estimation

    return np.array(res)


def write_to_file(_estimator, _driver, _subdir, _features):
    """
    Wraps feature generation and writes results to file.

    This function cannot be local, it has to be pickled by joblib.
    """
    print 'computing model for driver', _driver, 'using', _features

    X = compute_all_features(int(_driver), _features)

    submission_path = os.path.join(_subdir, str(_driver) + '.csv')
    submission = _estimator(X, submission_path)

    #print 'storing prediction file to', _subdir

    # save feature prediction to file
    write_submission_to_file(submission_path, submission)


def create_complete_submission(estimator, features, parallel):
    """
    Creates a submission file for all drivers and their trips.

    :param estimator: function that computes the features for each driver
    :param parallel: Should generation script run parallel.
    :return:
    """
    subdir = os.path.join(pathToSubmissions, datetime.now().strftime('%Y%m%d__%H%M%S'))
    # create tmp directory for dumping all trips into
    os.makedirs(subdir)

    if parallel:
        drivers = list_all_drivers()
        Parallel(n_jobs=8)(delayed(write_to_file)(estimator, driver, subdir, features) for driver in drivers)

    else:
        #for driver in list_all_drivers():
        #for driver in [12]:
        #    write_to_file(estimator, driver, subdir, features)
        #    raise Exception('Stop me')
        [write_to_file(estimator, driver, subdir, features) for driver in list_all_drivers()]

    test_submission(subdir)
    subprocess.call("cat *.csv >submission.csv", cwd=subdir)

    # when this script is finished, call cat *.csv >submission.csv, to merge all files into 1
    return subdir


def test_submission(_subdir):
    import glob

    for f in glob.glob('%s/*.csv' % _subdir):
        with open(f) as fr:
            if len(fr.readlines()) != 200:
                print f, len(fr.readlines())


# if __name__ == '__main__':
#     #subdir = create_complete_submission()
#
#     subdir = '../submissions/091533'
#     #_create_submission('1250', subdir)
#     test_submission(subdir)
#
#     with open(subdir + '/submission.csv') as f:
#         print len(f.readlines())

#
# if __name__ == '__main__':
#     features = [
#      [ 1, 1.,          True],
#      [ 1, 2.,          False],
#      [ 1, 3. ,         False],
#      [ 1, 4. ,         1],
#      [ 1, 5. ,         True],
#      [ 2, 6. ,         True],
#      [ 2, 7. ,         False],
#      [ 2, 8. ,         0.0],
#      [ 2, 9. ,         False]]
#     create_submission('/tmp/submission_test.csv', features)
#

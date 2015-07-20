import math

import numpy as np

import DriverDataIO as io
import DriverChallengeHelperFunctions as helpers
from Submission import create_submission
from joblib import Parallel, delayed  # for parallel for loop
from datetime import datetime  # for directory names with current time
import os
from collections import OrderedDict
import subprocess


PARALLEL = True


#pathToDriverData = '../../driverchallenge_data/drivers'
pathToDriverData = '../drivers'


def interpolate_speed(_speed):
    """
    This function interpolates between speed points and adds a time point for every integer speed value.
    """
    import math
    interpolatedSpeed = []
    
    for t0, s0 in enumerate(_speed[:-1]):
        s1 = _speed[t0+1]
        
        
        interpolatedSpeed.append((s0,t0))
        
        left = min(s0, s1)
        right = max(s0,s1)
        #print "left {0} right {1}".format(left, right)
        for point in range(int(math.ceil(left)), int(math.floor(right))):
            t = (point - s0)/(s1-s0)+t0
            #print point, t
            interpolatedSpeed.append((point,t))
            
    return interpolatedSpeed


def get_acceleration(speed):
    """
    Computes acceleration for list of speed measurements.
    Acceleration is negative for breaking.
    @rtype list
    """
    acc = []
    for i in range(len(speed)-1):
        if speed[i+1][1] - speed[i][1] == 0.0:
            acc.append(0.0)
        else:
            acc.append((speed[i+1][0] - speed[i][0])/(speed[i+1][1] - speed[i][1]))
    return acc


def bin_speed_interval(_speed, _from, _to, _acceleration=True):
    """
    Returns time values where the speed falls between the given _from and _to margins. 
    @param _acceleration: if true, then acceleration feature is computed, otherwise deceleration.
    @return time points and the speed value for points where speed lies in the requested interval.
    @rtype: list(tuple)
    """
    acc = get_acceleration(_speed)
    
    _speed = _speed[1:]
    
    intervals = []
    # start at index 2 due to shift of indices after deriving distance to acceleration
    for (sp, ac) in zip(_speed, acc):
        if (sp[0] >= _from and sp[0] <= _to):
            if (_acceleration and ac > 0) or (not _acceleration and ac < 0):
                intervals.append((sp[0],sp[1]))
        
    return intervals


def find_intervals(_interval, _acceleration=True):
    """
    Splits a list of points into connected intervals. 
    Points inside one interval are adjacent to each other. 
    Speed increases/decreases along the interval.
    Intervals are separated by at least one point where speed decreases/increases.
    """
    # allow only intervals with this minimum length to be used for feature extraction. 
    # Smaller intervals introduce high uncertainty nd numerical instability
    MIN_ALLOWED_INTERVAL_LENGTH = 2
    
    intervals = []
    pop = []
    for point in _interval:
        if len(pop) > 0 and ((_acceleration and point < pop[-1]) or (not _acceleration and point > pop[-1])):
            if len(pop) > MIN_ALLOWED_INTERVAL_LENGTH:  
                intervals.append(pop)
                
            pop = []
            
        pop.append(point)
        
    if len(pop) > MIN_ALLOWED_INTERVAL_LENGTH:
        intervals.append(pop)
    
    return intervals


def compute_acceleration_feature(_intervals, feat=np.mean):
    """
    This function computes acceleration feature for a list of connected, 
    disjoint intervals with increasing speed.
    
    Mean acceleration across all intervals is chosen as the acceleration feature.
    
    @param feat: specify which type of statistic to use as the feature.
    @type feat: numpy.mean or numpy.std or numpy.median
    """
    prefeature = []
    for interval in _intervals:
        if abs(interval[-1][0] - interval[0][0]) < 0.1:
            continue
            
        value = (interval[-1][1] - interval[0][1])/(interval[-1][0] - interval[0][0])
        prefeature.append(abs(value))
        #print value

    if len(prefeature):
        return feat(prefeature)

    else:
        return 0.0  # if there isn't a single valid interval

    
def compute_all_acc_features(driver):
    feature_array = []  # container for all acceleration features
    
    for i in range(1, 201):  # all trips
        _trip = io.get_trip(driver, i, pathToDriverData)
        speed = helpers.get_speed(_trip)
        interpolated_speed = interpolate_speed(speed)
        feature_val = []
        #for (interval_begin, interval_end, acceleration) in [(10, 30, True), (10, 30, False)]:
        for (interval_begin, interval_end, acceleration, feat) in [(10, 30, True, np.median), (10, 30, True, np.std)]:
            interval = bin_speed_interval(interpolated_speed, interval_begin, interval_end, acceleration)
            contiguous_intervals = find_intervals(interval, acceleration)
            if len(contiguous_intervals) == 0:
                if len(feature_val) > 0:
                    feature_val.append(np.mean(feature_val))

                else:
                    feature_val.append(0.0)

            else:
                feature_val.append( compute_acceleration_feature(contiguous_intervals, feat=feat) )

        if all(not math.isnan(val) for val in feature_val):
            feature_array.append([i] + feature_val)

    features = np.array(feature_array)
    
    return features
    

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


def _create_submission(_driver, _path):
    """
    Creates a single submission file.

    :param _driver: driver for which a csv submission file will be created
    :param _path: subdirectory where this submission part will be stored
    """
    res = []
    features = compute_all_acc_features(int(_driver))
    D = np.ones(len(features))*int(_driver)
    features = np.c_[ D, features]

    res.extend(fit_elliptic_envelope(features))
    submission_path = os.path.join(_path, str(_driver) + '.csv')
    create_submission(submission_path, res)


# @profile
def create_complete_submission():
    subdir = '../submissions/' + datetime.now().strftime('%I%M%S')
    # create tmp directory for dumping all trips into
    os.makedirs(subdir)

    if PARALLEL:
        Parallel(n_jobs=8)(delayed(_create_submission)(driver, subdir) for driver in list_all_drives())

    else:
        #for driver in list_all_drivers():
        #    _create_submission(driver, subdir)
        #    raise Exception('Stop me')
        [_create_submission(driver, subdir) for driver in list_all_drives()]

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

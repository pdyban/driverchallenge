import math

import numpy as np

import DriverDataIO as io
import DriverChallengeHelperFunctions as helpers
from Submission import create_submission
import gc

pathToDriverData = '../../driverchallenge_data/drivers'

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
        
    return feat(prefeature)

    
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
            feature_val.append( compute_acceleration_feature(contiguous_intervals, feat=feat) )

        if all(not math.isnan(val) for val in feature_val):
            feature_array.append([i] + feature_val)

    features = np.array(feature_array)
    
    return features
    

def fit_elliptic_envelope(_features, outliers_fraction = 0.02, plot=False):
    from sklearn import covariance

    clf = covariance.EllipticEnvelope(contamination=.1)

    #print features[:,2], features[:,3]
    features_only = np.c_[_features[:,2], _features[:,3]]
    #print features_only

    clf.fit(features_only)  
    y_pred = clf.decision_function(features_only).ravel()

    # define a threshold for probabilities
    from scipy import stats
    threshold = stats.scoreatpercentile(y_pred, 100 * outliers_fraction)
    y_pred = y_pred > threshold
    #print y_pred

    # plot results
    if plot:
        plt.scatter([i[0] for i in features_only], [i[1] for i in features_only], c=y_pred, cmap='cool')
    
    return np.c_[_features[:,0], _features[:,1], y_pred]



def list_all_drivers():
    import os

    drivers = []
    # max_num_drivers = 10
    cdrivernum = 1;
    for f in os.listdir(pathToDriverData):
        try:
            int(f)
            drivers.append(f)
        except ValueError:
            pass

        cdrivernum += 1
        # if cdrivernum > max_num_drivers:
        #    break
    return drivers

def list_all_drives():
    drives = {}
    for driver in list_all_drivers():
        drives[driver] = 200
        
    return drives

# @profile
def create_complete_submission():
    all_drives = list_all_drives()
    drive_index = 0.0
    for driver, num_drives in all_drives.iteritems():
        if drive_index % 100 == 0:
            print drive_index / len(all_drives)
        res = []
        features = compute_all_acc_features(int(driver))
        D = np.ones(len(features))*int(driver)
        features = np.c_[ D, features]

        res.extend( fit_elliptic_envelope(features) )
        submission_path = '../submissions/' + driver + '.csv'
        create_submission(submission_path, res)
        drive_index += 1

if __name__ == '__main__':
    create_complete_submission()

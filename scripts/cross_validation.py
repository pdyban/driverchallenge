# coding=utf-8
from numpy.core.multiarray import concatenate

__author__ = 'missoni'

"""
This script generates a submission based on a cross-validaiton approach.

Inspiration from here (http://webmining.olariu.org/kaggle-driver-telematics/):

    For each driver, take 180 trips and label them as 1s. Take 180 trips from other drivers and label them as 0s. Train and test on the remaining 20 trips from this driver and on another 20 trips from other drivers. That’s it.

    Can we improve on this? Yes. Take more than just 180 trips from other drivers. Best results I’ve got were with values between 4×180 and 10×180. In order to avoid an unbalanced training set, I also duplicated the data from the current driver.

"""

from features import AccelerationFeature
import numpy as np
from utils import create_complete_submission
from sklearn.ensemble import RandomForestClassifier
from utils import list_all_drivers, PATHTODRIVERDATA


def cross_validate(_features_true, _features_false, percentage):
    """
    1) split data by cross validation into train & test
    2) validate using random forest
    3) summarize results
    :param _features: input (driver_id, trip_id, all features)
    :return: (driver_id, trip_id,prob)
    """

    res0 = []
    res1 = []

    for p in range((int)(1.0/percentage)):
        _from = _features_true.shape[0] * p*percentage
        _to = _from + ( percentage * _features_true.shape[0] )

        trainFeature0 = np.vstack((_features_true[0:_from], _features_true[_to:]))
        testFeature0 = _features_true[_from:_to]

        _from = _features_false.shape[0] * p*percentage
        _to = _from + percentage * _features_false.shape[0]

        trainFeature1 = np.vstack((_features_false[0:_from], _features_false[_to:]))
        #testFeature1 = _features_false[_from:_to]

        # extend train feature true in order to balance against a bigger number of false features
        trainFeature0 = np.repeat(trainFeature0, (trainFeature1.shape[0]/trainFeature0.shape[0]), axis=0)

        X_train = np.vstack((trainFeature0, trainFeature1))
        X_test = testFeature0
        Y_train = np.append(np.ones(trainFeature0.shape[0]), np.zeros(trainFeature1.shape[0]))
        #Y_test = np.append(np.ones(testFeature0.size), np.zeros(testFeature1.size))

        clf = RandomForestClassifier(n_estimators=25)
        clf.fit(X_train, Y_train)
        clf_probs = clf.predict_proba(X_test)

        res0 = np.append(res0, [elem[0] for elem in clf_probs])
        #res1 = np.append(res1, [elem[0] for elem in clf_probs[clf_probs.shape[0]/2:]])


    return res0, res1


    #return np.c_[_features[:, 0], _features[:, 1], y_pred]


def cvalidate_driver(X, _path):
    """
    Creates a single submission file.

    :param X: contains (driver_id, trip_id, all features...)
            _driver: driver for which a csv submission file will be created
            _path: subdirectory where this submission part will be stored
    :return list of values: (driver, trip, prediction) for each trip for this driver
    """
    res = cross_validate(X)

    return res

if __name__ == '__main__':
    feat1 = np.load('../features_npy/feat1.npy')
    feat2 = np.load('../features_npy/feat2.npy')

    # for cross-validation, extend feature arrays by the width of the sliding window
    feat1_ext = np.vstack((feat1, feat1[:1000, :]))
    feat2_ext = np.vstack((feat2, feat2[:1000, :]))

    features = np.hstack((feat1_ext[:, 2, np.newaxis], feat2_ext[:, 2, np.newaxis]))

    # naive test: driver 1 vs driver 2
    #features_true = feat1[:200, 2, np.newaxis]
    #features_false = feat2[200:400, 2, np.newaxis]

    pred = np.empty(0)[:, np.newaxis]

    drivers = list_all_drivers(PATHTODRIVERDATA)[:10]
    for driver_index, driver in enumerate(drivers):
        # test 4 times driver 1 vs driver 2, 3, 4
        begin = driver_index*200
        end = (driver_index+1)*200
        end_false = (driver_index+4+1)*200

        features_true = features[begin:end, :]
        features_false = features[end:end_false, :]

        res0, res1 = cross_validate(features_true, features_false, 0.2)

        pred = np.vstack((pred, res0[:, np.newaxis]))

    res = np.c_[(feat1[:, 0], feat1[:, 1], pred)]

    import matplotlib.pyplot as plt

    #plt.scatter(range(res0.shape[0]), res0)
    #plt.show()


    hist = np.histogram(res0)

    # the histogram of the data
    n, bins, patches = plt.hist(res[:200, 2], normed=0, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(res[200:400, 2], normed=0, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(res[400:600, 2], normed=0, facecolor='yellow', alpha=0.75)
    #plt.plot(hist[1][:-1], hist[0], 'r--', linewidth=1)
    plt.show()



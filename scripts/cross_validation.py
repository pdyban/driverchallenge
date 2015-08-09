# coding=utf-8
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


def cross_validate(_features0, _features1, percentage):
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
        _from = _features0.shape[0] * p*percentage
        _to = _from + ( percentage * _features0.shape[0] )

        trainFeature0 = np.vstack((_features0[0:_from], _features0[_to:]))
        testFeature0 = _features0[_from:_to]

        trainFeature1 = np.vstack((_features1[0:_from], _features1[_to:]))
        testFeature1 = _features1[_from:_to]


        X_train = np.vstack((trainFeature0, trainFeature1))
        X_test = np.vstack((testFeature0, testFeature1))
        Y_train = np.append(np.ones(trainFeature0.size), np.zeros(trainFeature1.size))
        Y_test = np.append(np.ones(testFeature0.size), np.zeros(testFeature1.size))

        clf = RandomForestClassifier(n_estimators=25)
        clf.fit(X_train, Y_train)
        clf_probs = clf.predict_proba(X_test)

        res0 = np.append(res0, [elem[0] for elem in clf_probs[:clf_probs.shape[0]/2]])
        res1 = np.append(res1, [elem[0] for elem in clf_probs[clf_probs.shape[0]/2:]])


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
    # submission result:
    features = [AccelerationFeature(10, 30, True, np.mean), ]  # using RDP

    #create_complete_submission(cvalidate_driver, features, False)

    from utils import local_io

    pathToDriverData = '/home/qwerty/kaggle/drivers_npy'

    features_driver1 = []
    features_driver2 = []

    for i in range(1,201):

        trip1 = local_io.get_trip_npy(1, i, pathToDriverData)
        trip2 = local_io.get_trip_npy(2, i, pathToDriverData)

        trip1_feature = features[0].compute(trip1)
        trip2_feature = features[0].compute(trip2)
        features_driver1 = np.append(features_driver1, trip1_feature)
        features_driver2 = np.append(features_driver2, trip2_feature)


    res0, res1 = cross_validate(np.vstack(features_driver1), np.vstack(features_driver2), 0.2)

    import matplotlib.pyplot as plt

    plt.scatter(range(200), res0)
    plt.show()



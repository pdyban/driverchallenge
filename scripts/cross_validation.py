# coding=utf-8

__author__ = 'missoni'

"""
This script generates a submission based on a cross-validaiton approach.

Inspiration from here (http://webmining.olariu.org/kaggle-driver-telematics/):

    For each driver, take 180 trips and label them as 1s. Take 180 trips from other drivers and label them as 0s. Train and test on the remaining 20 trips from this driver and on another 20 trips from other drivers. That’s it.

    Can we improve on this? Yes. Take more than just 180 trips from other drivers. Best results I’ve got were with values between 4×180 and 10×180. In order to avoid an unbalanced training set, I also duplicated the data from the current driver.

"""

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier


PARALLEL = True


def cross_validate(_features_true, _features_false, percentage):
    """
    1) split data by cross validation into train & test
    2) validate using random forest
    3) summarize results
    :param _features: input (driver_id, trip_id, all features)
    :return: (driver_id, trip_id,prob)
    """

    res0 = []
    score = 0.0

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
        Y_test = np.ones(testFeature0.shape[0])

        clf = RandomForestClassifier(n_estimators=25)
        clf.fit(X_train, Y_train)

        score = score + clf.score(X_test, Y_test)

        clf_probs = clf.predict_proba(X_test)

        res0 = np.append(res0, [elem[0] for elem in clf_probs])

    score = score * percentage

    return res0, score


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
    features = []
    for feat_files in os.listdir('../features_npy/'):

        feat1 = np.load(os.path.join('../features_npy/', feat_files))
        drivers_list = feat1[:, 0]
        trips_list = feat1[:, 1]

        # for cross-validation, extend feature arrays by the width of the sliding window
        feat1_ext = np.vstack((feat1, feat1[:1000, :]))

        features.append(feat1_ext[:, 2, np.newaxis])

    features = np.hstack([f for f in features])

    #features = np.hstack((feat1_ext[:, 2, np.newaxis], feat2_ext[:, 2, np.newaxis]))

    # naive test: driver 1 vs driver 2
    #features_true = feat1[:200, 2, np.newaxis]
    #features_false = feat2[200:400, 2, np.newaxis]

    #pred = np.empty(0)[:, np.newaxis]

    #pred_mthreaded = []

    def compute_iteration(driver_index):
        print 'evaluating driver index', driver_index#, '(%d%%)' % (driver_index*100.0/num_drivers)

        # test 4 times driver 1 vs driver 2, 3, 4
        begin = driver_index*200
        end = (driver_index+1)*200
        end_false = (driver_index+4+1)*200

        features_true = features[begin:end, :]
        features_false = features[end:end_false, :]

        res0, score = cross_validate(features_true, features_false, 0.2)

        return res0[:, np.newaxis], score
        #pred = np.vstack((pred, res0[:, np.newaxis]))

        #mean_score += score

    #mean_score = 0.0

    num_drivers = 2738
    if PARALLEL:
        from multiprocessing import Pool
        p = Pool()
        pred_mthreaded = p.map(compute_iteration, range(num_drivers))

    else:
        pred_mthreaded = []
        for driver_index in range(num_drivers):
            pred_mthreaded.append(compute_iteration(driver_index))

    print 'mean score', np.mean([score[1] for score in pred_mthreaded])

    # replace with drivers_list and trips_list
    res = np.c_[(drivers_list, trips_list, np.vstack([pred[0] for pred in pred_mthreaded]))]

    np.save(open('../tmp/res.npy', 'w'), res)

    import matplotlib.pyplot as plt

    # the histogram of the data
    n, bins, patches = plt.hist(res[:200, 2], normed=0, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(res[200:400, 2], normed=0, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(res[400:600, 2], normed=0, facecolor='yellow', alpha=0.75)
    #plt.plot(hist[1][:-1], hist[0], 'r--', linewidth=1)
    plt.show()



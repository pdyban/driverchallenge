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
from sklearn.linear_model import LogisticRegression
from time import time
from datetime import datetime
from utils import write_submission_to_file
from itertools import izip
from csv import writer


PARALLEL = True
NUMDRIVERS = 2736
SPREAD = 4
THRESHOLD = 0.5
SAVE_NPY = False  # should save prediction probabilities as numpy arrays
DISPLAY_PLOT = False  # should pop up plot after each feature is completed
SAVE_PLOT = True  # should save plot to png file


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

        #clf = RandomForestClassifier(n_estimators=20)
        clf = LogisticRegression(C=0.001)
        #from sklearn.ensemble import GradientBoostingClassifier
        #clf = GradientBoostingClassifier()
        clf.fit(X_train, Y_train)

        #score = score + clf.score(X_test, Y_test)

        clf_probs = clf.predict_proba(X_test)

        score += sum(1 for i in clf_probs if i[1] > THRESHOLD) * 1.0 / len(clf_probs)

        res0 = np.append(res0, [elem[1] for elem in clf_probs])

    score = score * percentage

    return res0, score


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


def compute_iteration(args):
    features, driver_index = args

    if driver_index % 100 == 0:
        print 'evaluating driver index', driver_index

    # test 4 times driver 1 vs driver 2, 3, 4
    begin = driver_index*200
    end = (driver_index+1)*200
    end_false = (driver_index+SPREAD+1)*200

    features_true = features[begin:end, :]
    features_false = features[end:end_false, :]

    res0, score = cross_validate(features_true, features_false, 0.2)

    return res0, score


def run_crossvalidation_create_submission(features, subdir):
    features = np.hstack([f for f in features])

    start = time()
    if PARALLEL:
        from multiprocessing import Pool
        p = Pool()
        pred_mthreaded = p.map(compute_iteration, izip([features]*NUMDRIVERS, range(NUMDRIVERS)))

    else:
        pred_mthreaded = []
        for driver_index in range(NUMDRIVERS):
            pred_mthreaded.append(compute_iteration(features, driver_index))

    end = time()

    mean_score = np.mean([score[1] for score in pred_mthreaded])
    print 'model evaluation finished in', end-start, 'seconds with mean score =', mean_score

    pred = np.vstack([pred[0][:, np.newaxis] for pred in pred_mthreaded])

    filename = os.path.join('../submissions/',
                          subdir,
                          datetime.now().strftime('%Y%m%d__%H%M%S'))

    res = np.c_[(drivers_list, trips_list, pred)]

    if SAVE_NPY:
        np.save(open('../tmp/res.npy', 'w'), res)

    # apply threshold for submission
    res_filtered = np.c_[(drivers_list, trips_list, np.array(np.where(pred > THRESHOLD, 1, 0)))]
    write_submission_to_file('%s.csv' % filename, res_filtered, True)

    if DISPLAY_PLOT or SAVE_PLOT:
        import matplotlib.pyplot as plt

        # the histogram of the data
        n, bins, patches = plt.hist(res[:200, 2], normed=0, facecolor='green', alpha=0.75)
        n, bins, patches = plt.hist(res[200:400, 2], normed=0, facecolor='red', alpha=0.75)
        n, bins, patches = plt.hist(res[400:600, 2], normed=0, facecolor='yellow', alpha=0.75)
        n, bins, patches = plt.hist(res[600:800, 2], normed=0, facecolor='blue', alpha=0.75)
        if DISPLAY_PLOT:
            plt.show()

        if SAVE_PLOT:
            plt.savefig(os.path.join('%s.png' % filename))

    # store estimated scores in a csv
    with open(os.path.join(os.path.dirname(filename), 'scores.csv'), 'a') as scores_files:
        r = writer(scores_files, delimiter=';')
        r.writerow([filename, mean_score])

    return mean_score


def list_feature_files():
    #return [os.path.join('../features_npy/', f) for f in ['feat%d.npy' % i for i in [0,1,2,16]]]
    #return [os.path.join('../features_npy_old/', f) for f in ['feat%d.npy' % i for i in [0, 1, 2, 16]]]
    return (os.path.join('../features_npy_old/', f) for f in os.listdir('../features_npy_old/')
           if not f.startswith('.'))


if __name__ == '__main__':
    features = []
    feature_files = list_feature_files()

    subdir = datetime.now().strftime('%Y%m%d__%H%M%S')

    for feat_files in feature_files:

        feat1 = np.load(feat_files)
        drivers_list = feat1[:, 0]
        trips_list = feat1[:, 1]

        # for cross-validation, extend feature arrays by the width of the sliding window
        feat1_ext = np.vstack((feat1, feat1[:200*SPREAD, :]))

        features.append(feat1_ext[:, 2, np.newaxis])

    score = run_crossvalidation_create_submission(features, subdir)

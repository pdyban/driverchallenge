__author__ = 'missoni'

from feature import Feature
from speed import Speed
import numpy as np

import math


class AccelerationFeature(Speed, Feature):
    """
    Computes acceleration feature.
    """
    def __init__(self, _from, _to, _acceleration, _feature, _interpolate=True):
        super(AccelerationFeature, self).__init__()
        self.to_ = _to
        self.from_ = _from
        self.acc = _acceleration
        self.feat = _feature  # e.g. np.median, np.std
        self.interpolate = _interpolate

    def __repr__(self):
        return "%s %s feature for speed interval (%d, %d)" % (self.feat.__name__, ['decceleration', 'acceleration'][self.acc], self.from_, self.to_)

    def compute(self, trip):

        speed = self.get_speed(trip)
        speed = self.smooth_speed(speed)
        interpolated_speed = self.interpolate_speed(speed)

        interval = self.bin_speed_interval(interpolated_speed, self.from_, self.to_, self.acc)  # last parameter is acceleration indicator, True for acceleration
        contiguous_intervals = self.find_intervals(interval, self.acc)  # last param is acceleration parameter

        if len(contiguous_intervals) == 0:
            feature_val = Feature.INVALIDDATAREPLACEMENT

        else:
            feature_val = self.compute_acceleration_feature(contiguous_intervals, feat=self.feat)

        return feature_val


    def compute_acceleration_feature(self, _intervals, feat):
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

        if len(prefeature):
            return feat(prefeature)

        else:
            return Feature.INVALIDDATAREPLACEMENT  # if there isn't a single valid interval
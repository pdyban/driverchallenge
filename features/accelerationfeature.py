__author__ = 'missoni'

from feature import Feature
from speed import Speed
import numpy as np

import math


class AccelerationFeature(Speed, Feature):
    """
    Computes acceleration feature.
    """
    def __init__(self, _from, _to, _acceleration, _feature):
        super(AccelerationFeature, self).__init__()
        self.to_ = _to
        self.from_ = _from
        self.acc = _acceleration
        self.feat = _feature  # np.median, np.std

    def compute(self, trip):

        #feature_array = []  # container for all acceleration features

        speed = self.get_speed(trip)
        speed = self.smooth_speed(speed)
        interpolated_speed = self.interpolate_speed(speed)

        interval = self.bin_speed_interval(interpolated_speed, self.from_, self.to_, self.acc)  # last parameter is acceleration indicator, True for acceleration
        contiguous_intervals = self.find_intervals(interval, self.acc)  # last param is acceleration parameter

        feature_val = 0.0

        if len(contiguous_intervals) == 0:
            if len(feature_val) > 0:
                feature_val = np.mean(feature_val)

            else:
                feature_val = 0.0

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
            return 0.0  # if there isn't a single valid interval
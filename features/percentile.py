__author__ = 'missoni'

from feature import Feature
from speed import Speed
import numpy as np


class SpeedPercentileFeature(Speed, Feature):
    """
    Computes percentile feature over speed.
    """
    def __init__(self, value):
        super(SpeedPercentileFeature, self).__init__()
        self.value = value

    def __repr__(self):
        return "Speed percentile feature %d" % self.value

    def compute(self, trip):
        speed = self.get_speed(trip)
        return np.percentile(speed, self.value)


class AccelerationPercentileFeature(Speed, Feature):
    """
    Computes percentile feature over acceleration.
    """
    def __init__(self, value):
        super(AccelerationPercentileFeature, self).__init__()
        self.value = value

    def __repr__(self):
        return "Acceleration percentile feature %d" % self.value

    def compute(self, trip):
        speed = self.get_speed(trip)

        acc = []
        for item in range(len(speed)-1):
            acc.append(speed[item] - speed[item+1])

        return np.percentile(acc, self.value)

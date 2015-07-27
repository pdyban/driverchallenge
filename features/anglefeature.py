__author__ = 'scigor'

from feature import Feature
import numpy as np
import math
try:
    from rdp import rdp
except ImportError:
    from warnings import warn
    warn("RDP not found, Angle Feature will not run")
    rdp = lambda x, y: x


class AngleFeature(Feature):
    """
    Computes angle feature.
    """
    #INVALIDDATAREPLACEMENT = float('Nan')

    def __init__(self, _rdpFactor, _feature=np.mean):
        super(Feature, self).__init__()

        self.rdpFactor_ = _rdpFactor
        self.feat = _feature  # e.g. np.mean

    def __repr__(self):
        return 'angle feature %s RDP' % ['without', 'with'][self.rdpFactor_ > 0]

    def compute(self, trip):

        if self.rdpFactor_ > 0:
            trip = rdp(trip, self.rdpFactor_)

        angle_feature_elements = []
        for index, p in list(enumerate(trip)):
            current_angle = self.get_angle(index, trip)

            if current_angle == -1:
                continue

            length0 = self.distance(trip[index-1], p)
            length1 = self.distance(p, trip[index+1])
            driven_distance = length0 + length1
            feature_value = driven_distance * current_angle
            angle_feature_elements.append(feature_value)

        if len(angle_feature_elements) > 0:
            return self.feat(angle_feature_elements)

        else:
            return AngleFeature.INVALIDDATAREPLACEMENT

    def distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def get_angle(self, p, trip):
        if 0 < p < len(trip)-1:
            p1 = trip[p-1]
            p2 = trip[p]
            p3 = trip[p+1]
            p12 = self.distance(p1, p2)
            p13 = self.distance(p1, p3)
            p23 = self.distance(p2, p3)
            numerator = p12**2 + p23**2 - p13**2
            denominator = 2 * p12 * p23
            if denominator == 0:
                return -1

            return math.acos(round(numerator/denominator, 5))/math.pi

        return -1

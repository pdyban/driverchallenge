__author__ = 'scigor'

from feature import Feature
import numpy as np
import math


class AngleFeature(Feature):
    """
    Computes angle feature.
    """
    def __init__(self):
        super(Feature, self).__init__()

    def compute(self, trip):
        angle_feature_elements = []
        for index,p in list(enumerate(trip)):
            current_angle = self.get_angle(self, index, trip)

            if current_angle == -1:
                continue

            length0 = self.distance(trip[index-1], p)
            length1 = self.distance(p, trip[index+1])
            driven_distance = length0 + length1
            feature_value = driven_distance * current_angle;
            angle_feature_elements.append(feature_value)

        if(len(angle_feature_elements) > 0 ):
            return np.mean(angle_feature_elements)


    def distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


    def get_angle(self, p, trip):
        if(p > 0 and p < len(trip)-1):
            p1 = trip[p-1]
            p2 = trip[p]
            p3 = trip[p+1]
            p12 = self.distance(p1, p2)
            p13 = self.distance(p1, p3)
            p23 = self.distance(p2, p3)
            numerator = p12**2 + p23**2 - p13**2
            denomenator = 2 * p12 * p23
            if denomenator == 0:
                return -1
            return math.acos(round(numerator/denomenator, 5))/pi
        return -1

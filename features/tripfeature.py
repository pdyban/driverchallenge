__author__ = 'missoni'

from feature import Feature

class TripLengthFeature(Feature):
    """
    Measures trip length.
    """
    def __init__(self):
        super(Feature, self).__init__()

    def compute(self, trip):
        return len(trip)
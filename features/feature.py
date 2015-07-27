__author__ = 'missoni'


class Feature(object):
    """
    Interface for different features.
    """
    INVALIDDATAREPLACEMENT = 0.0  # todo: think about a replacement value for non-available data

    def __init__(self):
        super(Feature, self).__init__()

    def compute(self, trip):
        """
        Returns value of the feature for the trip.

        :param trip:
        :rtype: float
        """
        raise NotImplementedError('Implement this!')

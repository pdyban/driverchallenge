__author__ = 'missoni'


class Feature(object):
    """
    Interface for different features.
    """
    def __init__(self):
        super(Feature, self).__init__()

    def compute(self, trip):
        """
        Returns value of the feature for the trip.

        :param trip:
        :return:
        """
        raise NotImplementedError('Implement this!')
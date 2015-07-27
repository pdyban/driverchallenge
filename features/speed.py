__author__ = 'missoni'

import math

class Speed(object):
    """
    Contains convenience functions for speed features.
    """
    def __init__(self):
        super(Speed, self).__init__()

    def interpolate_speed(self, _speed):
        """
        This function interpolates between speed points and adds a time point for every integer speed value.
        """
        interpolated_speed = []

        for t0, s0 in enumerate(_speed[:-1]):
            s1 = _speed[t0+1]

            interpolated_speed.append((s0,t0))

            left = min(s0, s1)
            right = max(s0, s1)

            for point in range(int(math.ceil(left)), int(math.floor(right))):
                t = (point - s0)/(s1-s0)+t0
                interpolated_speed.append((point,t))

        return interpolated_speed

    def get_speed(self, _trip):
        speed = []

        def distance(p1, p2):
            return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

        for i in range(len(_trip)-1):
            speed.append(distance(_trip[i], _trip[i+1])*3.6)

        return speed

    def smooth_speed(self, _speed, filter=9):
        """
        Smoothes signal using median filter and the given filter size.
        """
        import scipy
        from scipy.signal import medfilt
        return medfilt(_speed, filter)

    def compute_acceleration(self, speed):
        """
        Computes acceleration for list of speed measurements.
        Acceleration is negative for breaking.
        @rtype list
        """
        acc = []

        for i in range(len(speed)-1):
            if speed[i+1][1] - speed[i][1] == 0.0:
                acc.append(0.0)
            else:
                acc.append((speed[i+1][0] - speed[i][0])/(speed[i+1][1] - speed[i][1]))

        return acc

    def bin_speed_interval(self, _speed, _from, _to, _acceleration=True):
        """
        Returns time values where the speed falls between the given _from and _to margins.
        @param _acceleration: if true, then acceleration feature is computed, otherwise deceleration.
        @return time points and the speed value for points where speed lies in the requested interval.
        @rtype: list(tuple)
        """
        acc = self.compute_acceleration(_speed)

        _speed = _speed[1:]

        intervals = []

        # start at index 2 due to shift of indices after deriving distance to acceleration
        for (sp, ac) in zip(_speed, acc):
            if _from <= sp[0] <= _to:
                if (_acceleration and ac > 0) or (not _acceleration and ac < 0):
                    intervals.append((sp[0], sp[1]))

        return intervals

    def find_intervals(self, _interval, _acceleration=True):
        """
        Splits a list of points into connected intervals.
        Points inside one interval are adjacent to each other.
        Speed increases/decreases along the interval.
        Intervals are separated by at least one point where speed decreases/increases.
        """
        # allow only intervals with this minimum length to be used for feature extraction.
        # Smaller intervals introduce high uncertainty nd numerical instability
        MIN_ALLOWED_INTERVAL_LENGTH = 2

        intervals = []
        pop = []
        for point in _interval:
            if len(pop) > 0 and ((_acceleration and point < pop[-1]) or (not _acceleration and point > pop[-1])):
                if len(pop) > MIN_ALLOWED_INTERVAL_LENGTH:
                    intervals.append(pop)

                pop = []

            pop.append(point)

        if len(pop) > MIN_ALLOWED_INTERVAL_LENGTH:
            intervals.append(pop)

        return intervals
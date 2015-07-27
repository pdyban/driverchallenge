
# coding: utf-8

# this function loads a specific trip
# parameters: driver = driver id 1-3612, index = trip number 1-200, pathToDriverData = path to the drivers folder

import numpy as np

def get_trip(driver, index, pathToDriverData):
    with open(pathToDriverData + '/%d/%d.csv' % (driver,index)) as f:
        trip = []
        for line in f:
            try:
                trip.append([float(i) for i in line.split(',')])
            except ValueError:
                continue
    return trip


def get_trip_npy(driver, index, pathToDriverData):
    trip = np.load(pathToDriverData + '/%d/%d.npy' % (driver,index))
    return trip


def list_all_drivers(pathToDriverData):
    import os

    drivers = []

    for f in os.listdir(pathToDriverData):
        try:
            int(f)
            drivers.append(int(f))
        except ValueError:
            pass

    return sorted(drivers)


# the following functions write a submission to file

from csv import writer

def write_submission_to_file(filename, features):
    """
    Creates a csv-formatted file with the following syntax:
    driver, trip, value -> driver_trip, value ? True : False
    """
    with open(filename, 'w') as f:
        w = writer(f)

        for line in features:
            index = '%d_%d' % (line[0], line[1])
            w.writerow([index] + ['%d' % line[2]])

    print 'submission file written to', filename


# prepare plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def save_3d_plot_to_file(x, y, z, c, submission_path):
    p = plt.subplot(1, 1, 1, projection='3d')
    p.scatter(x, y, z, c=c, marker='+')
    # save plot to file
    p.figure.savefig(submission_path)
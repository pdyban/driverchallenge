
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
import os


def write_submission_to_file(filename, features, zip=False):
    """
    Creates a csv-formatted file with the following syntax:
    driver, trip, value -> driver_trip, value ? True : False
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'w') as f:
        w = writer(f)
        w.writerow(['driver_trip', 'prob'])

        for line in features:
            index = '%d_%d' % (line[0], line[1])
            w.writerow([index] + ['%d' % line[2]])

    print 'submission file written to', filename

    if zip:
        from zipfile import ZipFile, ZIP_DEFLATED
        zip_filename = filename.replace('.csv', '.zip')

        with ZipFile(zip_filename, 'w', compression=ZIP_DEFLATED) as zf:
            zf.write(filename)
            print 'zipped submission to', zip_filename


# prepare plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def save_3d_plot_to_file(x, y, z, c, path):
    p = plt.subplot(1, 1, 1, projection='3d')
    p.scatter(x, y, z, c=c, marker='+')
    # save plot to file
    p.figure.savefig(path)


def save_1d_plot_to_file(x, c, path):
    #p = plt.subplot(1, 1, 1, projection='3d')
    #plt.hlines(1, min(x), max(x))  # Draw a horizontal line
    #plt.xlim(0,21)
    #plt.ylim(-0.5, 1.5)
    p = plt.subplot(1, 1, 1)
    p.scatter(x, c, marker='o')
    p.figure.savefig(path)

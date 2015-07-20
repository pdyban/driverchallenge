
# coding: utf-8

# this function loads a specific trip
# parameters: driver = driver id 1-3612, index = trip number 1-200, pathToDriverData = path to the drivers folder

# In[ ]:
import numpy

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
    trip = numpy.load(pathToDriverData + '/%d/%d.npy' % (driver,index))
    return trip
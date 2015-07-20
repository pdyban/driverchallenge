__author__ = 'igor'

import DriverDataIO as io
import numpy
import os

pathToDriverData = '../../driverchallenge_data/drivers'
pathToDriverDataOut = '../../driverchallenge_data/drivers_npy'

def list_all_drivers():
    import os

    drivers = []

    for f in os.listdir(pathToDriverData):
        try:
            int(f)
            drivers.append(int(f))
        except ValueError:
            pass

    return sorted(drivers)


print "started"

for driver in list_all_drivers():
    #create folder
    dirPath = pathToDriverDataOut + "/" + str(driver)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    for i in range(1, 201):  # all trips
        _trip = io.get_trip(driver, i, pathToDriverData)
        outPath = pathToDriverDataOut + "/" + str(driver) + "/" + str(i) + ".npy"
        numpy.save(outPath, _trip)


print "finished"
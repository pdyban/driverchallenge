
# coding: utf-8

# The idea is to use the angle to differentiate drivers. A more agressive driver might drive through a curve faster than a more calm driver.

# In[182]:

import DriverDataIO
import DriverChallengeHelperFunctions as dcHF
import DriverChallengeVisualization as dcVis
import AccelerationFeature
import numpy
import sklearn
import matplotlib.pyplot as plt

from math import *



# In[186]:

def angle_feature(ctrip):
    angles = []
    angle_feature_elements = []
    for index,p in list(enumerate(ctrip)):
        current_angle = dcHF.get_angle(index, ctrip)
        angles.append(current_angle)

        if current_angle != -1:
            length0 = dcHF.distance(ctrip[index-1], p)
            length1 = dcHF.distance(p, ctrip[index+1])
            driven_distance = length0 + length1
            feature_value = driven_distance * current_angle;
            angle_feature_elements.append(feature_value)
        else:
            angle_feature_elements.append(-1)
            
    #dcVis.plot_speed(angles)        
    return angle_feature_elements


# # computing feature for all drives

# In[190]:

def one_driver_angle_feature(driverid):
    feature_array = []
    for i in range(1,201):
        _trip = DriverDataIO.get_trip_npy(driverid,i, pathToDriverData)
        _angle_feature_elements = angle_feature(_trip)
        drive_mean = numpy.mean(_angle_feature_elements)

        feature_array.append((i, drive_mean))

    features = numpy.array(feature_array)


    return features


# In[191]:
    

# In[195]:

def find_outliers(_features):
    from sklearn import covariance
    clf = covariance.EllipticEnvelope(contamination=.1)
    
    clf.fit(_features)  
    y_pred = clf.decision_function(_features).ravel()

    # define a threshold for probabilities
    outliers_fraction = 0.02
    from scipy import stats
    threshold = stats.scoreatpercentile(y_pred, 100 * outliers_fraction)
    y_pred = y_pred > threshold
    
    return y_pred
    


# In[196]:

# In[ ]:




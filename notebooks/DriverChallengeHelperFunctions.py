
# coding: utf-8

# In[1]:

from math import *
def distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

def get_speed(trip):
    speed = []
    for i in range(len(trip)-1):
        speed.append( distance(trip[i], trip[i+1])*3.6 )
    return speed

def smooth_speed(speed, filter=9):
    """
    Smoothes signal using median filter and the given filter size.
    """
    from scipy.signal import medfilt
    return medfilt(speed, filter)

def get_acceleration(speed):
    """
    Computes acceleration for list of speed measurements.
    Acceleration is negative for breaking.
    @rtype list
    """
    acc = []
    for i in range(len(speed)-1):
        acc.append(speed[i+1] - speed[i])
    return acc


# In[ ]:




# this function get the angle at the point p, $$ arcos(\frac{(P_{12}^2 + P_{23}^2 - P_{13}^2) }{ (2 \cdot P_{12} \cdot P_{23}))}) $$

# In[3]:

def get_angle(p, trip):
    if(p > 0 and p < len(trip)-1):
        p1 = trip[p-1]
        p2 = trip[p]
        p3 = trip[p+1]
        p12 = distance(p1, p2)
        p13 = distance(p1, p3)
        p23 = distance(p2, p3)
        numerator = p12**2 + p23**2 - p13**2
        denomenator = 2 * p12 * p23
        if denomenator == 0:
            return -1
        #print numerator
        #print denomenator
        #print numerator/denomenator
        return acos(round(numerator/denomenator, 5))/pi
        #return round(numerator/denomenator, 5)
    return -1

# In[ ]:




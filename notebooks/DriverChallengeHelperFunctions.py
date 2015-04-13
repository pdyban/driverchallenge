
# coding: utf-8

# In[1]:

from math import sqrt
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




# In[ ]:




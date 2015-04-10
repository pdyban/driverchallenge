
# coding: utf-8

# In[ ]:

from math import sqrt
def distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

def get_speed(trip):
    speed = []
    for i in range(len(trip)-1):
        speed.append( distance(trip[i], trip[i+1])*3.6 )
    return speed



# coding: utf-8

# In[5]:

import DriverDataIO
import DriverChallengeHelperFunctions as dcHF
import DriverChallengeVisualization as dcVis
import AccelerationFeature


# In[2]:

pathToDriverData = '../../driverchallenge_data/drivers'


# In[3]:

dcVis.plot_driver(1,pathToDriverData,10,'b')


# Compute and plot speed over time, then compute acceleration

# In[8]:

speed = dcHF.get_speed(DriverDataIO.get_trip(1,2,pathToDriverData))


# In[54]:

print len(speed)


# In[9]:

get_ipython().magic(u'matplotlib')


# In[11]:

dcVis.plot_speed(speed)


# In[13]:

from scipy import signal


# In[14]:

filter_hz = 1e-3


# In[15]:

sampling_frequency = 1


# In[16]:

b, a = signal.butter(3, filter_hz / (sampling_frequency/2.), btype='low')


# In[17]:

sf = signal.lfilter(b, a, speed)


# No idea what a Butterworth filter does, but it has not helped.

# Filter speed signal using a median filter. We want to get rid of noise, peaks and speed irregularities.

# In[22]:

filtered_speed = signal.medfilt(speed, 9)


# In[23]:

dcVis.plot_compare(filtered_speed, speed)


# Smoothed filtered speed. Next step: select window 10 to 30 kmph.

# In[24]:

def find_window_30(speed):
    
    begin = None
    begin_time = None
    for second, item in enumerate(speed):
        if item > 10:
            continue
        begin = item
        begin_time = second
        break
        
    print 'speed', begin, 'at', begin_time, 'sec'
    
    for second, item in enumerate(speed[begin+1:], start=begin_time+1):    
        if item > 10:
            begin = item
            begin_time = second
            break
    
    for second, item in enumerate(speed[begin+1:], start=begin_time+1):    
        if item > 30:
            end = item
            end_time = second
            break
        
    print 'speed', begin, 'at', begin_time, 'sec', end, 'at', end_time


# In[25]:

find_window_30(filtered_speed)


# Here, we have computed the interval where the driver's speed exceeded 10 kmph and achieved 30 kmph. This is a 10-to-30 speed window. Next, we want to measure the acceleration.

# In[26]:

def get_acceleration(speed_interval):
    acc = []
    for i in range(len(speed_interval)-1):
        acc.append(speed_interval[i+1] - speed_interval[i])
    return acc


# In[27]:

print get_acceleration(filtered_speed[14:24])


# This is the acceleration in the first 10-to-30 interval.

# In[28]:

def plot_acc(acc):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(range(len(acc)),acc,'-', color='b')
            
    plt.show()


# In[29]:

plot_acc(get_acceleration(filtered_speed[14:24]))


# Next step will be to analyze acceleration across all 10-to-30 intervals and compute some scalar metric, e.g. acceleration time. Then, compute the same metric for other speed intervals, 50 to 70, 70 to 100, 100 to 120 and above.

# Ideas for acceleration features:
# - average (mean, median) acceleration in speed intervals (10 to 30, 30 to 50 etc.),
# - acceleration variability in speed intervals,
# - the same features for decceleration.

# In[5]:




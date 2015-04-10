
# coding: utf-8

# In[5]:

def plot_driver(driver, pathToDriverData, limit=200, color='b'):
    import DriverDataIO
    import matplotlib.pyplot as plt
    trips = range(1,limit+1)
    fig, ax = plt.subplots()

    for index in trips:
        trip = DriverDataIO.get_trip(driver, index, pathToDriverData)
        ax.plot([i[0] for i in trip],[i[1] for i in trip],'-', color='b')
            
    plt.show()


# In[ ]:

def plot_speed(speed):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(range(len(speed)),speed,'-', color='b')
            
    plt.show()


# used to compare two plots e.g. original and post-processed trip

# In[ ]:

def plot_compare(before, after):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(range(len(before)),before,'-', color='b')
    ax.plot(range(len(after)),after,'-', color='r')
            
    plt.show()


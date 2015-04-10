
# coding: utf-8

# In[ ]:

def get_acceleration(speed_interval):
    acc = []
    for i in range(len(speed_interval)-1):
        acc.append(speed_interval[i+1] - speed_interval[i])
    return acc


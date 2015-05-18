
# coding: utf-8

# In[22]:

from csv import writer

def create_submission(filename, features):
    """
    Creates a csv-formatted file with the following syntax:
    driver, trip, value -> driver_trip, value ? True : False
    """
    with open(filename, 'w') as f:
        w = writer(f)
        #w.writerow(['driver_trip', 'prob'])
        for line in features:
            index = '%d_%d' % (line[0], line[1])
            w.writerow([index] + ['%d' % line[2]])
    
    print 'submission file written to', filename


# In[6]:

# test submission


# In[23]:

if __name__ == '__main__':
    features = [
     [ 1, 1.,          True],
     [ 1, 2.,          False],
     [ 1, 3. ,         False],
     [ 1, 4. ,         1],
     [ 1, 5. ,         True],
     [ 2, 6. ,         True],
     [ 2, 7. ,         False],
     [ 2, 8. ,         0.0],
     [ 2, 9. ,         False]]
    create_submission('/tmp/submission_test.csv', features)


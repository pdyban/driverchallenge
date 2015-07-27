
import numpy as np

from features import AccelerationFeature
import utils

#submission = __import__("Submission")
#scigor = __import__("Session 4 scigor side curvature speed feature")

# submission result: 0.52
#features = [(10, 30, True, np.median), (31, 50, True, np.median), (51, 80, True, np.median), (10, 30, False, np.median)]
#pdyban.create_complete_submission(features, parallel=True)

# submission result: 0.51
#features = [(10, 30, True, np.std), (31, 50, True, np.std), (51, 80, True, np.std), (10, 30, False, np.std)]
#pdyban.create_complete_submission(features, parallel=True)

# submission result: 0.47
#features = [(10, 30, True, np.median), (31, 50, True, np.median), (51, 80, True, np.median), (10, 30, False, np.median)]
#features += [(10, 30, True, np.std), (31, 50, True, np.std), (51, 80, True, np.std), (10, 30, False, np.std)]
#pdyban.create_complete_submission(features, parallel=True)

# submission result
#features = []

features = [AccelerationFeature(10, 31, True, np.median)]


utils.create_complete_submission(features, parallel=True)

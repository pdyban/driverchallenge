# Driver Telematics Analysis
Driver Telematics Analysis is a Kaggle challenge. For more details, see the [challenge page](http://www.kaggle.com/c/axa-driver-telematics-analysis/data). Besides solving a machine learning problem, we want to learn how to use git and [scikit-learn](http://scikit-learn.org/stable/).

Submissions can be generated by running scripts from ``scripts`` directory, using root as working directory. Features implement a common interface and are stored inside ``features`` package. Utilities like plotting, i/o are part of ``utils`` package. Working notes are stored as [IPython notebooks](http://nbviewer.ipython.org) in ``notebooks`` directory.

# Todo:
- replace power computation in the distance function with numpy's squared distance (should be more efifcient)
- plot angle feature in color over trip track
- plot angle feature histograms, before and after RDP
- compute best RDP epsilon value
- create script that reduces trips using RDP and stores them as *.npy
- run RDP, recompute angle feature submissions
- analyze [article by Olariu](http://webmining.olariu.org/kaggle-driver-telematics/)

# Useful Links
- [A nice ebook explains how to use Git](http://www.git-tower.com/learn/ebook/command-line/introduction)
- [A Git commands cheat sheet](http://www.git-tower.com/blog/git-cheat-sheet/)
- [Learn Git in 15 minutes with an interactive shell](https://try.github.io/levels/1/challenges/1)
- [Advanced Git crash course in another 15 minutes](http://gitreal.codeschool.com/enroll)
- [List of all algorithms in scikit-learn](http://scikit-learn.org/dev/user_guide.html)
- [2nd place interview](http://blog.kaggle.com/2015/04/20/axa-winners-interview-learning-telematic-fingerprints-from-gps-data/)
- [GMM, EM](http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
- [Clustering](http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

# Scientific papers
- [Elliptic Envelope](http://www.geo.upm.es/postgrado/CarlosLopez/papers/FastAlgMCD99.pdf)

Features:
- AccelerationFeature(10, 31, True, np.median),
- AccelerationFeature(30, 51, True, np.median),
- AccelerationFeature(50, 71, True, np.median),
- AccelerationFeature(5, 130, True, np.median),
- AccelerationFeature(10, 31, True, np.mean),
- AccelerationFeature(30, 51, True, np.mean),
- AccelerationFeature(50, 71, True, np.mean),
- AccelerationFeature(5, 130, True, np.mean),
- AccelerationFeature(10, 31, False, np.median),
- AccelerationFeature(30, 51, False, np.median),
- AccelerationFeature(50, 71, False, np.median),
- AccelerationFeature(5, 130, False, np.median),
- AccelerationFeature(10, 31, False, np.mean),
- AccelerationFeature(30, 51, False, np.mean),
- AccelerationFeature(50, 71, False, np.mean),
- AccelerationFeature(5, 130, False, np.mean),
- AngleFeature(0, np.mean),
- AngleFeature(1, np.mean),
- SpeedPercentile(5),

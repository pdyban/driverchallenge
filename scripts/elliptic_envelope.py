__author__ = 'missoni'


def fit_elliptic_envelope(_features, outliers_fraction=0.02, plot=False):
    from sklearn import covariance

    clf = covariance.EllipticEnvelope(contamination=.1)

    features_only = np.c_[_features[:, 2], _features[:, 3]]

    clf.fit(features_only)
    y_pred = clf.decision_function(features_only).ravel()

    # define a threshold for probabilities
    from scipy import stats
    threshold = stats.scoreatpercentile(y_pred, 100 * outliers_fraction)
    y_pred = y_pred > threshold

    # plot results
    if plot:
        plt.scatter([i[0] for i in features_only], [i[1] for i in features_only], c=y_pred, cmap='cool')

    return np.c_[_features[:,0], _features[:,1], y_pred]
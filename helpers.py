import scipy as sp
import pandas as pd
import numpy as np


from scipy.stats import chi2
from sklearn.covariance import MinCovDet


def robust_mahalanobis_params(df):
    # Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(df.values.T)
    X = rng.multivariate_normal(mean=np.mean(
        df, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_  # robust covariance metric
    robust_mean = cov.location_  # robust mean
    inv_covmat = sp.linalg.inv(mcd)  # inverse covariance metric

    return robust_mean,  inv_covmat


def robust_mahalanobis_method(df):
    # Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(df.values.T)
    X = rng.multivariate_normal(mean=np.mean(
        df, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_  # robust covariance metric
    robust_mean = cov.location_  # robust mean
    inv_covmat = sp.linalg.inv(mcd)  # inverse covariance metric

    # Robust M-Distance
    x_minus_mu = df - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    #print('robust_mean', robust_mean.shape)
    #print('x_minus_mu', x_minus_mu.shape)
    #print('inv_covmat', inv_covmat.shape)
    #print('left_term', left_term.shape)
    #print('mahal', mahal.shape)
    #print('md', md.shape)

    # Flag as outlier
    outlier = []
    # degrees of freedom = number of variables
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md


def mahalanobis_method(df):
    # M-Distance
    x_minus_mu = df - np.mean(df)
    cov = np.cov(df.values.T)  # Covariance
    inv_covmat = sp.linalg.inv(cov)  # Inverse covariance
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    # Flag as outlier
    outlier = []
    # Cut-off point
    # degrees of freedom = number of variables
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md

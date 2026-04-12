import sys
import os
import time
import numpy as np
import scipy.stats
import sklearn
from numpy.linalg import inv
import scipy.stats

########## Function to import netcdf files

from netCDF4 import Dataset

def load_data(path_to_data, climate_variable):

        nc_fid = Dataset(path_to_data, 'r');
        climate_field = nc_fid.variables[climate_variable][:];
        return climate_field

def masked_array_to_numpy(data):
    return np.ma.filled(data.astype(np.float32), np.nan);

def get_nonmask_indices(data):
    return np.argwhere(~np.isnan(np.sum(data,axis=0)));

########## REMOVE MEAN

# x_t is an array with n time series of length T
# format: np.shape(xt) = (n, T)

def remove_mean(x_t):

    # x_t is an array with n time series of length T
    # format: np.shape(xt) = (n, T)
    d = np.shape(x_t)[0]
    y_t = x_t.copy()
    for i in range(d):
        y_t[i] = y_t[i] - np.mean(y_t[i])
    return y_t

########## STANDARDIZE

# x_t is an array with n time series of length T
# format: np.shape(xt) = (n, T)

def standardize(x_t):

    # xt is an array with n time series of length T
    # format: np.shape(xt) = (n, T)
    d = np.shape(x_t)[0]
    y_t = x_t.copy()
    for i in range(d):
        y_t[i] = y_t[i]/np.std(y_t[i])
    return y_t

########## COMPUTE LAG CORRELATIONS

# Here we are assuming that time series have been rescaled to
# zero mean and unit variance
# np.shape(x) = np.shape(y) = T (T being the length of the time series)

def lagged_correlation(x,y,tau):

    # Check that the two time series have
    # - the same length
    assert len(x) == len(y);
    #length of time series
    T = len(x);

    ##assert that lag can nit be greater than T
    assert tau < T;

    if(tau == 0):
        return np.dot(x,y)/T;
    if(tau > 0):
        x = x[0:T-tau];
        y = y[tau:T]
        return np.dot(x,y)/T;

    if(tau < 0):
        tau = np.abs(tau)
        y = y[0:T-tau];
        x = x[tau:T]
        return np.dot(x,y)/T;

########## LAG 1 AUTOCORRELATIONS

# Compute lag-1 autocorrelation of an orbit x_t
# Orbit must have this shape: 
# np.shape(x_t) = (n, T) (n time series of length T)

def phi_vector(x_t):
    # we scale the process to zero mean
    x_t = remove_mean(x_t)
    # we scale the process to unit variance
    x_t = standardize(x_t)
    # Compute lag-1 correlation
    phi = []
    for i in range(len(x_t)):
        phi.append(lagged_correlation(x_t[i],x_t[i],1))
    phi = np.array(phi)
    return phi

########## COMPUTE STANDARD DEVIATIONS

# Compute the sigmas of each time series in xt
# np.shape(x_t) = (n, T) (n time series of length T)

def sigmas(x_t):
    # we scale the process to zero mean
    x_t = remove_mean(x_t)
    # Compute standard deviation of each time series
    sigmas = np.std(x_t,axis = 1)
    return sigmas




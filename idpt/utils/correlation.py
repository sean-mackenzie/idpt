# import modules
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import curve_fit
from skimage.feature import match_template
import numpy as np

import logging

logger = logging.getLogger(__name__)


def sk_norm_cross_correlation(img1, img2):
    if img1.size >= img2.size:
        result = match_template(img1, img2)
    else:
        logger.warning(
            "Unable to correlate mismatched templates: (img1, img2): ({}, {})".format(img1.shape, img2.shape))
        result = np.nan
    return result


def akima_interpolation(z_calib, sim, max_idx):
    # find index of maximum image correlation
    x_interp = z_calib
    y_interp = sim

    if len(z_calib) < 3 or len(sim) < 3:
        return z_calib, sim

    # determine the bounds of the fit
    lower_index = np.maximum(0, max_idx - 1)
    upper_index = np.minimum(max_idx + 1, len(z_calib) - 1)

    if lower_index >= len(z_calib) - 2:
        lower_index = len(z_calib) - 3
    if upper_index < 2:
        upper_index = 2

    # fit Akima cubic polynomial
    x_local = np.linspace(z_calib[lower_index], z_calib[upper_index], 50)
    sim_interp = Akima1DInterpolator(x_interp, y_interp)(x_local)

    return x_local, sim_interp


def parabolic_interpolation(z_calib, sim, max_idx):
    # if there are only two values, we cannot fit a three-point estimator
    if len(z_calib) < 3 or len(sim) < 3:
        return z_calib, sim

    # determine the bounds of the fit
    lower_index = np.maximum(0, max_idx - 1)
    upper_index = np.minimum(max_idx + 1, len(z_calib) - 1)

    if lower_index >= len(z_calib) - 2:
        lower_index = len(z_calib) - 3
    if upper_index < 2:
        upper_index = 2

    lower_bound = z_calib[lower_index]
    upper_bound = z_calib[upper_index]

    # fit parabola
    popt, pcov = curve_fit(parabola, z_calib[lower_index:upper_index + 1], sim[lower_index:upper_index + 1])

    # create interpolation space and get resulting parabolic curve
    x_local = np.linspace(lower_bound, upper_bound, 50)
    sim_interp = parabola(x_local, *popt)

    return x_local, sim_interp


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c
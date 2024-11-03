# import modules

from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import curve_fit
from skimage.feature import match_template
import numpy as np

from idpt.utils.subresolution import fit_2d_gaussian_on_corr

import logging

logger = logging.getLogger(__name__)


def get_similarity_function(function):
    """

    :param function:
    :return:
    """
    if function == 'sknccorr':
        sim_func = sk_norm_cross_correlation
        optim = np.argmax
    else:
        raise ValueError("Unknown similarity function {}".format(function))
    return sim_func, optim


def sk_norm_cross_correlation(img1, img2):
    """

    :param img1:
    :param img2:
    :return:
    """
    if img1.size >= img2.size:
        result = match_template(img1, img2)
    else:
        logger.warning(
            "Unable to correlate mismatched templates: (img1, img2): ({}, {})".format(img1.shape, img2.shape))
        result = np.nan
    return result


def correlate_against_stack(template, stack, sim_func, optim=None):
    """

    :param template:
    :param stack:
    :param sim_func:
    :param optim:
    :return:
    """
    if isinstance(sim_func, str):
        sim_func, optim = get_similarity_function(sim_func)

    response_stack = []
    similarity_stack = []
    for i in stack:
        response_image = sim_func(i, template)
        similarity_stack.append(np.max(response_image))
        response_stack.append(response_image)

    idx_peak_correlation = optim(similarity_stack)

    return idx_peak_correlation, similarity_stack, response_stack


def localize_discrete(idx_peak_correlation, similarity_stack, response_stack, z_calib):
    """

    :param idx_peak_correlation:
    :param similarity_stack:
    :param response_stack:
    :param z_calib:
    :return:
    """
    correlation_coefficient = similarity_stack[idx_peak_correlation]
    z_discrete = z_calib[idx_peak_correlation]

    # x-y discrete

    # get the correlation map where peak correlation was found
    response_image = response_stack[idx_peak_correlation]
    res_length = np.floor(response_image.shape[0] / 2)

    # x,y coordinates in the image space where the highest correlation was found
    ij = np.unravel_index(np.argmax(response_image), response_image.shape)
    xmt, ymt = ij[::-1]
    dx_discrete = res_length - xmt
    dy_discrete = res_length - ymt

    if dx_discrete is None and dy_discrete is None:
        dx_discrete, dy_discrete = 0, 0

    return correlation_coefficient, dx_discrete, dy_discrete, z_discrete


def localize_subresolution(idx_peak_correlation, similarity_stack, response_stack, z_calib, optim):
    """

    :param idx_peak_correlation:
    :param similarity_stack:
    :param response_stack:
    :param z_calib:
    :param optim:
    :return:
    """
    # sub-resolution z-localization
    z_interp, sim_interp = parabolic_interpolation(z_calib, similarity_stack, idx_peak_correlation)
    z_subresolution = z_interp[optim(sim_interp)]
    similarity_subresolution = sim_interp[optim(sim_interp)]

    # sub-resolution xy-localization

    # get the correlation map where peak correlation was found
    response_image = response_stack[idx_peak_correlation]
    res_length = np.floor(response_image.shape[0] / 2)

    # x,y coordinates in the image space where the highest correlation was found
    ij = np.unravel_index(np.argmax(response_image), response_image.shape)
    xmt, ymt = ij[::-1]

    xg, yg = None, None
    result = response_image - np.min(response_image)
    pad_width = 0

    if np.size(result) > 5:
        xgt, ygt = fit_2d_gaussian_on_corr(result, xmt + pad_width, ymt + pad_width)
        if xgt is not None and ygt is not None:
            xg = res_length - xgt
            yg = res_length - ygt

    if xg is None and yg is None:
        xg, yg = 0, 0
    dx_subresolution, dy_subresolution = xg, yg

    return similarity_subresolution, dx_subresolution, dy_subresolution, z_subresolution


def akima_interpolation(z_calib, sim, max_idx):
    """

    :param z_calib:
    :param sim:
    :param max_idx:
    :return:
    """
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
    """

    :param z_calib:
    :param sim:
    :param max_idx:
    :return:
    """
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
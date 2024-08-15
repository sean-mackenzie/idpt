
import math
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import iqr, gaussian_kde
from sklearn.neighbors import KernelDensity
import functools


def fit_line(x, a, b):
    return a * x + b


def fit_kde(y, bandwidth=None):
    """ kdex, kdey, bandwidth = fit_kde(y, bandwidth=None) """

    if bandwidth is None:
        """ Silverman's rule of thumb: https://en.wikipedia.org/wiki/Kernel_density_estimation """
        bandwidth = 0.9 * np.min([np.std(y), iqr(y) / 1.34]) * len(y) ** (-1 / 5)

    # get extents of range that KDE will evaluate over
    ymin, ymax = np.min(y), np.max(y)
    y_range = ymax - ymin

    # setup arrays
    y = y[:, np.newaxis]
    y_plot = np.linspace(ymin - y_range / 12.5, ymax + y_range / 12.5, 300)[:, np.newaxis]

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(y)
    log_dens_y = kde.score_samples(y_plot)

    kdex = y_plot[:, 0]
    kdey = np.exp(log_dens_y)

    return kdex, kdey, bandwidth


def kde_scipy(y, y_grid, bandwidth=0.2, **kwargs):
    """
    pdf, y_grid = kde_scipy(y, y_grid, bandwidth=0.2, **kwargs)

    Reference: https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    """

    if y_grid is None:
        ymin, ymax = np.min(y), np.max(y)
        y_range = ymax - ymin
        y_grid = np.linspace(ymin - y_range / 10, ymax + y_range / 10, 200)

    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(y, bw_method=bandwidth / y.std(ddof=1), **kwargs)
    return kde.evaluate(y_grid), y_grid


def fit_3d_plane(points):
    fun = functools.partial(plane_error, points=points)
    params0 = np.array([0, 0, 0])
    res = minimize(fun, params0)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]

    point = np.array([0.0, 0.0, c])
    normal = np.array(cross([1, 0, a], [0, 1, b]))
    d = -point.dot(normal)

    popt = [a, b, c, d, normal]

    minx = np.min(points[:, 0])
    miny = np.min(points[:, 1])
    maxx = np.max(points[:, 0])
    maxy = np.max(points[:, 1])

    xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    return xx, yy, z, popt


def fit_in_focus_plane(df, param_zf, microns_per_pixel, img_xc, img_yc):
    """ dict_fit_plane = fit.fit_in_focus_plane(df, param_zf, microns_per_pixel, img_xc, img_yc) """

    if not len(df) == len(df.id.unique()):
        df = df.groupby('id').mean().reset_index()

    # fitting stats:
    num_locations = len(df)
    x_span = df.x.max() - df.x.min()
    y_span = df.y.max() - df.y.min()
    num_density = num_locations / (x_span * y_span)

    zf_mean_of_points = df[param_zf].mean()
    zf_std_of_points = df[param_zf].std()

    # fit plane (x, y, z units: pixels)
    points_pixels = np.stack((df.x, df.y, df[param_zf])).T
    px_pixels, py_pixels, pz_pixels, popt_pixels = fit_3d_plane(points_pixels)
    d, normal = popt_pixels[3], popt_pixels[4]

    # calculate fit error
    fit_results = calculate_z_of_3d_plane(df.x, df.y, popt=popt_pixels)
    rmse, r_squared = calculate_fit_error(fit_results, data_fit_to=df[param_zf].to_numpy())

    # fit plane (x, y, z units: microns) to calculate tilt angle
    points_microns = np.stack((df.x * microns_per_pixel, df.y * microns_per_pixel, df[param_zf])).T
    px_microns, py_microns, pz_microns, popt_microns = fit_3d_plane(points_microns)
    tilt_x = np.rad2deg(np.arctan((pz_microns[0, 1] - pz_microns[0, 0]) / (px_microns[0, 1] - px_microns[0, 0])))
    tilt_y = np.rad2deg(np.arctan((pz_microns[1, 0] - pz_microns[0, 0]) / (py_microns[1, 0] - py_microns[0, 0])))

    # calculate zf at image center: (x = 256, y = 256)
    zf_mean_image_center = calculate_z_of_3d_plane(img_xc, img_yc, popt=popt_pixels)

    dict_fit_plane = {'z_f': param_zf,
                      'z_f_fit_plane_image_center': zf_mean_image_center,
                      'z_f_mean_points': zf_mean_of_points,
                      'z_f_std_points': zf_std_of_points,
                      'rmse': rmse,
                      'r_squared': r_squared,
                      'tilt_x_degrees': tilt_x, 'tilt_y_degrees': tilt_y,
                      'num_locations': num_locations,
                      'num_density_pixels': num_density, 'num_density_microns': num_density / microns_per_pixel ** 2,
                      'x_span_pixels': x_span, 'x_span_microns': x_span * microns_per_pixel,
                      'y_span_pixels': y_span, 'y_span_microns': y_span * microns_per_pixel,
                      'popt_pixels': popt_pixels,
                      'px': px_pixels, 'py': py_pixels, 'pz': pz_pixels,
                      'd': d, 'normal': normal,
                      }

    return dict_fit_plane


def plane_error(params, points):
    result = 0
    for (x, y, z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff ** 2
    return result


def calculate_z_of_3d_plane(x, y, popt):
    """
    Calculate the z-coordinate of a point lying on a 3D plane.

    :param x:
    :param y:
    :param popt:
    :return:
    """

    a, b, c, d, normal = popt[0], popt[1], popt[2], popt[3], popt[4]

    z = (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]

    return z


def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a * x + b * y + c
    return z


# ---------------------------------------------- HELPER FUNCTIONS ------------------------------------------------------


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]

def calculate_fit_error(fit_results, data_fit_to, fit_func=None, fit_params=None, data_fit_on=None):
    """
    To run:
    rmse, r_squared = fit.calculate_fit_error(fit_results, data_fit_to)

    Two options for calculating fit error:
        1. fit_func + fit_params: the fit results are calculated.
        2. fit_results: the fit results are known for each data point.

    Old way of doing this (updated 6/11/22):
    abs_error = fit_results - data_fit_to
    r_squared = 1.0 - (np.var(abs_error) / np.var(data_fit_to))

    :param fit_func: the function used to calculate the fit.
    :param fit_params: generally, popt.
    :param fit_results: the outputs at each input data point ('data_fit_on')
    :param data_fit_on: the input data that was inputted to fit_func to generate the fit.
    :param data_fit_to: the output data that fit_func was fit to.
    :return:
    """

    # --- calculate prediction errors
    if fit_results is None:
        fit_results = fit_func(data_fit_on, *fit_params)

    residuals = calculate_residuals(fit_results, data_fit_to)
    r_squared_me = 1 - (np.sum(np.square(residuals))) / (np.sum(np.square(fit_results - np.mean(fit_results))))

    se = np.square(residuals)  # squared errors
    mse = np.mean(se)  # mean squared errors
    rmse = np.sqrt(mse)  # Root Mean Squared Error, RMSE
    r_squared = 1.0 - (np.var(np.abs(residuals)) / np.var(data_fit_to))

    #TODO: should investigate this
    # print("wiki r-squared: {}; old r-squared: {}".format(np.round(r_squared_me, 4), np.round(r_squared, 4)))
    # I think the "wiki r-squared" is probably the correct one...
    # 8/23/22 - the wiki is definitely wrong because values range from +1 to -20...

    return rmse, r_squared

# ------------------------------------- STATISTICAL ANALYSIS FUNCTIONS -------------------------------------------------


def calculate_residuals(fit_results, data_fit_to):
    residuals = fit_results - data_fit_to
    return residuals


def calculate_precision(arr):
    return np.std(arr - np.mean(arr))


def calculate_coefficient_of_variation(arr):
    return np.std(arr) / np.mean(arr)
# utils.subresolution.py

# import modules
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import rotate


def gauss_1d_function(x, a, x0, sigma):
    """

    :param x:
    :param a:
    :param x0:
    :param sigma:
    :return:
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_2d_function(xy, a, x0, y0, sigmax, sigmay):
    """

    :param xy:
    :param a:
    :param x0:
    :param y0:
    :param sigmax:
    :param sigmay:
    :return:
    """
    return a * np.exp(-((xy[:, 0] - x0) ** 2 / (2 * sigmax ** 2) + (xy[:, 1] - y0) ** 2 / (2 * sigmay ** 2)))


def bivariate_gaussian_pdf(xy, a, x0, y0, sigmax, sigmay, rho):
    """

    :param xy:
    :param a:
    :param x0:
    :param y0:
    :param sigmax:
    :param sigmay:
    :param rho:
    :return:
    """
    return a * np.exp(
        -((1 / (2 * (1 - rho ** 2))) * ((xy[:, 0] - x0) ** 2 / sigmax ** 2 - 2 * rho * (xy[:, 0] - x0) * (
                    xy[:, 1] - y0) / (sigmax * sigmay) +
          (xy[:, 1] - y0) ** 2 / sigmay ** 2)
          )
    )


def bivariate_gaussian_pdf_bkg(xy, a, x0, y0, sigmax, sigmay, rho, bkg):
    """

    :param xy:
    :param a:
    :param x0:
    :param y0:
    :param sigmax:
    :param sigmay:
    :param rho:
    :param bkg:
    :return:
    """
    return a * np.exp(
        -((1 / (2 * (1 - rho ** 2))) * ((xy[:, 0] - x0) ** 2 / sigmax ** 2 - 2 * rho * (xy[:, 0] - x0) * (
                    xy[:, 1] - y0) / (sigmax * sigmay) +
          (xy[:, 1] - y0) ** 2 / sigmay ** 2)
          )
    ) + bkg


def fit_2d_gaussian_on_corr(res, xc, yc):
    """

    :param res:
    :param xc:
    :param yc:
    :return:
    """
    scaling_factor = 1

    # make grid
    X = np.arange(np.shape(res)[1])
    Y = np.arange(np.shape(res)[0])
    X, Y = np.meshgrid(X, Y)

    # flatten arrays
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = res.flatten()

    # stack for gaussian curve fitting
    XYZ = np.stack([Xf.flatten(), Yf.flatten(), Zf.flatten()]).T

    # fit 2D gaussian
    guess = [1, xc * scaling_factor, yc * scaling_factor, 1.5 * scaling_factor, 1.5 * scaling_factor]

    try:
        popt, pcov = curve_fit(gauss_2d_function, XYZ[:, :2], XYZ[:, 2], p0=guess)
        A, xc, yc, sigmax, sigmay = popt

    except RuntimeError:
        xc, yc = None, None

    return xc, yc


def fit_2d_gaussian_on_image(img, normalize=True, guess='sigma_improved', rotate_degrees=0, bivariate_pdf=False):
    """

    :param img:
    :param normalize:
    :param guess:
    :param rotate_degrees:
    :param bivariate_pdf:
    :return:
    """
    if rotate_degrees != 0:
        img = rotate(img, angle=rotate_degrees, reshape=False, mode='grid-constant', cval=np.percentile(img, 5))

    if normalize:
        img = img - img.min() + 1

    y, x = np.shape(img)
    xc, yc = x // 2, y // 2

    bounds = None
    if guess == 'center':
        guess_A = np.max(img) / 2
        guess_sigma = xc / 2
        bounds = ([0, x / 8, y / 8, 0, 0], [2 ** 16, 7 * x / 8, 7 * y / 8, x, y])
    elif guess == 'sigma_improved':
        guess_A, guess_c, guess_sigma = get_amplitude_center_sigma_improved(x_space=None, y_profile=None, img=img)
    else:
        raise ValueError()

    # make grid
    X = np.arange(np.shape(img)[1])
    Y = np.arange(np.shape(img)[0])
    X, Y = np.meshgrid(X, Y)

    # flatten arrays
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = img.flatten()

    # stack for gaussian curve fitting
    XYZ = np.stack([Xf.flatten(), Yf.flatten(), Zf.flatten()]).T

    # fit 2D gaussian
    guess = [guess_A, xc, yc, guess_sigma, guess_sigma]

    try:
        if bivariate_pdf is True and normalize is False:
            guess = [guess_A, xc, yc, guess_sigma, guess_sigma, 0, 100]
            try:
                popt, pcov = curve_fit(bivariate_gaussian_pdf_bkg, XYZ[:, :2], XYZ[:, 2],
                                       guess,
                                       bounds=([0, 0, 0, 0, 0, -0.99, 0],
                                               [2**16, 512, 512, 100, 100, 0.99, 2**16])
                                       )
            except ValueError:
                pass

        elif bivariate_pdf:
            guess = [guess_A, xc, yc, guess_sigma, guess_sigma, 0]
            popt, pcov = curve_fit(bivariate_gaussian_pdf, XYZ[:, :2], XYZ[:, 2], guess)
        elif bounds is not None:
            popt, pcov = curve_fit(gauss_2d_function, XYZ[:, :2], XYZ[:, 2], p0=guess, bounds=bounds)
        else:
            popt, pcov = curve_fit(gauss_2d_function, XYZ[:, :2], XYZ[:, 2], guess)
    except RuntimeError:
        popt = None

    return popt


# --------------------------------------------- HELPER FUNCTIONS -------------------------------------------------------


def evaluate_fit_2d_gaussian_on_image(img, fit_func, popt):
    """

    :param img:
    :param fit_func:
    :param popt:
    :return:
    """

    XYZ = flatten_image(img)

    # 2D Gaussian from fit
    if fit_func == 'bivariate_pdf':
        fZ = bivariate_gaussian_pdf_bkg(XYZ[:, :2], *popt)
    else:
        raise ValueError("Currently only 'bivariate_pdf' is implemented.")

    # data used for fitting
    img_arr = XYZ[:, 2]

    # get residuals
    residuals = calculate_residuals(fit_results=fZ, data_fit_to=img_arr)

    # get goodness of fit
    rmse, r_squared = calculate_fit_error(fit_results=fZ, data_fit_to=img_arr)

    return XYZ, fZ, rmse, r_squared, residuals


def fit_gaussian_calc_diameter(img, normalize=True, rotate_degrees=0, bivariate_pdf=False):
    """

    :param img:
    :param normalize:
    :param rotate_degrees:
    :param bivariate_pdf:
    :return:
    """
    popt = fit_2d_gaussian_on_image(img,
                                    normalize=normalize,
                                    guess='sigma_improved',
                                    rotate_degrees=rotate_degrees,
                                    bivariate_pdf=bivariate_pdf,
                                    )

    if popt is None:
        return None, None, None, None, None, None, None, None, None
    elif len(popt) == 5:
        popt = np.append(popt, None)
        popt = np.append(popt, None)

    A, xc, yc, sigmax, sigmay, rho, bkg = popt

    dia_x, dia_y = calc_diameter_from_theory(img, A, xc, yc, sigmax, sigmay)

    return dia_x, dia_y, A, yc, xc, sigmay, sigmax, rho, bkg


def calc_diameter_from_theory(img, A, xc, yc, sigmax, sigmay):
    """

    :param img:
    :param A:
    :param xc:
    :param yc:
    :param sigmax:
    :param sigmay:
    :return:
    """
    beta = np.sqrt(3.67)

    # diameter threshold: intensity < np.exp(-3.67 ** 2)
    diameter_threshold = np.exp(-1 * beta ** 2) * A

    # spatial arrays
    x_arr = np.linspace(0, sigmax * 5, 1000)
    y_arr = np.linspace(0, sigmay * 5, 1000)

    # xy-radius intensity distribution (fitted Gaussian distribution)
    x_intensity = gauss_1d_function(x=x_arr, a=A, x0=0, sigma=sigmax)
    y_intensity = gauss_1d_function(x=y_arr, a=A, x0=0, sigma=sigmay)

    # find where intensity distribution, I_xy(x, y) < np.exp(-3.67 ** 2) * maximum intensity at the center
    # NOTE: is "maximum intensity at the center" defined as A (fitted Gaussian amplitude) or peak pixel intensity?
    x_intensity_raw = x_intensity - np.exp(-1 * beta ** 2) * A
    y_intensity_raw = y_intensity - np.exp(-1 * beta ** 2) * A
    x_intensity_rel = np.abs(x_intensity_raw)
    y_intensity_rel = np.abs(y_intensity_raw)

    # radius (in pixels) is equal to xy-coordinate where xy_intensity_rel is minimized
    radius_x = x_arr[np.argmin(x_intensity_rel)]
    radius_y = y_arr[np.argmin(y_intensity_rel)]

    dia_x = radius_x
    dia_y = radius_y

    return dia_x, dia_y


def get_amplitude_center_sigma_improved(x_space=None, y_profile=None, img=None):
    """

    :param x_space:
    :param y_profile:
    :param img:
    :return:
    """
    if y_profile is None:
        y_slice = np.unravel_index(np.argmax(img, axis=None), img.shape)[0]
        y_profile = img[y_slice, :]

    if x_space is None:
        x_space = np.arange(len(y_profile))

    # get amplitude
    raw_amplitude = y_profile.max() - y_profile.min()

    # get center
    raw_c = x_space[np.argmax(y_profile)]

    # get sigma
    y_pl_zero = len(np.where(y_profile[:np.argmax(y_profile)] - np.mean(y_profile) < 0)[0])
    y_pr_zero = len(np.where(y_profile[np.argmax(y_profile):] - np.mean(y_profile) < 0)[0])

    raw_sigma = np.mean([y_pl_zero, y_pr_zero])

    return raw_amplitude, raw_c, raw_sigma


def calculate_residuals(fit_results, data_fit_to):
    residuals = fit_results - data_fit_to
    return residuals


def calculate_fit_error(fit_results, data_fit_to, fit_func=None, fit_params=None, data_fit_on=None):
    """

    :param fit_results:
    :param data_fit_to:
    :param fit_func:
    :param fit_params:
    :param data_fit_on:
    :return:
    """

    # --- calculate prediction errors
    if fit_results is None:
        fit_results = fit_func(data_fit_on, *fit_params)

    residuals = calculate_residuals(fit_results, data_fit_to)

    se = np.square(residuals)  # squared errors
    mse = np.mean(se)  # mean squared errors
    rmse = np.sqrt(mse)  # Root Mean Squared Error, RMSE
    r_squared = 1.0 - (np.var(np.abs(residuals)) / np.var(data_fit_to))

    return rmse, r_squared


def flatten_image(img):
    """ XYZ = flatten_image(img) """

    # make grid
    X = np.arange(np.shape(img)[1])
    Y = np.arange(np.shape(img)[0])
    X, Y = np.meshgrid(X, Y)

    # flatten arrays
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = img.flatten()

    # stack for gaussian curve fitting
    XYZ = np.stack([Xf.flatten(), Yf.flatten(), Zf.flatten()]).T

    return XYZ
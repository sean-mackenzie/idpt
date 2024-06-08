# utils.subresolution.py
"""
A good reference for PSF-based z-determination: https://link.springer.com/content/pdf/10.1007/s00348-014-1809-2.pdf
"""

# import modules
import numpy as np
from scipy.optimize import curve_fit


def gauss_1d_function(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_2d_function(xy, a, x0, y0, sigmax, sigmay):
    return a * np.exp(-((xy[:, 0] - x0) ** 2 / (2 * sigmax ** 2) + (xy[:, 1] - y0) ** 2 / (2 * sigmay ** 2)))


def bivariate_gaussian_pdf(xy, a, x0, y0, sigmax, sigmay, rho):
    return a * np.exp(
        -((1 / (2 * (1 - rho ** 2))) * ((xy[:, 0] - x0) ** 2 / sigmax ** 2 - 2 * rho * (xy[:, 0] - x0) * (
                    xy[:, 1] - y0) / (sigmax * sigmay) +
          (xy[:, 1] - y0) ** 2 / sigmay ** 2)
          )
    )


def bivariate_gaussian_pdf_bkg(xy, a, x0, y0, sigmax, sigmay, rho, bkg):
    return a * np.exp(
        -((1 / (2 * (1 - rho ** 2))) * ((xy[:, 0] - x0) ** 2 / sigmax ** 2 - 2 * rho * (xy[:, 0] - x0) * (
                    xy[:, 1] - y0) / (sigmax * sigmay) +
          (xy[:, 1] - y0) ** 2 / sigmay ** 2)
          )
    ) + bkg


def fit_2d_gaussian_on_corr(res, xc, yc):
    # xc_original = xc
    # yc_original = yc

    scaling_factor = 1
    # res = rescale(res, scaling_factor)

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


        """
        popt, img_norm = processing.fit_2d_gaussian_on_image(img, normalize=True, bkg_mean=bkg_mean)
        A, xc, yc, sigmax, sigmay = popt

        # calculate the fit error
        XYZ, fZ, rmse, r_squared, residuals = processing.evaluate_fit_2d_gaussian_on_image(img_norm, popt)
        """

        # xc = xc / scaling_factor
        # yc = yc / scaling_factor

        # 3D plot similarity map and 2D Gaussian fit
        # evaluate_fit_gaussian_and_plot_3d(res, popt, scaling_factor)
        # print("Original ({}, {}); Fitted ({}, {})".format(xc_original, yc_original, np.round(xc, 3), np.round(yc, 3)))

    except RuntimeError:
        xc, yc = None, None

    return xc, yc
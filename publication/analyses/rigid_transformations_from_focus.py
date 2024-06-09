# publication/rigid_transformations_from_focus.py

import os
from os.path import join
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from publication.analyses.utils import iterative_closest_point as icp

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import scienceplots
# A note on SciencePlots colors
"""
Blue: #0C5DA5
Green: #00B945
Red: #FF9500
Orange: #FF2C00

Other Colors:
Light Blue: #7BC8F6
Paler Blue: #0343DF
Azure: #069AF3
Dark Green: #054907
"""

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'

plt.style.use(['science', 'ieee', 'std-colors'])  # 'ieee', 'std-colors'
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


# TODO: call ICP functions directly instead of these ones below? Are they copied exactly or similar?


def flatten_list_of_lists(l):
    return [item for sublist in l for item in sublist]


def pci_best_fit_transform(A, B):
    '''
    Reference: 2016 Clay Flannigan
    https://github.com/ClayFlannigan/icp

    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def pci_nearest_neighbor(src, dst):
    '''
    Reference: 2016 Clay Flannigan
    https://github.com/ClayFlannigan/icp

    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def pci_icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    Based on reference: 2016 Clay Flannigan
    https://github.com/ClayFlannigan/icp

    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = pci_nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = pci_best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = pci_best_fit_transform(A, src[:m, :].T)

    return T, distances, i, indices


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def distances_xyz(AR, BR, TR):
    """ xdists, ydists, zdists = distances_xyz(AR=, BR=) """

    # get number of dimensions
    m = AR.shape[1]
    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, AR.shape[0]))
    dst = np.ones((m + 1, BR.shape[0]))
    src[:m, :] = np.copy(AR.T)
    dst[:m, :] = np.copy(BR.T)

    # update the current source
    src_transformed = np.dot(TR, src)
    src_transformed = src_transformed.T

    # calculate x, y, and z distances (errors)
    x_distances = src_transformed[:, 0] - BR[:, 0]
    y_distances = src_transformed[:, 1] - BR[:, 1]
    z_distances = src_transformed[:, 2] - BR[:, 2]

    return x_distances, y_distances, z_distances


def rigid_transforms_from_focus(df_test, df_focus, min_num, distance_threshold, include_AA=True):
    # get z_true values and sort
    zts = df_test.z_true.sort_values(ascending=True).unique()
    z_range_rt = len(zts)

    # initialize
    data = []
    dfBB_icps = []

    for ii in range(z_range_rt):

        z_frames = df_test[df_test['z_nominal'] == zts[ii]].frame.unique()
        for fr in z_frames:

            dfA = df_focus.copy()
            dfB = df_test[(df_test['z_nominal'] == zts[ii]) & (df_test['frame'] == fr)].reset_index()
            dfA['z'] = 0

            A = dfA[['x', 'y', 'z']].to_numpy()
            B = dfB[['x', 'y', 'z']].to_numpy()

            i_num_A = len(A)
            i_num_B = len(B)

            # minimum number of particles per frame threshold
            if np.min([i_num_A, i_num_B]) < min_num:
                continue

            # match particle positions between A and B using NearestNeighbors
            if len(A) > len(B):
                ground_truth_xy = A[:, :2]
                ground_truth_pids = dfA.id.to_numpy()
                locations = B[:, :2]
                fit_to_pids = dfB.id.to_numpy()
            else:
                ground_truth_xy = B[:, :2]
                ground_truth_pids = dfB.id.to_numpy()
                locations = A[:, :2]
                fit_to_pids = dfA.id.to_numpy()

            # calcualte distance using NearestNeighbors
            nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth_xy)
            distances, indices = nneigh.kneighbors(locations)

            idx_locations = np.arange(len(locations))
            idx_locations = idx_locations[:, np.newaxis]

            if len(A) > len(B):
                uniq_pids = indices
                fit_uniq_pids = idx_locations
            else:
                uniq_pids = idx_locations
                fit_uniq_pids = indices

            sorted_uniq_pids = np.where(distances < distance_threshold, uniq_pids, np.nan)
            sorted_uniq_pids = flatten_list_of_lists(sorted_uniq_pids)

            sorted_fit_uniq_pids = np.where(distances < distance_threshold, fit_uniq_pids, np.nan)
            sorted_fit_uniq_pids = flatten_list_of_lists(sorted_fit_uniq_pids)

            # remove NaNs from array: x = x[~numpy.isnan(x)]
            sorted_uniq_pids = np.array(sorted_uniq_pids)
            sorted_fit_uniq_pids = np.array(sorted_fit_uniq_pids)

            sorted_uniq_pids = sorted_uniq_pids[~np.isnan(sorted_uniq_pids)]
            sorted_fit_uniq_pids = sorted_fit_uniq_pids[~np.isnan(sorted_fit_uniq_pids)]

            if len(A) > len(B):  # meaning: A is "ground truth"
                dfAA = dfA.iloc[sorted_uniq_pids]
                dfBB = dfB.iloc[sorted_fit_uniq_pids]
            else:  # meaning: B is "ground truth"
                dfAA = dfA.iloc[sorted_fit_uniq_pids]
                dfBB = dfB.iloc[sorted_uniq_pids]

            A = dfAA[['x', 'y', 'z']].to_numpy()
            B = dfBB[['x', 'y', 'z']].to_numpy()
            N = len(A)

            # iterative closest point algorithm (ref: https://github.com/ClayFlannigan/icp)
            # T, distances, iterations = icp.icp(A, B, tolerance=0.000001)  # originally: icp.icp(B, A,...)
            T, distances, iterations, dst_indices = pci_icp(A, B, tolerance=0.000001)  # also returns indices

            # distance by direction
            xdists, ydists, zdists = distances_xyz(AR=A, BR=B, TR=T)
            rmse_xdist = np.sqrt(np.mean(np.square(xdists)))
            rmse_ydist = np.sqrt(np.mean(np.square(ydists)))
            rmse_zdist = np.sqrt(np.mean(np.square(zdists)))

            # get matching indices from dfBB
            dfBB_icp = dfBB.iloc[dst_indices]
            dfBB_icp['errx'] = xdists
            dfBB_icp['erry'] = ydists
            dfBB_icp['errxy'] = np.sqrt(dfBB_icp['errx'] ** 2 + dfBB_icp['erry'] ** 2)
            dfBB_icp['errz'] = zdists

            # if you want dfBB and dfAA (below), or, if you only want dfBB (below, below)
            if include_AA:
                # dfAA should already be the correct indices
                dfAA_icp = dfAA[['id', 'x', 'y', 'z']]
                dfAA_icp = dfAA_icp.rename(
                    columns={'id': 'a_id', 'x': 'a_' + 'x', 'y': 'a_' + 'y', 'z': 'a_z'})

                # reset the natural index of both dfAA and dfBB
                dfAA_icp = dfAA_icp.reset_index(drop=True)
                dfBB_icpAB = dfBB_icp.reset_index(drop=True)

                # concat
                dfBB_icpAB = pd.concat([dfBB_icpAB, dfAA_icp], axis=1)
                dfBB_icpAB['ab_errx'] = dfBB_icpAB['a_' + 'x'] - dfBB_icpAB['x']
                dfBB_icpAB['ab_erry'] = dfBB_icpAB['a_' + 'y'] - dfBB_icpAB['y']
                dfBB_icpAB['ab_errxy'] = np.sqrt(dfBB_icpAB['ab_errx'] ** 2 + dfBB_icpAB['ab_erry'] ** 2)
                dfBB_icpAB['ab_errz'] = dfBB_icpAB['a_z'] - dfBB_icpAB['z']

                # remove rows with x-y errors that exceed limit
                dfBB_icpAB = dfBB_icpAB[(dfBB_icpAB['errx'].abs() < distance_threshold) &
                                        (dfBB_icpAB['erry'].abs() < distance_threshold)]

                dfBB_icps.append(dfBB_icpAB)

            else:
                # remove rows with x-y errors that exceed limit
                dfBB_icp = dfBB_icp[(dfBB_icp['errx'].abs() < distance_threshold) &
                                    (dfBB_icp['erry'].abs() < distance_threshold)]

                dfBB_icps.append(dfBB_icp)

            # Make C a homogeneous representation of B
            C = np.ones((N, 4))
            C[:, 0:3] = np.copy(B)
            # Transform C
            C = np.dot(T, C.T).T

            # evaluate transformation results
            deltax, deltay, deltaz = T[0, 3], T[1, 3], T[2, 3]
            precision_dist = np.std(distances)
            rmse_dist = np.sqrt(np.mean(np.square(distances)))
            data.append([fr, 0, zts[ii], zts[ii],
                         precision_dist, rmse_dist,
                         deltax, deltay, deltaz,
                         rmse_xdist, rmse_ydist, rmse_zdist,
                         len(distances), i_num_A, i_num_B])

    dfBB_icps = pd.concat(dfBB_icps)
    df_icps = pd.DataFrame(np.array(data), columns=['frame', 'zA', 'zB', 'z',
                                                    'precision', 'rmse',
                                                    'dx', 'dy', 'dz',
                                                    'rmse_x', 'rmse_y', 'rmse_z',
                                                    'num_icp', 'numA', 'numB'])
    return dfBB_icps, df_icps


def depth_averaged_rmse_rigid_transforms_from_focus(df):
    df = df[['errx', 'erry', 'errxy', 'errz']]
    df['bin'] = 1
    df['rmse_errx'] = df['errx'] ** 2
    df['rmse_erry'] = df['erry'] ** 2
    df['rmse_errxy'] = df['errxy'] ** 2
    df['rmse_errz'] = df['errz'] ** 2
    df['rmse_errxyz'] = df['errx'] ** 2 + df['erry'] ** 2 + df['errz'] ** 2
    dfm = df.groupby('bin').mean().reset_index()
    dfm['rmse_errx'] = np.sqrt(dfm['rmse_errx'])
    dfm['rmse_erry'] = np.sqrt(dfm['rmse_erry'])
    dfm['rmse_errxy'] = np.sqrt(dfm['rmse_errxy'])
    dfm['rmse_errz'] = np.sqrt(dfm['rmse_errz'])
    dfm['rmse_errxyz'] = np.sqrt(dfm['rmse_errxyz'])
    dfm = dfm[['bin', 'rmse_errx', 'rmse_erry', 'rmse_errxy', 'rmse_errz', 'rmse_errxyz']]
    return dfm


def read_coords_idpt(path_coords):
    # read test coords
    df = pd.read_excel(path_coords)

    # rename columns
    df = df.rename(columns={'cm_discrete': 'cm', 'z_sub': 'z', 'x_sub': 'x', 'y_sub': 'y'})
    df['z_nominal'] = df['frame']
    df = df[['frame', 'id', 'cm', 'x', 'y', 'z', 'z_nominal']]

    return df


def regularize_coordinates_between_image_sets(df, r0, length_per_pixel):
    """
    Align z-coordinates, make r-coordinate, scale in-plane coordinates to microns.

    :param df:
    :param r0:
    :param length_per_pixel:
    :return:
    """
    # dataset alignment
    z_zero_from_calibration = 49.9  # image of best-focus: z = 50.0 (arb. units)
    z_zero_of_calib_id_from_calibration = 49.6  # the in-focus z-position of calib particle in calib images (arb. units)
    z_zero_from_test_img_center = 68.6  # 68.51
    z_zero_of_calib_id_from_test = 68.1  # the in-focus z-position of calib particle in test images (arb. units)

    # TODO: limit images in dataset to only those within this z-range
    z_range = [-50, 55]

    # maintain original names where possible
    df['z_true'] = df['z_nominal']

    # 3.1: resolve z-position as a function of 'frame' discrepancy
    df['z_true_corr'] = (df['z_true'] - df['z_true'] % 3) / 3 * 5 + 5

    # 3.2: shift 'z_true' according to z_f (test images)
    df['z_true_minus_zf_from_test'] = df['z_true_corr'] - z_zero_of_calib_id_from_test

    # 3.3: shift 'z' according to z_f (calibration images)
    df['z_minus_zf_from_calib'] = df['z'] - z_zero_of_calib_id_from_calibration

    # STEP #5: update 'z' and 'z_true' coordinates & add 'error' column
    df['z'] = df['z_minus_zf_from_calib']
    df['z_true'] = df['z_true_minus_zf_from_test']

    # return with same columns as input
    df['z_nominal'] = df['z_true']
    df = df[['frame', 'id', 'cm', 'x', 'y', 'z', 'z_nominal']]

    # filter z-range
    df = df[(df['z_nominal'] >= z_range[0]) & (df['z_nominal'] <= z_range[1])]

    # make radial coordinate
    df['r'] = np.sqrt((df['x'] - r0[0]) ** 2 + (df['y'] - r0[1]) ** 2)

    # convert in-plane units to microns
    for pix2microns in ['x', 'y', 'r']:
        df[pix2microns] = df[pix2microns] * length_per_pixel

    return df


def read_coords_true_in_plane_positions(path_coords, length_per_pixel):
    dfxyzf = pd.read_excel(path_coords)
    dfxyzf['z'] = 0
    dfxyzf['gauss_rc'] = np.sqrt(dfxyzf['gauss_xc'] ** 2 + dfxyzf['gauss_yc'] ** 2)

    for pix2microns in ['gauss_xc', 'gauss_yc', 'gauss_rc']:
        dfxyzf[pix2microns] = dfxyzf[pix2microns] * length_per_pixel

    dfxyzf['x'] = dfxyzf['gauss_xc']
    dfxyzf['y'] = dfxyzf['gauss_yc']
    dfxyzf['r'] = dfxyzf['gauss_rc']

    return dfxyzf


if __name__ == '__main__':
    # A. experimental details
    mag_eff = 10.01  # effective magnification (experimentally measured)
    NA_eff = 0.45  # numerical aperture of objective lens
    microns_per_pixel = 1.6  # conversion ratio from pixels to microns (experimentally measured)
    size_pixels = 16  # units: microns (size of the pixels on the CCD sensor)
    num_pixels = 512
    area_pixels = num_pixels ** 2

    # B. IDPT processing details
    padding = 5  # units: pixels
    img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding

    # C. directories and file paths
    base_dir = '/Users/mackenzie/PythonProjects/idpt/publication'
    # datasets
    path_test_coords = join(base_dir, 'results/test/test_test-coords.xlsx')
    path_xy_at_zf = join(base_dir, 'analyses/ref/true_positions_fiji.xlsx')
    # results
    path_results = join(base_dir, 'results/rigid_transforms')
    path_pubfigs = join(base_dir, 'results/pubfigs')
    path_supfigs = join(base_dir, 'results/supfigs')

    # ---

    # 0. setup
    # filters
    # TODO: rename the following
    z_error_limit = 5  # units: microns
    filter_step_size = z_error_limit
    in_plane_distance_threshold = np.round(2 * microns_per_pixel, 1)  # units: microns
    min_num_particles_for_icp = 5  # threshold number of particles per frame for ICP
    min_cm = 0.0

    # -

    # 1. read test coords
    dft = read_coords_idpt(path_test_coords)

    # 2. pre-process coordinates
    dft = regularize_coordinates_between_image_sets(dft, r0=(img_xc, img_yc), length_per_pixel=microns_per_pixel)

    # 3. read coords: "true" in-plane positions of particles at focus (measured using ImageJ)
    dfxyzf = read_coords_true_in_plane_positions(path_xy_at_zf, length_per_pixel=microns_per_pixel)

    # 4. filter "invalid" measurements
    dft = dft[dft['error'].abs() < filter_step_size]
    dft = dft[dft['cm'] > min_cm]

    # ---

    # 5. rigid transformations from focus using ICP
    dfBB_icp, df_icp = rigid_transforms_from_focus(dft, dfxyzf, min_num_particles_for_icp, in_plane_distance_threshold)
    dfBB_icp.to_excel(join(path_results, 'dfBB_icp.xlsx'))
    df_icp.to_excel(join(path_results, 'df_icp.xlsx'))

    # 6. depth-dependent r.m.s. error
    dfdz_icp = df_icp.groupby('z').mean().reset_index()
    dfdz_icp.to_excel(join(path_results, 'dfdz_icp.xlsx'))

    # 6. depth-averaged r.m.s. error
    dfBB_icp_mean = depth_averaged_rmse_rigid_transforms_from_focus(dfBB_icp)
    dfBB_icp_mean.to_excel(join(path_results, 'icp_mean-rmse.xlsx'))

    # ----------------------------------------------------------------------------------------------------------------------
    # 7. Publication figures

    plot_pubfigs = False
    if plot_pubfigs:

        dfirt = dfdz_icp

        # ---

        # read: results from error relative to calibration particle
        path_read_err_rel_p_cal = join('/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/'
                                       '11.06.21_error_relative_calib_particle',
                                       'results',
                                       'relative-to-tilt-corr-calib-particle_08.06.23_raw-original',
                                       'spct-is-corr-fc',
                                       'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                                       'ztrue_is_fit-plane-xyzc',
                                       'bin-z_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))

        dfim = pd.read_excel(join(path_read_err_rel_p_cal, 'idpt_cm0.5_bin-z_rmse-z.xlsx'))
        dfsm = pd.read_excel(join(path_read_err_rel_p_cal, 'spct_cm0.5_bin-z_rmse-z.xlsx'))
        dfssm = pd.read_excel(join(path_read_err_rel_p_cal, 'spct_cm0.9_bin-z_rmse-z.xlsx'))

        # filter before plotting
        dfim = dfim[dfim['count_id'] > min_counts_bin_z]
        dfsm = dfsm[dfsm['count_id'] > min_counts_bin_z]
        dfssm = dfssm[dfssm['count_id'] > min_counts_bin_z]

        # ---

        # plot local correlation coefficient

        # setup - general
        clr_i = sciblue
        clr_s = scigreen
        clr_ss = sciorange
        if include_cmin_zero_nine:
            lgnd_i = 'IDPT' + r'$(C_{m,min}=0.5)$'
            lgnd_s = 'SPCT' + r'$(C_{m,min}=0.5)$'
            lgnd_ss = 'SPCT' + r'$(C_{m,min}=0.9)$'
        else:
            lgnd_i = 'IDPT'
            lgnd_s = 'SPCT'
            lgnd_ss = 'SPCT'
        zorder_i, zorder_s, zorder_ss = 3.5, 3.3, 3.4

        ms = 4
        xlbl = r'$z \: (\mu m)$'
        xticks = [-50, -25, 0, 25, 50]

        # -

        # setup plot

        # variables: error relative calib particle
        px = 'bin'
        py = 'cm'
        pyb = 'percent_meas'
        py4 = 'rmse_z'

        # variables: rigid transformations
        px1 = 'z'
        py1 = 'rmse_x'
        py2 = 'rmse_y'

        ylbl_cm = r'$C_{m}^{\delta}$'
        ylim_cm = [0.71, 1.02]  # data range: [0.7, 1.0]
        yticks_cm = [0.8, 0.9, 1.0]  # data ticks: np.arange(0.75, 1.01, 0.05)

        ylbl_phi = r'$\phi^{\delta}$'
        ylim_phi = [0, 1.1]
        yticks_phi = [0, 0.5, 1]

        ylbl_rmse_xy = r'$\sigma_{xy}^{\delta} \: (\mu m)$'
        ylim_rmse_xy = [0, 1]
        yticks_rmse_xy = [0, 0.5, 1]

        ylbl_rmse_z = r'$\sigma_{z}^{\delta} \: (\mu m)$'
        ylim_rmse_z = [0, 2.6]
        yticks_rmse_z = [0, 1, 2]

        # plot
        if include_cmin_zero_nine:
            fig, axs = plt.subplots(2, 2, sharex=True, figsize=(size_x_inches * 2, size_y_inches * 1.25))
        else:
            fig, axs = plt.subplots(2, 2, sharex=True, figsize=(size_x_inches * 2, size_y_inches * 1.05))

        ax2, ax3, ax1, ax4 = axs.ravel()

        ax1.plot(dfim[px], dfim[py], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax1.plot(dfsm[px], dfsm[py], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)

        if include_cmin_zero_nine:
            ax1.plot(dfssm[px], dfssm[py], '-o', ms=ms, color=clr_ss, label=lgnd_ss, zorder=zorder_ss)

        ax1.set_xlabel(xlbl)
        ax1.set_xticks(xticks)
        ax1.set_ylabel(ylbl_cm)
        ax1.set_ylim(ylim_cm)
        ax1.set_yticks(yticks_cm)
        # ax1.legend(loc='lower center')  # loc='upper left', bbox_to_anchor=(1, 1))

        # -

        ax2.plot(dfim[px], dfim[pyb], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax2.plot(dfsm[px], dfsm[pyb], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)

        if include_cmin_zero_nine:
            ax2.plot(dfssm[px], dfssm[pyb], '-o', ms=ms, color=clr_ss, label=lgnd_ss, zorder=zorder_ss)

        # ax2.set_xlabel(xlbl)
        # ax2.set_xticks(xticks)
        ax2.set_ylabel(ylbl_phi)
        ax2.set_ylim(ylim_phi)
        ax2.set_yticks(yticks_phi)

        if include_cmin_zero_nine:
            ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.05))
        else:
            ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.0),
                       ncol=2)  # loc='upper left', bbox_to_anchor=(1, 1)) , ncol=2

        # -

        ax3.plot(dfirt[px1], np.sqrt(dfirt[py1] ** 2 + dfirt[py2] ** 2), '-o', ms=ms, color=clr_i, label=lgnd_i,
                 zorder=zorder_i)
        ax3.plot(dfsrt[px1], np.sqrt(dfsrt[py1] ** 2 + dfsrt[py2] ** 2), '-o', ms=ms, color=clr_s, label=lgnd_s,
                 zorder=zorder_s)

        if include_cmin_zero_nine:
            ax3.plot(dfssrt[px1], np.sqrt(dfssrt[py1] ** 2 + dfssrt[py2] ** 2), '-o', ms=ms, color=clr_ss,
                     label=lgnd_ss, zorder=zorder_ss)

        # ax3.set_xlabel(xlbl)
        # ax3.set_xticks(xticks)
        ax3.set_ylabel(ylbl_rmse_xy)
        # ax3.set_ylim(ylim_rmse_xy)
        # ax3.set_yticks(yticks_rmse_xy)
        # ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # -

        ax4.plot(dfim[px], dfim[py4], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax4.plot(dfsm[px], dfsm[py4], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)

        if include_cmin_zero_nine:
            ax4.plot(dfssm[px], dfssm[py4], '-o', ms=ms, color=clr_ss, label=lgnd_ss, zorder=zorder_ss)

        ax4.set_xlabel(xlbl)
        ax4.set_xticks(xticks)
        ax4.set_ylabel(ylbl_rmse_z)
        # ax4.set_ylim(ylim_rmse_z)
        # ax4.set_yticks(yticks_rmse_z)
        # ax4.legend(loc='upper left')  # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # hspace=0.175, wspace=0.25

        if include_cmin_zero_nine:
            plt.savefig(join(path_pubfigs, 'compare_local_Cm-phi-rmsexyz_by_z_all_auto-ticks.png'))
        else:
            plt.savefig(join(path_pubfigs, 'compare_local_Cm-phi-rmsexyz_by_z_alt-legend.png'))
        plt.show()
        plt.close()

        # ---

    # ---

    # ----------------------------------------------------------------------------------------------------------------------
    # 8. Supplementary Information figures
    # TODO: add plot of histogram of erros

    # plot accuracy of rigid transformations
    plot_rt_accuracy = True  # True False
    if plot_rt_accuracy:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                            figsize=(size_x_inches, size_y_inches * 1.5))

        ax1.plot(df_icp.z, df_icp.rmse, '-o', label='rmse(R.T.)')
        ax1.plot(df_icp.z, df_icp.precision, '-o', label='precision(R.T.)')
        ax1.set_ylabel('Transform')
        ax1.legend()

        ax2.plot(df_icp.z, df_icp.dx, '-o', label='dx')
        ax2.plot(df_icp.z, df_icp.dy, '-o', label='dy')
        ax2.plot(df_icp.z, df_icp.dz, '-o', label='|dz|-5')
        ax2.set_ylabel('displacement (um)')
        ax2.legend()

        ax3.plot(df_icp.z, df_icp.rmse_x, '-o', label='x')
        ax3.plot(df_icp.z, df_icp.rmse_y, '-o', label='y')
        ax3.plot(df_icp.z, df_icp.rmse_z, '-o', label='z')
        ax3.set_ylabel('r.m.s. error (um)')
        ax3.legend()
        plt.tight_layout()
        plt.savefig(join(path_supfigs, 'RT_accuracy.png'))
        plt.show()
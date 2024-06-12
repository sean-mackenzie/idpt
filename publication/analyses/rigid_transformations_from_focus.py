# publication/rigid_transformations_from_focus.py

import os
from os.path import join
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from publication.analyses.utils import iterative_closest_point as icp
from publication.analyses.utils import fit, bin

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
    zts = df_test['z_nominal'].sort_values(ascending=True).unique()
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


def regularize_coordinates_between_image_sets(df, r0):
    """
    Align z-coordinates, make r-coordinate, scale in-plane coordinates to microns.

    :param df:
    :param r0:
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

    return df


def read_coords_true_in_plane_positions(path_coords, length_per_pixel):
    dfxyzf = pd.read_excel(path_coords)
    dfxyzf['z'] = 0
    dfxyzf['gauss_rc'] = np.sqrt(dfxyzf['gauss_xc'] ** 2 + dfxyzf['gauss_yc'] ** 2)

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
    num_frames_per_step = 3
    true_num_particles_per_frame = 88
    true_num_particles_per_z = true_num_particles_per_frame * num_frames_per_step

    # B. IDPT processing details
    padding = 5  # units: pixels
    img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding

    # C. directories and file paths
    base_dir = '/Users/mackenzie/PythonProjects/idpt/publication'
    # datasets
    path_test_coords = join(base_dir, 'results/test/test_test-coords.xlsx')
    path_xy_at_zf = join(base_dir, 'analyses/ref/true_positions_fiji.xlsx')
    # results
    path_pubfigs = join(base_dir, 'results/pubfigs')
    path_supfigs = join(base_dir, 'results/supfigs')

    # ---

    # 0. setup
    # filters
    # TODO: rename the following
    out_of_plane_threshold = 5  # units: microns
    in_plane_threshold = np.round(2 * microns_per_pixel, 1)  # units: microns
    min_num_particles_for_icp = 5  # threshold number of particles per frame for ICP
    min_cm = 0.0
    min_counts = 1
    min_counts_bin_z = 20
    min_counts_bin_r = 20
    min_counts_bin_rz = 5

    # -

    # ------------------------------------------------------------------------------------------------------------------
    # EVALUATE Z_TRUE RELATIVE TO IDPT FITTED PLANE

    fit_plane_analysis = True
    if fit_plane_analysis:

        # 0. setup
        path_results = join(base_dir, 'results/fit_plane')

        # specific settings (that should get deleted after confirming they aren't needed)
        correct_tilt = True  # False True
        correct_spct_tilt_using_idpt_fit_plane = True
        assign_z_true_to_fit_plane_xyzc = True
        # idpt
        i_test_name = 'test_coords_particle_image_stats_tm16_cm19_aligned'  # _dzf-post-processed'
        i_calib_id_from_testset = 54  # 42
        i_calib_id_from_calibset = 54  # 42
        # step 0. filter dft such that it only includes particles that could reasonably be on the tilt surface
        reasonable_z_tilt_limit = 3.25
        reasonable_r_tilt_limit = int(np.round(250 / microns_per_pixel))  # convert units microns to pixels

        # -

        # 1. read test coords
        dft = read_coords_idpt(path_test_coords)

        # 2. pre-process coordinates
        dft = regularize_coordinates_between_image_sets(dft, r0=(img_xc, img_yc))

        # ---

        # 4. FUNCTION: fit plane at each z-position

        # get z-positions
        z_nominals = dft['z_nominal'].unique()

        # initialize lists
        dfis = []
        i_fit_plane_img_xyzc = []
        i_fit_plane_rmsez = []

        # iterate through z-positions
        for z_nominal in z_nominals:
            # clear z_calib
            z_calib = None
            i_z_calib = None

            # get all measurements at this nominal z-position
            dfit = dft[dft['z_nominal'] == z_nominal]

            # --- FUNCTION: correct tilt
            # step 0. filter dft such that it only includes particles that could reasonably be on the tilt surface
            dfit_within_tilt = dfit[np.abs(dfit['z'] - z_nominal) < reasonable_z_tilt_limit]

            # step 1. fit plane to particle positions
            i_dict_fit_plane = fit.fit_in_focus_plane(df=dfit_within_tilt,  # note: x,y units are pixels at this point
                                                      param_zf='z',
                                                      microns_per_pixel=microns_per_pixel,
                                                      img_xc=img_xc,
                                                      img_yc=img_yc)

            i_fit_plane_img_xyzc.append(i_dict_fit_plane['z_f_fit_plane_image_center'])
            i_fit_plane_rmsez.append(i_dict_fit_plane['rmse'])

            # step 2. correct coordinates using fitted plane
            dfit['z_plane'] = fit.calculate_z_of_3d_plane(dfit.x, dfit.y, popt=i_dict_fit_plane['popt_pixels'])
            dfit['z_plane'] = dfit['z_plane'] - i_dict_fit_plane['z_f_fit_plane_image_center']
            dfit['z_corr'] = dfit['z'] - dfit['z_plane']

            # add column for tilt
            dfit['tilt_x_degrees'] = i_dict_fit_plane['tilt_x_degrees']
            dfit['tilt_y_degrees'] = i_dict_fit_plane['tilt_y_degrees']

            # rename
            dfit = dfit.rename(columns={'z': 'z_no_corr'})
            dfit = dfit.rename(columns={'z_corr': 'z'})

            # get average position of calibration particle
            if assign_z_true_to_fit_plane_xyzc:
                i_z_calib = i_dict_fit_plane['z_f_fit_plane_image_center']

            # TODO: rename columns to be more intuitive
            dfit['z_calib'] = i_z_calib
            dfit['error_rel_p_calib'] = dfit['z'] - i_z_calib

            dfis.append(dfit)
        dfis = pd.concat(dfis)

        # export: (1) all measurements (valid & invalid), (2) invalid only, (3) valid only
        # make absolute error
        dfis['abs_error_rel_p_calib'] = dfis['error_rel_p_calib'].abs()

        # Export 1/3: all measurements (valid + invalid)
        dfis.to_excel(join(path_results, 'idpt_error_relative_calib_particle_stack_all.xlsx'))

        # Export 2/3: invalid measurements only
        dfis_invalid = dfis[dfis['abs_error_rel_p_calib'] > out_of_plane_threshold]
        dfis_invalid.to_excel(join(path_results, 'idpt_error_relative_calib_particle_stack_invalid-only.xlsx'))

        # Export 3/3: valid measurements only
        dfis = dfis[dfis['abs_error_rel_p_calib'] < out_of_plane_threshold]
        dfis.to_excel(join(path_results, 'idpt_error_relative_calib_particle_stack.xlsx'))

        # ---

        plot_fit_plane = False
        if plot_fit_plane:

            # analyze fit plane xyzc and rmse-z
            df_fit_plane = pd.DataFrame(data=np.vstack([z_nominals, i_fit_plane_img_xyzc, i_fit_plane_rmsez]).T,
                                        columns=['z_nominal', 'iz_xyc', 'irmsez'])

            df_fit_plane['iz_diff'] = df_fit_plane['iz_xyc'] - df_fit_plane['z_nominal']
            df_fit_plane.to_excel(join(path_results, 'both_fit-respective-plane-xyzc-rmsez_by_z-true.xlsx'))

            # plot fit_plane_image_xyzc (the z-position at the center of the image) and rmse-z as a function of z_true
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            ax1.plot(df_fit_plane['z_nominal'], df_fit_plane['iz_diff'], '-o', label='IDPT')
            ax2.plot(df_fit_plane['z_nominal'], df_fit_plane['irmsez'], '-o', label='IDPT')
            ax1.set_ylabel(r'$z_{nom} - z_{xyc} \: (\mu m)$')
            ax2.set_ylabel('r.m.s.e.(z) fit plane ' + r'$(\mu m)$')
            ax2.set_xlabel(r'$z_{nominal}$')
            plt.tight_layout()
            plt.savefig(join(path_results, 'both_fit-respective-plane-xyzc-rmsez_by_z-true.png'))
            plt.close()

            # ---

            dfig = dfis.copy()
            fig, ax = plt.subplots()
            ax.scatter(dfig['r'], dfig['error_rel_p_calib'], s=2, label='IDPT')
            ax.set_xlabel('r')
            ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
            ax.legend()
            plt.tight_layout()
            plt.savefig(join(path_results, 'scatter-error_rel_p_calib___.png'))
            plt.close()

            fig, ax = plt.subplots()
            ax.scatter(dfig['r'], dfig['abs_error_rel_p_calib'], s=2, label='IDPT')
            ax.set_xlabel('r')
            ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
            ax.legend()
            plt.tight_layout()
            plt.savefig(join(path_results, 'scatter-abs-error_rel_p_calib___.png'))
            plt.close()

            # ---

            # plot tilt per frame
            dfig = dfig.groupby('z_nominal').mean().reset_index()
            # TODO: this should be (512 + 2 * padding) * microns_per_pixel
            xspan = 512 * microns_per_pixel
            ms = 4

            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

            ax1.plot(dfig['z_nominal'], dfig.tilt_x_degrees, '-o', ms=ms, label='x', color='r')
            ax1.plot(dfig['z_nominal'], dfig.tilt_y_degrees, '-s', ms=ms, label='y', color='k')

            ax2.plot(dfig['z_nominal'], np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_x_degrees))), '-o', ms=ms, label='x', color='r')
            ax2.plot(dfig['z_nominal'], np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_y_degrees))), '-s', ms=ms, label='y', color='k')

            ax1.set_ylabel('Tilt ' + r'$(deg.)$')
            ax1.legend()
            ax2.set_ylabel(r'$\Delta z_{FoV} \: (\mu m)$')
            ax2.set_xlabel(r'$z_{nominal} \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(join(path_results, 'sample-tilt-by-fit-IDPT_by_z-true.png'))
            plt.close()

            # -

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.2))

            ax1.plot(dfig['z_nominal'], dfig.tilt_x_degrees, '-o', ms=ms, label='x', color='r')
            ax1.plot(dfig['z_nominal'], dfig.tilt_y_degrees, '-s', ms=ms, label='y', color='k')

            ax2.plot(dfig['z_nominal'], np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_x_degrees))), '-o', ms=ms, label='x', color='r')
            ax2.plot(dfig['z_nominal'], np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_y_degrees))), '-s', ms=ms, label='y', color='k')

            ax3.plot(df_fit_plane['z_nominal'], df_fit_plane['irmsez'], '-o', ms=ms, label='IDPT')
            # ax3.plot(df_fit_plane['z_nominal'], df_fit_plane['srmsez'], '-o', label='SPCT')

            ax1.set_ylabel('Tilt ' + r'$(deg.)$')
            ax1.legend()
            ax2.set_ylabel(r'$\Delta z_{FoV} \: (\mu m)$')
            ax3.set_ylabel(r'$\sigma_z^{fit} \: (\mu m)$')
            ax3.set_xlabel(r'$z_{nominal} \: (\mu m)$')
            ax3.legend()
            plt.tight_layout()
            plt.savefig(join(path_results, 'sample-tilt-and-rmsez-by-fit-IDPT-and_by_z-true.png'))
            plt.close()

            # -

            fig, ax = plt.subplots(figsize=(size_x_inches * 1.05, size_y_inches * 0.75))

            ax.plot(df_fit_plane['irmsez'], np.abs(dfig.tilt_x_degrees), 'o', ms=ms, label='x', color='r')
            ax.plot(df_fit_plane['irmsez'], np.abs(dfig.tilt_y_degrees), 's', ms=ms, label='y', color='k')

            ax.set_ylabel('Tilt ' + r'$(deg.)$')
            ax.set_xlabel(r'$\sigma_z^{fit} \: (\mu m)$')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(join(path_results, 'abs-sample-tilt_by_rmsez-fit-IDPT.png'))
            plt.close()

        analyze_rmse_relative_calib_post_tilt_corr = False  # False True
        if analyze_rmse_relative_calib_post_tilt_corr:
            plot_bin_z = True
            plot_bin_r = True
            plot_bin_r_z = True
            plot_bin_id = True
            plot_cmin_zero_nine = False

            correct_tilt_by_fit_idpt = True
            assign_z_true_to_fit_plane_xyzc = True

            dfi = dfis.copy()

            # ---

            # process data

            # number of z-positions
            num_z_positions = len(dfi['z_nominal'].unique())
            true_total_num = true_num_particles_per_z * num_z_positions

            # scale to microns
            dfi['r_microns'] = dfi['r'] * microns_per_pixel

            # square all errors
            dfi['rmse_z'] = dfi['error_rel_p_calib'] ** 2

            # ---

            # ---

            # -------------
            # bin by axial position
            path_save_z = path_results

            # bin by z

            if plot_bin_z:

                # setup 2D binning
                z_trues = dfi['z_nominal'].unique()

                column_to_bin = 'z_nominal'
                column_to_count = 'id'
                bins = z_trues
                round_to_decimal = 1
                return_groupby = True

                # compute 1D bin (z)
                dfim, dfistd = bin.bin_generic(dfi, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

                # compute rmse-z
                dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])

                # compute final stats and package prior to exporting
                def package_for_export(df_):
                    """ df = package_for_export(df_=df) """
                    df_['true_num_per_z'] = true_num_particles_per_z
                    df_['percent_meas'] = df_['count_id'] / df_['true_num_per_z']
                    df_ = df_.rename(columns=
                                     {
                                      'z_calib': 'z_assert_true',
                                      'error_rel_p_calib': 'error_rel_z_assert_true',
                                      'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'}
                                     )
                    df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'r', 'z_plane', 'r_microns'])
                    return df_


                dfim = package_for_export(df_=dfim)

                # export
                dfim.to_excel(join(path_save_z, 'idpt_cm0.5_bin-z_rmse-z.xlsx'))

                # ---

                # plotting

                # filter before plotting
                dfim = dfim[dfim['count_id'] > min_counts_bin_z]


                # plot: rmse_z by z_nominal (i.e., bin)
                fig, ax = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.))

                ax.plot(dfim.bin, dfim['rmse_z'], '-o', label='IDPT' + r'$(C_{m,min}=0.5)$')

                save_lbl = 'bin-z_rmse-z_by_z'
                ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
                ax.set_ylim([0, 3.25])
                ax.set_yticks([0, 1, 2, 3])
                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_xticks([-50, -25, 0, 25, 50])
                ax.legend(loc='upper left')  # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

                plt.tight_layout()
                plt.savefig(join(path_save_z, save_lbl + '.png'))
                plt.close()

                # ---

                # plot: local (1) correlation coefficient, (2) percent measure, and (3) rmse_z

                # setup
                zorder_i, zorder_s, zorder_ss = 3.5, 3.3, 3.4
                ms = 4

                # plot
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                                    figsize=(size_x_inches * 1.35, size_y_inches * 1.25))

                ax1.plot(dfim.bin, dfim['cm'], '-o', ms=ms, label='IDPT', zorder=zorder_i)
                ax2.plot(dfim.bin, dfim['percent_meas'], '-o', ms=ms, label='IDPT', zorder=zorder_i)
                ax3.plot(dfim.bin, dfim['rmse_z'], '-o', ms=ms, label='IDPT', zorder=zorder_i)

                save_lbl = 'bin-z_local-cm-percent-meas-rmse-z_by_z'

                ax1.set_ylabel(r'$C_{m}^{\delta}$')
                ax1.legend(loc='upper left',
                           bbox_to_anchor=(
                           1, 1))  # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

                ax2.set_ylabel(r'$\phi_{z}^{\delta}$')

                ax3.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
                # ax3.set_ylim([0, 2.8])
                # ax3.set_yticks([0, 1, 2])
                ax3.set_xlabel(r'$z \: (\mu m)$')
                ax3.set_xticks([-50, -25, 0, 25, 50])

                plt.tight_layout()
                plt.savefig(join(path_save_z, save_lbl + '.png'))
                plt.close()

                # ---

                # compute mean rmse-z (using 1 bin)
                bin_h = 1

                dfim, _ = bin.bin_generic(dfi, column_to_bin, column_to_count, bin_h, round_to_decimal, return_groupby)

                dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])


                # compute final stats and package prior to exporting
                def package_for_export(df_):
                    """ df = package_for_export(df_=df) """
                    df_['true_num'] = true_total_num
                    df_['percent_meas'] = df_['count_id'] / df_['true_num']
                    df_ = df_.rename(columns={'error_rel_p_calib': 'error_rel_z_assert_true',
                                              'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'})
                    df_ = df_.drop(
                        columns=['frame', 'id', 'z_no_corr', 'z_calib', 'x', 'y', 'r', 'z_plane', 'r_microns'])
                    return df_


                dfim = package_for_export(df_=dfim)

                dfim.to_excel(join(path_save_z, 'idpt_cm0.5_mean_rmse-z_by_z.xlsx'))

            # -

            # -------------------

            # -

            # -------------------
            # bin by radial position

            # bin by r

            if plot_bin_r:
                path_save_r = path_results

                # setup 2D binning
                r_bins = 4

                column_to_bin = 'r_microns'
                column_to_count = 'id'
                bins = r_bins
                round_to_decimal = 1
                return_groupby = True

                # compute 1D bin (z)
                dfim, dfistd = bin.bin_generic(dfi, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

                # compute rmse-z
                dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])


                # compute final stats and package prior to exporting
                def package_for_export(df_):
                    """ df = package_for_export(df_=df) """
                    df_ = df_.rename(columns=
                                     {'r': 'r_pixels',
                                      'error_rel_p_calib': 'error_rel_z_assert_true',
                                      'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'}
                                     )
                    df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'z_plane', 'z_calib',
                                            'tilt_x_degrees', 'tilt_y_degrees'])
                    return df_


                dfim = package_for_export(df_=dfim)

                # export
                dfim.to_excel(join(path_save_r, 'idpt_cm0.5_bin-r_rmse-z.xlsx'))

                # ---

                # plotting

                # filter before plotting
                dfim = dfim[dfim['count_id'] > min_counts_bin_r]

                # plot
                fig, ax = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.))

                ax.plot(dfim.bin, dfim['rmse_z'], '-o', label='IDPT' + r'$(C_{m,min}=0.5)$')

                save_lbl = 'bin-r_rmse-z_by_r'
                ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
                ax.set_ylim([0, 2.4])
                ax.set_xlim([50, 500])
                ax.set_xticks([100, 200, 300, 400, 500])
                ax.set_xlabel(r'$r \: (\mu m)$')
                # ax.set_xticks([-50, -25, 0, 25, 50])
                ax.legend(loc='upper left')  # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

                plt.tight_layout()
                plt.savefig(join(path_save_r, save_lbl + '.png'))
                plt.close()

                # ---

                # compute mean rmse-z (using 1 bin)
                bin_h = 1

                dfim, _ = bin.bin_generic(dfi, column_to_bin, column_to_count, bin_h, round_to_decimal, return_groupby)

                dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])

                # compute final stats and package prior to exporting
                def package_for_export(df_):
                    """ df = package_for_export(df_=df) """
                    df_ = df_.rename(columns=
                                     {'error_rel_p_calib': 'error_rel_z_assert_true',
                                      'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'}
                                     )
                    df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'z_plane', 'z_calib',
                                            'tilt_x_degrees', 'tilt_y_degrees', 'r', 'r_microns'])
                    return df_


                dfim = package_for_export(df_=dfim)

                dfim.to_excel(join(path_save_r, 'idpt_cm0.5_mean_rmse-z_by_r.xlsx'))

            # -------------------

            # -

            # -------------------
            # bin by radial and axial position

            # 2d-bin by r and z

            if plot_bin_r_z:
                path_save_rz = path_results

                # setup 2D binning
                z_trues = dfi['z_nominal'].unique()
                r_bins = [150, 300, 450]

                columns_to_bin = ['r_microns', 'z_nominal']
                column_to_count = 'id'
                bins = [r_bins, z_trues]
                round_to_decimals = [1, 1]
                return_groupby = True
                plot_fit = False

                # compute 2D bin (r, z)
                dfim, dfistd = bin.bin_generic_2d(dfi, columns_to_bin, column_to_count, bins, round_to_decimals,
                                                  min_counts_bin_rz, return_groupby)

                # compute rmse-z
                dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])

                # resolve floating point bin selecting
                dfim = dfim.round({'bin_tl': 0, 'bin_ll': 1})
                dfistd = dfistd.round({'bin_tl': 0, 'bin_ll': 1})
                dfim = dfim.sort_values(['bin_tl', 'bin_ll'])
                dfistd = dfistd.sort_values(['bin_tl', 'bin_ll'])

                # compute final stats and package prior to exporting
                def package_for_export(df_):
                    """ df = package_for_export(df_=df) """
                    df_ = df_.rename(columns=
                                     {
                                      'z_calib': 'z_assert_true',
                                      'r': 'r_pixels',
                                      'error_rel_p_calib': 'error_rel_z_assert_true',
                                      'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'}
                                     )
                    df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'z_plane',
                                            'tilt_x_degrees', 'tilt_y_degrees'])
                    return df_


                dfim = package_for_export(df_=dfim)

                # export
                dfim.to_excel(join(path_save_rz, 'idpt_cm0.5_bin_r-z_rmse-z.xlsx'))

                # ---

                # plot
                clrs = ['black', 'blue', 'red']
                if plot_cmin_zero_nine:
                    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                                        figsize=(size_x_inches * 1, size_y_inches * 1.25))
                else:
                    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1, size_y_inches * 1))

                for i, bin_r in enumerate(dfim.bin_tl.unique()):
                    dfibr = dfim[dfim['bin_tl'] == bin_r]
                    ax1.plot(dfibr.bin_ll, dfibr['rmse_z'], '-o', ms=4, color=clrs[i], label=int(np.round(bin_r, 0)))

                ax1.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
                ax1.set_ylim([0, 3.2])
                ax1.set_yticks([0, 1, 2, 3])
                ax1.legend(loc='upper center', ncol=3, title=r'$r^{\delta} \: (\mu m)$')  # ,  title=r'$r^{\delta}$')
                # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

                ax2.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
                ax2.set_ylim([0, 3.2])
                ax2.set_yticks([0, 1, 2, 3])

                ax2.set_xlabel(r'$z \: (\mu m)$')
                ax2.set_xticks([-50, -25, 0, 25, 50])

                save_lbl = 'bin_r-z_rmse-z_by_r-z'

                plt.tight_layout()
                plt.savefig(join(path_save_rz, save_lbl + '.png'))
                plt.close()

                # ---

                # compute mean rmse-z per radial bin
                bins = [r_bins, 1]
                dfim, dfistd = bin.bin_generic_2d(dfi, columns_to_bin, column_to_count, bins, round_to_decimals,
                                                  min_counts_bin_rz, return_groupby)

                dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])


                # compute final stats and package prior to exporting
                def package_for_export(df_):
                    """ df = package_for_export(df_=df) """
                    df_ = df_.rename(columns=
                                     {'r': 'r_pixels',
                                      'error_rel_p_calib': 'error_rel_z_assert_true',
                                      'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'}
                                     )
                    df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'z_plane', 'z_calib',
                                            'tilt_x_degrees', 'tilt_y_degrees'])
                    return df_


                dfim = package_for_export(df_=dfim)

                dfim.to_excel(join(path_save_rz, 'idpt_cm0.5_bin-rz_mean-rmse-z.xlsx'))

            # -------------------

            # -------------


    # ------------------------------------------------------------------------------------------------------------------
    # EVALUATE X_TRUE, Y_TRUE RELATIVE TO RIGID TRANSFORMATIONS FROM FOCUS

    rt_from_f = False
    if rt_from_f:
        dfis = pd.read_excel(join(base_dir, 'results/fit_plane', 'idpt_error_relative_calib_particle_stack.xlsx'))

        # 0. setup
        path_results = join(base_dir, 'results/rigid_transforms')

        # 1. read test coords
        # NOTE: these are the WRONG TEST COORDS! This should read the output of IDPT-fitted plane corrected coords!
        # dft = read_coords_idpt(path_test_coords)
        dft = dfis.copy()

        # 2. pre-process coordinates
        # NOTE: not necessary here since already done when fitting IDPT plane
        # dft = regularize_coordinates_between_image_sets(dft, r0=(img_xc, img_yc), length_per_pixel=microns_per_pixel)

        # 3. read coords: "true" in-plane positions of particles at focus (measured using ImageJ)
        dfxyzf = read_coords_true_in_plane_positions(path_xy_at_zf, length_per_pixel=microns_per_pixel)

        # 4. convert x, y, r coordinates from units pixels to microns
        # TODO: convert x,y,r to microns
        for pix2microns in ['x', 'y', 'r']:
            dft[pix2microns] = dft[pix2microns] * microns_per_pixel
            dfxyzf[pix2microns] = dfxyzf[pix2microns] * microns_per_pixel

        # ---

        # 5. rigid transformations from focus using ICP
        dfBB_icp, df_icp = rigid_transforms_from_focus(dft, dfxyzf, min_num_particles_for_icp, in_plane_threshold)
        dfBB_icp.to_excel(join(path_results, 'dfBB_icp.xlsx'))
        df_icp.to_excel(join(path_results, 'df_icp.xlsx'))

        # 6. depth-dependent r.m.s. error
        dfdz_icp = df_icp.groupby('z').mean().reset_index()
        dfdz_icp.to_excel(join(path_results, 'dfdz_icp.xlsx'))

        # 6. depth-averaged r.m.s. error
        dfBB_icp_mean = depth_averaged_rmse_rigid_transforms_from_focus(dfBB_icp)
        dfBB_icp_mean.to_excel(join(path_results, 'icp_mean-rmse.xlsx'))

        # --------------------------------------------------------------------------------------------------------------
        # 7. Publication figures

        plot_pubfigs = False
        if plot_pubfigs:

            dfirt = dfdz_icp

            # ---

            # read: results from error relative to calibration particle
            """path_read_err_rel_p_cal = join('/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/'
                                           '11.06.21_error_relative_calib_particle',
                                           'results',
                                           'relative-to-tilt-corr-calib-particle_08.06.23_raw-original',
                                           'spct-is-corr-fc',
                                           'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                                           'ztrue_is_fit-plane-xyzc',
                                           'bin-z_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit,
                                                                                              min_counts))"""

            path_read_err_rel_p_cal = join(join(base_dir, 'results/fit_plane'))
            dfim = pd.read_excel(join(path_read_err_rel_p_cal, 'idpt_cm0.5_bin-z_rmse-z.xlsx'))

            # ---

            # plot local correlation coefficient

            # setup - general
            clr_i = sciblue
            clr_s = scigreen
            clr_ss = sciorange
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
            fig, axs = plt.subplots(2, 2, sharex=True, figsize=(size_x_inches * 2, size_y_inches * 1.05))

            ax2, ax3, ax1, ax4 = axs.ravel()

            ax1.plot(dfim[px], dfim[py], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)

            ax1.set_xlabel(xlbl)
            ax1.set_xticks(xticks)
            ax1.set_ylabel(ylbl_cm)
            ax1.set_ylim(ylim_cm)
            ax1.set_yticks(yticks_cm)
            # ax1.legend(loc='lower center')  # loc='upper left', bbox_to_anchor=(1, 1))

            # -

            ax2.plot(dfim[px], dfim[pyb], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)

            # ax2.set_xlabel(xlbl)
            # ax2.set_xticks(xticks)
            ax2.set_ylabel(ylbl_phi)
            ax2.set_ylim(ylim_phi)
            ax2.set_yticks(yticks_phi)

            ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.0),
                       ncol=2)  # loc='upper left', bbox_to_anchor=(1, 1)) , ncol=2

            # -

            ax3.plot(dfirt[px1], np.sqrt(dfirt[py1] ** 2 + dfirt[py2] ** 2), '-o', ms=ms, color=clr_i, label=lgnd_i,
                     zorder=zorder_i)

            # ax3.set_xlabel(xlbl)
            # ax3.set_xticks(xticks)
            ax3.set_ylabel(ylbl_rmse_xy)
            # ax3.set_ylim(ylim_rmse_xy)
            # ax3.set_yticks(yticks_rmse_xy)
            # ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

            # -

            ax4.plot(dfim[px], dfim[py4], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)

            ax4.set_xlabel(xlbl)
            ax4.set_xticks(xticks)
            ax4.set_ylabel(ylbl_rmse_z)
            # ax4.set_ylim(ylim_rmse_z)
            # ax4.set_yticks(yticks_rmse_z)
            # ax4.legend(loc='upper left')  # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.3)  # hspace=0.175, wspace=0.25

            plt.savefig(join(path_pubfigs, 'compare_local_Cm-phi-rmsexyz_by_z_alt-legend.png'))
            plt.close()

            # ---

        # ---

        # plot accuracy of rigid transformations
        plot_rt_accuracy = False  # True False
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
            plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT SUPPLEMENTARY FIGURES
    plot_sup_figs = False
    if plot_sup_figs:

        # path_supfigs

        plot_histogram_errors_z = False
        if plot_histogram_errors_z:
            # plot histogram of errors
            dfi = pd.read_excel(join(base_dir, 'results/fit_plane', 'idpt_error_relative_calib_particle_stack.xlsx'))

            # histogram of z-errors
            error_col = 'error_rel_p_calib'
            binwidth_y = 0.1
            bandwidth_y = 0.075  # None
            xlim = 3
            ylim_top = 1000
            yticks = [0, 500, 1000]

            # iterate

            for estimate_kde in [False, True]:
                for df, mtd, mcm in zip([dfi], ['idpt'], [0.5]):
                    y = df[error_col].to_numpy()

                    # plot
                    fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5))

                    ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
                    ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
                    ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)
                    ny, binsy, patchesy = ax.hist(y, bins=ybins, orientation='vertical', color='gray', zorder=2.5)

                    ax.set_xlabel(r'$\epsilon_{z} \: (\mu m)$')
                    ax.set_xlim([-xlim, xlim])
                    ax.set_ylabel('Counts')
                    ax.set_ylim([0, ylim_top])
                    ax.set_yticks(yticks)

                    if estimate_kde:
                        kdex, kdey, bandwidth = fit.fit_kde(y, bandwidth=bandwidth_y)
                        # pdf, y_grid = kde_scipy(y, y_grid=None, bandwidth=bandwidth)

                        axr = ax.twinx()

                        axr.plot(kdex, kdey, linewidth=0.25, color='r', zorder=2.4)
                        # axr.plot(y_grid, pdf, linewidth=0.5, linestyle='--', color='b', zorder=2.4)

                        axr.set_ylabel('PDF')
                        axr.set_ylim(bottom=0)
                        # axr.set_yticks(yticks)
                        save_id = 'idpt_cmin{}_histogram_z-errors_kde-bandwidth={}.png'.format(mcm, np.round(bandwidth, 4))
                    else:
                        save_id = 'idpt_cmin{}_histogram_z-errors.png'.format(mcm)

                    plt.tight_layout()
                    plt.savefig(join(path_supfigs, save_id))
                    plt.close()

        # ---

        # plot x-y scatter of rmse_z per particle
        plot_histogram_errors_xy = False  # True False
        if plot_histogram_errors_xy:
            from sklearn.neighbors import KernelDensity
            # plot histogram for a single array
            # plot kernel density estimation
            def scatter_and_kde_y(y, binwidth_y=1, kde=True, bandwidth_y=0.5, xlbl='residual', ylim_top=525, yticks=[],
                                  save_path=None):

                fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5))

                # y
                ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
                ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
                ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)
                ny, binsy, patchesy = ax.hist(y, bins=ybins, orientation='vertical', color='gray', zorder=2.5)

                # kernel density estimation
                if kde:
                    ymin, ymax = np.min(y), np.max(y)
                    y_range = ymax - ymin
                    y_plot = np.linspace(ymin - y_range / 5, ymax + y_range / 5, 250)

                    y = y[:, np.newaxis]
                    y_plot = y_plot[:, np.newaxis]

                    kde_y = KernelDensity(kernel="gaussian", bandwidth=bandwidth_y).fit(y)
                    log_dens_y = kde_y.score_samples(y_plot)
                    scale_to_max = np.max(ny) / np.max(np.exp(log_dens_y))

                    # p2 = ax.fill_betweenx(y_plot[:, 0], 0, np.exp(log_dens_y) * scale_to_max, fc="None", ec=scired, zorder=2.5)
                    # p2.set_linewidth(0.5)
                    ax.plot(y_plot[:, 0], np.exp(log_dens_y) * scale_to_max, linewidth=0.75, linestyle='-',
                            color=scired)

                ax.set_xlabel(xlbl + r'$(\mu m)$')
                ax.set_xlim([-3, 3])
                ax.set_ylabel('Counts')
                ax.set_ylim([0, ylim_top])
                ax.set_yticks(yticks)
                ax.grid(alpha=0.25)

                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()


            dfi = pd.read_excel(join(base_dir, 'results/rigid_transforms', 'dfBB_icp.xlsx'))

            # plot formatting
            ylim_top = 1050
            yticks = [0, 500, 1000]

            # plot histogram
            err_cols = ['errx', 'erry', 'errz']
            err_lbls = ['x residual ', 'y residual ', 'z residual ']

            for ycol, xlbl in zip(err_cols, err_lbls):
                y = dfi[ycol].to_numpy()
                save_path = join(path_supfigs, 'hist_{}.png'.format(ycol))
                scatter_and_kde_y(y, binwidth_y=0.1, kde=False, bandwidth_y=0.25,
                                  xlbl=xlbl, ylim_top=ylim_top, yticks=yticks,
                                  save_path=save_path)

            # ---

        # ---
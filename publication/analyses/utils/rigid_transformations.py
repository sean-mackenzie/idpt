# publication/analyses/utils/rigid_transformations.py

"""
Portions of the following code are reproductions of the original codebase:
https://github.com/ClayFlannigan/icp.
Copyright 2016 Clay Flannigan
Licensed under the Apache License, Version 2.0

We acknowledge where significant changes have been made to the original
code. The original codebase license is copied below:

Copyright 2016 Clay Flannigan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def flatten_list_of_lists(l):
    return [item for sublist in l for item in sublist]


def pci_best_fit_transform(A, B):
    '''
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
    """
    xdists, ydists, zdists = distances_xyz(AR=, BR=)

    Portions of the following code have been adapted from the original codebase:
    https://github.com/ClayFlannigan/icp.
    Copyright 2016 Clay Flannigan
    Licensed under the Apache License, Version 2.0
    """

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


def rigid_transforms_from_focus(df_test, df_focus, min_num, distance_threshold, include_AA=True, return_outliers=False):
    # get z_true values and sort
    zts = df_test['z_nominal'].sort_values(ascending=True).unique()
    z_range_rt = len(zts)

    cols_icp = ['errx', 'erry', 'errxy', 'errz',
                'a_id', 'a_x', 'a_y', 'a_z',
                'ab_errx', 'ab_erry', 'ab_errxy', 'ab_errz']
    for col in cols_icp:
        if col in df_test.columns:
            df_test = df_test.drop(columns=[col])

    # initialize
    data = []
    dfBB_icps = []
    df_outliers = []

    for ii in range(z_range_rt):

        z_frames = df_test[df_test['z_nominal'] == zts[ii]].frame.unique()
        for fr in z_frames:

            dfA = df_focus.copy()
            dfB = df_test[(df_test['z_nominal'] == zts[ii]) & (df_test['frame'] == fr)].reset_index()
            dfA['z'] = 0

            # resolve GDPTlab tracking, which can assign the same ID to multiple particles
            if len(dfB) > len(dfB['id'].unique()):
                dfB['id'] = np.arange(1000, 1000 + len(dfB))

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

            if return_outliers:
                if len(A) > len(B):
                    flat_distances = np.array(flatten_list_of_lists(distances))
                    outlier_distances = flat_distances[flat_distances > distance_threshold]
                    outlier_pids = fit_to_pids[flat_distances > distance_threshold]
                    outlier_df = dfB[dfB['id'].isin(outlier_pids)]
                    outlier_df['distances'] = outlier_distances
                else:
                    flat_distances = np.array(flatten_list_of_lists(distances))
                    outlier_distances = flat_distances[flat_distances > distance_threshold]
                    outlier_pids = fit_to_pids[flat_distances > distance_threshold]
                    outlier_df = dfA[dfA['id'].isin(outlier_pids)]
                    outlier_df['distances'] = outlier_distances
                if len(outlier_df) > 0:
                    df_outliers.append(outlier_df)

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
            sorted_fit_uniq_pids = sorted_fit_uniq_pids[~np.isnan(sorted_fit_uniq_pids)].astype(int)

            if len(A) > len(B):  # meaning: A is "ground truth"
                dfAA = dfA.iloc[sorted_uniq_pids]
                dfBB = dfB.iloc[sorted_fit_uniq_pids]
            else:  # meaning: B is "ground truth"
                #for sfqp in sorted_fit_uniq_pids:
                #    dfAA__ = dfA.iloc[int(sfqp)]
                #    dfAA_ = dfA.iloc[sfqp]
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
            rmse_xydist = np.sqrt(np.mean(np.square(np.sqrt(xdists ** 2 + ydists ** 2))))
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
                         rmse_xdist, rmse_ydist, rmse_xydist, rmse_zdist,
                         len(distances), i_num_A, i_num_B])

    dfBB_icps = pd.concat(dfBB_icps)
    df_icps = pd.DataFrame(np.array(data), columns=['frame', 'zA', 'zB', 'z',
                                                    'precision', 'rmse',
                                                    'dx', 'dy', 'dz',
                                                    'rmse_x', 'rmse_y', 'rmse_xy', 'rmse_z',
                                                    'num_icp', 'numA', 'numB'])
    if return_outliers:
        if len(df_outliers) > 0:
            df_outliers = pd.concat(df_outliers)
        return dfBB_icps, df_icps, df_outliers
    else:
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
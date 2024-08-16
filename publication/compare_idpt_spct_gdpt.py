# publication/rigid_transformations_from_focus.py

import os
from os.path import join
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
from publication.analyses.utils import fit, bin
from publication.analyses.utils.rigid_transformations import rigid_transforms_from_focus, \
    depth_averaged_rmse_rigid_transforms_from_focus
import matplotlib as mpl
from matplotlib.pyplot import cm

mpl.use('TkAgg')
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


# --- HELPER FUNCTIONS


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def scatter_and_kde_y(y, binwidth_y, kde, bandwidth_y):
    fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5))
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
    return fig, ax


# --- DATA PROCESSING FUNCTIONS


def read_coords(path_coords, padding):
    df = pd.read_excel(path_coords)
    # adjust for padding (which is pre-configured for 5-pixel padding)
    relative_padding = padding - 5
    df.loc[:, 'x'] = df.loc[:, 'x'] + relative_padding
    df.loc[:, 'y'] = df.loc[:, 'y'] + relative_padding
    return df


def read_coords_idpt(path_coords, num_pixels, padding):
    # read test coords
    df = pd.read_excel(path_coords)
    # rename columns
    df = df.rename(columns={'cm_discrete': 'cm', 'z_sub': 'z', 'x_sub': 'x', 'y_sub': 'y'})
    df = df[['frame', 'id', 'cm', 'x', 'y', 'z']]
    df = df[(df['x'] > 0) & (df['x'] < num_pixels + padding)]
    df = df[(df['y'] > 0) & (df['y'] < num_pixels + padding)]
    return df


def read_coords_gdpt(path_coords, measurement_depth, padding):
    # read test coords
    df = pd.read_excel(path_coords)
    # rename columns
    df = df.rename(columns={'fr': 'frame', 'Z': 'z', 'X': 'x', 'Y': 'y'})
    df = df[['frame', 'id', 'cm', 'x', 'y', 'z']]
    # regularize frames with IDPT
    df.loc[:, 'frame'] = df.loc[:, 'frame'] - 1
    # scale z
    if df['z'].max() <= 1.0:
        df.loc[:, 'z'] = df.loc[:, 'z'] * measurement_depth
    # adjust for padding
    df.loc[:, 'x'] = df.loc[:, 'x'] + padding
    df.loc[:, 'y'] = df.loc[:, 'y'] + padding
    return df


def regularize_coordinates_between_image_sets(df, r0, z_range, zf_calibration, zf_test):
    # maintain original names where possible
    df['z_true'] = df['frame']

    # 3.1: resolve z-position as a function of 'frame' discrepancy
    df['z_true_corr'] = (df['z_true'] - df['z_true'] % 3) / 3 * 5 + 5

    # 3.2: shift 'z_true' according to z_f (test images)
    df['z_true_minus_zf_from_test'] = df['z_true_corr'] - zf_test

    # 3.3: shift 'z' according to z_f (calibration images)
    df['z_minus_zf_from_calib'] = df['z'] - zf_calibration

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


def align_datasets(dict_data, dict_inputs, dict_paths, dict_plots, make_xy_units):
    path_results = dict_paths['results']
    path_idpt_coords = dict_paths['idpt']
    path_spct_coords = dict_paths['spct']
    path_gdpt_coords = dict_paths['gdpt']
    path_true_coords = dict_paths['true']

    num_pixels = dict_inputs['num_pixels']
    microns_per_pixel = dict_inputs['microns_per_pixel']
    padding = dict_inputs['padding']
    measurement_depth = dict_inputs['measurement_depth']
    baseline_frame = dict_inputs['baseline_frame']
    zf_calibration = dict_inputs['zf_calibration']
    zf_test = dict_inputs['zf_test']
    z_range = dict_inputs['z_range']
    r0 = dict_inputs['r0']

    save_plot = dict_plots['dataset_alignment']

    # - PROCESSING

    # 1a. read idpt
    dfi = read_coords_idpt(path_idpt_coords, num_pixels=num_pixels, padding=padding)

    # 1b. read spct
    df_spct = read_coords(path_spct_coords, padding=padding)

    # 1c. read gdpt
    df_gdpt = read_coords_gdpt(path_gdpt_coords, measurement_depth=measurement_depth, padding=padding)

    # 2. regularize coords
    dfi = regularize_coordinates_between_image_sets(dfi, r0, z_range, zf_calibration, zf_test)
    df_spct = regularize_coordinates_between_image_sets(df_spct, r0, z_range, zf_calibration, zf_test)
    df_gdpt = regularize_coordinates_between_image_sets(df_gdpt, r0, z_range, zf_calibration, zf_test)

    # 3. remove particles near the borders
    xmin, xmax, ymin, ymax = dfi['x'].min(), dfi['x'].max(), dfi['y'].min(), dfi['y'].max()
    df_spct = df_spct[(df_spct['x'] > xmin - 5) & (df_spct['x'] < xmax + 5) &
                      (df_spct['y'] > ymin - 5) & (df_spct['y'] < ymax + 5)]
    df_gdpt = df_gdpt[(df_gdpt['x'] > xmin - 5) & (df_gdpt['x'] < xmax + 5) &
                      (df_gdpt['y'] > ymin - 5) & (df_gdpt['y'] < ymax + 5)]
    # 1d. read fiji true positions
    df_true = read_coords(path_true_coords, padding=padding)
    # make radial coordinate
    df_true['r'] = np.sqrt((df_true['x'] - r0[0]) ** 2 + (df_true['y'] - r0[1]) ** 2)

    # 3. convert x, y, r coordinates from units pixels to microns
    if make_xy_units == 'microns':
        for pix2microns in ['x', 'y', 'r']:
            dfi[pix2microns] = dfi[pix2microns] * microns_per_pixel
            df_spct[pix2microns] = df_spct[pix2microns] * microns_per_pixel
            df_gdpt[pix2microns] = df_gdpt[pix2microns] * microns_per_pixel
            df_true[pix2microns] = df_true[pix2microns] * microns_per_pixel

    # output
    dict_coords = dict({'aligned': {'IDPT': dfi, 'SPCT': df_spct, 'GDPT': df_gdpt, 'TRUE': df_true}})
    dict_data.update(dict_coords)

    # - EVALUATION

    # plot in-plane positions to verify
    if save_plot:
        path_dataset_alignment = join(path_results, 'dataset_alignment')
        make_dir(path=path_dataset_alignment)

        # setup
        dfs = [df_true, dfi, df_spct, df_gdpt]
        lbls = ['FIJI', 'IDPT', 'SPCT', 'GDPT']
        sizes = [12, 4, 2, 1]
        markers = ['o', 'D', 's', '*']
        clrs = ['k', sciblue, scigreen, sciorange]

        # plot x-y
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches * 1.5))
        for i, df in enumerate(dfs):
            if not lbls[i] == 'FIJI':
                df = df[df['frame'] == baseline_frame]
            ax.scatter(df['x'], df['y'], s=sizes[i], marker=markers[i], color=clrs[i], label=lbls[i])
        ax.set_xlabel(r'$x \: (\mu m)$')
        ax.set_ylabel(r'$y \: (\mu m)$')
        ax.legend(fontsize='small')
        plt.savefig(join(path_dataset_alignment, 'verify_in-plane_positions.png'))
        plt.close()

        # plot z
        fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1.5, size_y_inches * 2))
        for i, df in enumerate(dfs):
            if lbls[i] == 'FIJI':
                continue
            axes[i - 1].scatter(df['z_nominal'], df['z'], s=1, marker=markers[i], color=clrs[i], label=lbls[i])
            axes[i - 1].set_ylabel(r"$z' \: (\mu m)$")
            axes[i - 1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            axes[i - 1].grid(alpha=0.125)
        axes[-1].set_xlabel(r'$z_{nominal} \: (\mu m)$')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5, wspace=0.05)
        plt.savefig(join(path_dataset_alignment, 'compare_out-of-plane_positions.png'))
        plt.close()

    return dict_data


def correct_z_by_plane(df, dict_plane):
    # correct coordinates using fitted plane
    df['z_plane'] = fit.calculate_z_of_3d_plane(df.x, df.y, popt=dict_plane['popt_pixels'])
    df['z_plane'] = df['z_plane'] - dict_plane['z_f_fit_plane_image_center']
    df['z_corr'] = df['z'] - df['z_plane']

    # add column for tilt
    df['tilt_x_degrees'] = dict_plane['tilt_x_degrees']
    df['tilt_y_degrees'] = dict_plane['tilt_y_degrees']

    # rename
    df = df.rename(columns={'z': 'z_no_corr'})
    df = df.rename(columns={'z_corr': 'z'})

    if dict_plane['assert_z_f_fit_plane_image_center'] is not None:
        df['z_calib'] = dict_plane['assert_z_f_fit_plane_image_center']
        df['error_rel_plane'] = df['z'] - dict_plane['assert_z_f_fit_plane_image_center']
    else:
        df['z_calib'] = dict_plane['z_f_fit_plane_image_center']
        df['error_rel_plane'] = df['z'] - dict_plane['z_f_fit_plane_image_center']

    return df


def package_plane_corrected_dataframe(method, list_of_dataframes, out_of_plane_threshold, path_results, path_invalid):
    df_all = pd.concat(list_of_dataframes)

    # export: (1) all measurements (valid & invalid), (2) invalid only, (3) valid only
    # make absolute error
    df_all['abs_error_rel_plane'] = df_all['error_rel_plane'].abs()

    # Export 1/3: all measurements (valid + invalid)
    df_all.to_excel(join(path_results, '{}_error_relative_plane_all.xlsx'.format(method)), index=False)

    # Export 2/3: invalid measurements only
    df_invalid = df_all[df_all['abs_error_rel_plane'] > out_of_plane_threshold]
    df_invalid.to_excel(join(path_invalid, '{}_error_relative_plane_invalid-only.xlsx'.format(method)), index=False)

    # Export 3/3: valid measurements only
    df = df_all[df_all['abs_error_rel_plane'] < out_of_plane_threshold]
    df.to_excel(join(path_results, '{}_error_relative_plane.xlsx'.format(method)), index=False)

    return df, df_all


def bin_by_z_fp(df, z_bins, true_num_particles_per_z, true_total_num, path_results, method, return_global):
    column_to_bin = 'z_nominal'
    column_to_count = 'id'
    bins = z_bins
    round_to_decimal = 1
    return_groupby = True

    # compute 1D bin (z)
    dfm, dfstd = bin.bin_generic(df, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)
    # compute rmse-z
    dfm['rmse_z'] = np.sqrt(dfm['rmse_z'])

    # compute final stats and package prior to exporting
    def package_for_export(df_):
        """ df = package_for_export(df_=df) """
        df_['true_num_per_z'] = true_num_particles_per_z
        df_['percent_meas'] = df_['count_id'] / df_['true_num_per_z']
        df_ = df_.rename(columns=
        {
            'z_calib': 'z_assert_true',
            'error_rel_plane': 'error_rel_z_assert_true',
            'abs_error_rel_plane': 'abs_error_rel_z_assert_true'}
        )
        df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'r', 'z_plane', 'r_microns'])
        return df_

    dfm = package_for_export(df_=dfm)
    # export
    dfm.to_excel(join(path_results, '{}_bin-z_rmse-z.xlsx'.format(method)))

    # compute mean rmse-z (using 1 bin)
    bin_global = 1
    dfm_global, _ = bin.bin_generic(df, column_to_bin, column_to_count, bin_global, round_to_decimal, return_groupby)
    dfm_global['rmse_z'] = np.sqrt(dfm_global['rmse_z'])

    # compute final stats and package prior to exporting
    def package_for_export(df_):
        """ df = package_for_export(df_=df) """
        df_['true_num'] = true_total_num
        df_['percent_meas'] = df_['count_id'] / df_['true_num']
        df_ = df_.rename(columns={'error_rel_plane': 'error_rel_z_assert_true',
                                  'abs_error_rel_plane': 'abs_error_rel_z_assert_true'})
        df_ = df_.drop(
            columns=['frame', 'id', 'z_no_corr', 'z_calib', 'x', 'y', 'r', 'z_plane', 'r_microns'])
        return df_

    dfm_global = package_for_export(df_=dfm_global)
    # export
    dfm_global.to_excel(join(path_results, '{}_mean_rmse-z_by_z.xlsx'.format(method)))

    if return_global:
        return dfm, dfm_global
    else:
        return dfm


def bin_by_r_fp(df, r_bins, path_results, method):
    column_to_bin = 'r_microns'
    column_to_count = 'id'
    bins = r_bins
    round_to_decimal = 1
    return_groupby = True

    # compute 1D bin (z)
    dfm, dfstd = bin.bin_generic(df, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)
    # compute rmse-z
    dfm['rmse_z'] = np.sqrt(dfm['rmse_z'])

    # compute final stats and package prior to exporting
    def package_for_export(df_):
        """ df = package_for_export(df_=df) """
        df_ = df_.rename(columns=
                         {'r': 'r_pixels',
                          'error_rel_plane': 'error_rel_z_assert_true',
                          'abs_error_rel_plane': 'abs_error_rel_z_assert_true'}
                         )
        df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'z_plane', 'z_calib',
                                'tilt_x_degrees', 'tilt_y_degrees'])
        return df_

    dfm = package_for_export(df_=dfm)
    # export
    dfm.to_excel(join(path_results, '{}_bin-r_rmse-z.xlsx'.format(method)))
    return dfm


def bin_by_rz_fp(df, r_bins, z_bins, min_counts_bin_rz, path_results, method):
    columns_to_bin = ['r_microns', 'z_nominal']
    column_to_count = 'id'
    bins = [r_bins, z_bins]
    round_to_decimals = [1, 1]
    return_groupby = True

    # compute 2D bin (r, z)
    dfm, dfstd = bin.bin_generic_2d(df, columns_to_bin, column_to_count, bins, round_to_decimals,
                                    min_counts_bin_rz, return_groupby)
    # compute rmse-z
    dfm['rmse_z'] = np.sqrt(dfm['rmse_z'])
    # resolve floating point bin selecting
    dfm = dfm.round({'bin_tl': 0, 'bin_ll': 1})
    dfstd = dfstd.round({'bin_tl': 0, 'bin_ll': 1})
    dfm = dfm.sort_values(['bin_tl', 'bin_ll'])
    dfstd = dfstd.sort_values(['bin_tl', 'bin_ll'])

    # compute final stats and package prior to exporting
    def package_for_export(df_):
        """ df = package_for_export(df_=df) """
        df_ = df_.rename(columns=
        {
            'z_calib': 'z_assert_true',
            'r': 'r_pixels',
            'error_rel_plane': 'error_rel_z_assert_true',
            'abs_error_rel_plane': 'abs_error_rel_z_assert_true'}
        )
        df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'z_plane',
                                'tilt_x_degrees', 'tilt_y_degrees'])
        return df_

    dfm = package_for_export(df_=dfm)
    # export
    dfm.to_excel(join(path_results, '{}_bin_r-z_rmse-z.xlsx'.format(method)))
    return dfm


def filter_cm(dfs, labels, cmin, path_results):
    dfs_valid = []
    for df, lbl in zip(dfs, labels):
        # export invalid
        df_invalid = df[df['cm'] < cmin]
        if len(df_invalid) > 0:
            df_invalid.to_excel(join(path_results, '{}_cm-invalid-only.xlsx'.format(lbl)), index=False)
        # pass valid
        dfs_valid.append(df[df['cm'] > cmin])
    return dfs_valid


def fit_plane_analysis(dict_data, dict_inputs, dict_filters, dict_paths, dict_plots, xy_units):
    if xy_units == 'microns':
        microns_per_pixel = 1
    elif xy_units == 'pixels':
        microns_per_pixel = dict_inputs['microns_per_pixel']
    else:
        raise ValueError("xy_units must be: ['microns', 'pixels'].")

    img_xc = dict_inputs['img_xc']
    img_yc = dict_inputs['img_xc']

    out_of_plane_threshold = dict_filters['out_of_plane_threshold']
    z_tilt_limit = dict_filters['z_tilt_limit']

    path_results = dict_paths['fit_plane']
    make_dir(path=path_results)

    path_outliers = dict_paths['outliers']
    make_dir(path=path_outliers)

    # get data
    if dict_data['dataset_fit_plane'] == 'aligned':
        dfi = dict_data['aligned']['IDPT']
        dfs = dict_data['aligned']['SPCT']
        dfg = dict_data['aligned']['GDPT']
        # filter cm
        min_cm = dict_filters['min_cm']
        dfi, dfs, dfg = filter_cm(dfs=[dfi, dfs, dfg],
                                  labels=['idpt', 'spct', 'gdpt'],
                                  cmin=min_cm,
                                  path_results=path_outliers)
    elif dict_data['dataset_fit_plane'].startswith('rigid_transformations'):
        dfi = pd.read_excel(join(dict_paths['rigid_transformations'], 'dfBB_icp_idpt.xlsx'))
        dfs = pd.read_excel(join(dict_paths['rigid_transformations'], 'dfBB_icp_spct.xlsx'))
        dfg = pd.read_excel(join(dict_paths['rigid_transformations'], 'dfBB_icp_gdpt.xlsx'))
    else:
        raise ValueError("Dataset not understood. Options are: ['aligned', 'corrected', 'corrected_all']")

    # get z-positions
    z_nominals = dfi['z_nominal'].unique()

    # initialize lists for fitted planes
    i_fit_plane_dicts = {}
    i_fit_plane_img_xyzc_assert = []
    i_fit_plane_img_xyzc = []
    i_fit_plane_rmsez = []
    # initialize lists for corrected dataframes
    dfis, dfss, dfgs = [], [], []

    # iterate through z-positions
    for z_nominal in z_nominals:
        # get all measurements at this nominal z-position
        dfiz = dfi[dfi['z_nominal'] == z_nominal]
        dfsz = dfs[dfs['z_nominal'] == z_nominal]
        dfgz = dfg[dfg['z_nominal'] == z_nominal]

        # --- correct tilt
        # step 0. filter dft such that it only includes particles that could reasonably be on the tilt surface
        dfiz_within_tilt = dfiz[np.abs(dfiz['z'] - z_nominal) < z_tilt_limit]

        # step 1. fit plane to particle positions
        dict_fit_plane = fit.fit_in_focus_plane(df=dfiz_within_tilt,  # note: x,y units are pixels at this point
                                                param_zf='z',
                                                microns_per_pixel=microns_per_pixel,
                                                img_xc=img_xc,
                                                img_yc=img_yc,
                                                )
        if z_nominal > 1.8 and z_nominal < 2.0:
            dict_fit_plane['assert_z_f_fit_plane_image_center'] = 1.5
        else:
            dict_fit_plane['assert_z_f_fit_plane_image_center'] = dict_fit_plane['z_f_fit_plane_image_center']

        i_fit_plane_dicts.update({np.round(z_nominal, 1): dict_fit_plane})
        i_fit_plane_img_xyzc_assert.append(dict_fit_plane['assert_z_f_fit_plane_image_center'])
        i_fit_plane_img_xyzc.append(dict_fit_plane['z_f_fit_plane_image_center'])
        i_fit_plane_rmsez.append(dict_fit_plane['rmse'])

        # correct dataframes
        # 2a. IDPT
        dfiz = correct_z_by_plane(df=dfiz, dict_plane=dict_fit_plane)
        dfis.append(dfiz)

        # 2b. SPCT
        dfsz = correct_z_by_plane(df=dfsz, dict_plane=dict_fit_plane)
        dfss.append(dfsz)

        # 2c. GDPT
        dfgz = correct_z_by_plane(df=dfgz, dict_plane=dict_fit_plane)
        dfgs.append(dfgz)

    # stack and export
    dfis, dfis_all = package_plane_corrected_dataframe(method='idpt', list_of_dataframes=dfis,
                                                       out_of_plane_threshold=out_of_plane_threshold,
                                                       path_results=path_results, path_invalid=path_outliers)

    dfss, dfss_all = package_plane_corrected_dataframe(method='spct', list_of_dataframes=dfss,
                                                       out_of_plane_threshold=out_of_plane_threshold,
                                                       path_results=path_results, path_invalid=path_outliers)
    dfgs, dfgs_all = package_plane_corrected_dataframe(method='gdpt', list_of_dataframes=dfgs,
                                                       out_of_plane_threshold=out_of_plane_threshold,
                                                       path_results=path_results, path_invalid=path_outliers)

    dict_corr = dict({'corrected': {'IDPT': dfis, 'SPCT': dfss, 'GDPT': dfgs},
                      'corrected_all': {'IDPT': dfis_all, 'SPCT': dfss_all, 'GDPT': dfgs_all},
                      })
    dict_data.update(dict_corr)

    # --- generate plots to evaluate accuracy of plane fits

    if dict_plots['fit_plane']['z_corr_valid']:
        path_plane_by_z = join(path_results,
                               'z-corr_by_z-nominal_error-less-than-{}'.format(out_of_plane_threshold))
        path_plane_by_z_1ax = join(path_plane_by_z, 'same-axes')
        make_dir(path_plane_by_z)
        make_dir(path_plane_by_z_1ax)

        plane_clr = 'gray'

        for z_nominal in z_nominals:
            dict_fit_plane = i_fit_plane_dicts[np.round(z_nominal, 1)]
            dfiz = dfis[dfis['z_nominal'] == z_nominal]
            dfsz = dfss[dfss['z_nominal'] == z_nominal]
            dfgz = dfgs[dfgs['z_nominal'] == z_nominal]

            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 1.5, size_y_inches * 0.75))

            # plot "true" depth position
            ax1.axhline(dict_fit_plane['assert_z_f_fit_plane_image_center'],
                        linestyle='--', linewidth=0.75, color='k', label='Est. True')
            ax2.axhline(dict_fit_plane['assert_z_f_fit_plane_image_center'],
                        linestyle='--', linewidth=0.75, color='k', label='Est. True')

            for df, lbl, clr in zip([dfiz, dfsz, dfgz], ['IDPT', 'SPCT', 'GDPT'],
                                    [sciblue, scigreen, sciorange]):
                # plot positions corrected by fitted plane
                ax1.scatter(df['x'], df['z'], s=1, color=clr, alpha=1, label=lbl)

                # plot positions corrected by fitted plane
                ax2.scatter(df['y'], df['z'], s=1, color=clr, alpha=1, label=lbl)

            ax1.set_ylabel(r'$z \: (\mu m)$')
            ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize='x-small',
                       markerscale=2, borderpad=0.2, handletextpad=0.3, columnspacing=1)
            ax1.set_xlabel(r'$x \: (\mu m)$')
            ax2.set_xlabel(r'$y \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(
                join(path_plane_by_z_1ax,
                     'scatter-1ax_z-corr_by_z-nominal={}.png'.format(np.round(z_nominal, 2))))
            plt.close()

    if dict_plots['fit_plane']['z_raw_valid']:
        path_plane_by_z = join(path_results,
                               'raw-z_by_z-nominal_error-less-than-{}'.format(out_of_plane_threshold))
        path_plane_by_z_1ax = join(path_plane_by_z, 'same-axes')
        path_plane_by_z_3ax = join(path_plane_by_z, 'sep-axes')
        make_dir(path_plane_by_z)
        make_dir(path_plane_by_z_1ax)
        make_dir(path_plane_by_z_3ax)

        plane_clr = 'gray'

        for z_nominal in z_nominals:
            dict_fit_plane = i_fit_plane_dicts[np.round(z_nominal, 1)]
            dfiz = dfis[dfis['z_nominal'] == z_nominal]
            dfsz = dfss[dfss['z_nominal'] == z_nominal]
            dfgz = dfgs[dfgs['z_nominal'] == z_nominal]

            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 1.5, size_y_inches * 0.75))

            # plot fitted plane viewed along x-axis
            plane_x = dict_fit_plane['px']
            plane_y = dict_fit_plane['py']
            plane_z = dict_fit_plane['pz']
            plot_plane_along_xix = [plane_x[0][0], plane_x[0][1]]
            plot_plane_along_xiz = [plane_z[0][0], plane_z[0][1]]
            plot_plane_along_xfx = [plane_x[1][0], plane_x[1][1]]
            plot_plane_along_xfz = [plane_z[1][0], plane_z[1][1]]
            ax1.plot(plot_plane_along_xix, plot_plane_along_xiz, color=plane_clr, alpha=0.5, label='Fit Plane')
            ax1.plot(plot_plane_along_xfx, plot_plane_along_xfz, color=plane_clr, alpha=0.5)

            # plotted fitted plane viewed along y-axis
            plot_plane_along_yiy = [plane_y[0][0], plane_y[1][0]]
            plot_plane_along_yiz = [plane_z[0][0], plane_z[1][0]]
            plot_plane_along_yfy = [plane_y[0][1], plane_y[1][1]]
            plot_plane_along_yfz = [plane_z[0][1], plane_z[1][1]]
            ax2.plot(plot_plane_along_yiy, plot_plane_along_yiz, color=plane_clr, alpha=0.5, label='Fit Plane')
            ax2.plot(plot_plane_along_yfy, plot_plane_along_yfz, color=plane_clr, alpha=0.5)

            for df, lbl, clr in zip([dfiz, dfsz, dfgz], ['IDPT', 'SPCT', 'GDPT'],
                                    [sciblue, scigreen, sciorange]):
                # plot positions corrected by fitted plane
                ax1.scatter(df['x'], df['z_no_corr'], s=1, color=clr, alpha=1, label=lbl)

                # plot positions corrected by fitted plane
                ax2.scatter(df['y'], df['z_no_corr'], s=1, color=clr, alpha=1, label=lbl)

            ax1.set_ylabel(r'$z \: (\mu m)$')
            ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize='x-small',
                       markerscale=2, borderpad=0.2, handletextpad=0.3, columnspacing=1)
            ax1.set_xlabel(r'$x \: (\mu m)$')
            ax2.set_xlabel(r'$y \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(
                join(path_plane_by_z_1ax,
                     'scatter-1ax_raw-z_by_z-nominal={}.png'.format(np.round(z_nominal, 2))))
            plt.close()

            fig, axes = plt.subplots(nrows=3, sharex=True)
            for ax, df, lbl, clr in zip(axes, [dfiz, dfsz, dfgz], ['IDPT', 'SPCT', 'GDPT'],
                                        [sciblue, scigreen, sciorange]):
                # plot fitted plane viewed along x-axis
                plane_x = dict_fit_plane['px']
                plane_z = dict_fit_plane['pz']
                plot_plane_along_xix = [plane_x[0][0], plane_x[0][1]]
                plot_plane_along_xiz = [plane_z[0][0], plane_z[0][1]]
                plot_plane_along_xfx = [plane_x[1][0], plane_x[1][1]]
                plot_plane_along_xfz = [plane_z[1][0], plane_z[1][1]]
                ax.plot(plot_plane_along_xix, plot_plane_along_xiz, color=plane_clr, alpha=0.5, label='Fit Plane')
                ax.plot(plot_plane_along_xfx, plot_plane_along_xfz, color=plane_clr, alpha=0.5)

                ax.scatter(df['x'], df['z_no_corr'], s=1, color=clr, alpha=1, label=lbl)
                ax.set_ylabel(r'$z \: (\mu m)$')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small',
                          markerscale=2, borderpad=0.1, handletextpad=0.15, columnspacing=1)
            axes[-1].set_xlabel(r'$x \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(
                join(path_plane_by_z_3ax,
                     'scatter-3ax_raw-z_by_z-nominal={}.png'.format(np.round(z_nominal, 2))))
            plt.close()

    if dict_plots['fit_plane']['z_raw_all']:
        path_plane_by_z = join(path_results, 'raw-z_by_z-nominal')
        path_plane_by_z_1ax = join(path_plane_by_z, 'same-axes')
        path_plane_by_z_3ax = join(path_plane_by_z, 'sep-axes')
        make_dir(path_plane_by_z)
        make_dir(path_plane_by_z_1ax)
        make_dir(path_plane_by_z_3ax)

        for z_nominal in z_nominals:
            dict_fit_plane = i_fit_plane_dicts[np.round(z_nominal, 1)]
            dfiz = dfis_all[dfis_all['z_nominal'] == z_nominal]
            dfsz = dfss_all[dfss_all['z_nominal'] == z_nominal]
            dfgz = dfgs_all[dfgs_all['z_nominal'] == z_nominal]

            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 1.5, size_y_inches * 0.75))

            # plot fitted plane viewed along x-axis
            plane_x = dict_fit_plane['px']
            plane_y = dict_fit_plane['py']
            plane_z = dict_fit_plane['pz']
            plot_plane_along_xix = [plane_x[0][0], plane_x[0][1]]
            plot_plane_along_xiz = [plane_z[0][0], plane_z[0][1]]
            plot_plane_along_xfx = [plane_x[1][0], plane_x[1][1]]
            plot_plane_along_xfz = [plane_z[1][0], plane_z[1][1]]
            ax1.plot(plot_plane_along_xix, plot_plane_along_xiz, color=plane_clr, alpha=0.5, label='Fit Plane')
            ax1.plot(plot_plane_along_xfx, plot_plane_along_xfz, color=plane_clr, alpha=0.5)

            # plotted fitted plane viewed along y-axis
            plot_plane_along_yiy = [plane_y[0][0], plane_y[1][0]]
            plot_plane_along_yiz = [plane_z[0][0], plane_z[1][0]]
            plot_plane_along_yfy = [plane_y[0][1], plane_y[1][1]]
            plot_plane_along_yfz = [plane_z[0][1], plane_z[1][1]]
            ax2.plot(plot_plane_along_yiy, plot_plane_along_yiz, color=plane_clr, alpha=0.5, label='Fit Plane')
            ax2.plot(plot_plane_along_yfy, plot_plane_along_yfz, color=plane_clr, alpha=0.5)

            for df, lbl, clr in zip([dfiz, dfsz, dfgz], ['IDPT', 'SPCT', 'GDPT'],
                                    [sciblue, scigreen, sciorange]):
                ax1.scatter(df['x'], df['z_no_corr'], s=1, color=clr, alpha=1, label=lbl)
                ax2.scatter(df['y'], df['z_no_corr'], s=1, color=clr, alpha=1, label=lbl)
            ax1.set_ylabel(r'$z \: (\mu m)$')
            ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize='x-small',
                       markerscale=2, borderpad=0.2, handletextpad=0.3, columnspacing=1)
            ax1.set_xlabel(r'$x \: (\mu m)$')
            ax2.set_xlabel(r'$y \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(join(path_plane_by_z_1ax,
                             'scatter-1ax_raw-z_by_z-nominal={}.png'.format(np.round(z_nominal, 2))))
            plt.close()

            fig, axes = plt.subplots(nrows=3, sharex=True)
            for ax, df, lbl, clr in zip(axes, [dfiz, dfsz, dfgz], ['IDPT', 'SPCT', 'GDPT'],
                                        [sciblue, scigreen, sciorange]):
                ax.scatter(df['x'], df['z_no_corr'], s=1, color=clr, alpha=1, label=lbl)
                ax.set_ylabel(r'$z \: (\mu m)$')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small',
                          markerscale=2, borderpad=0.2, handletextpad=0.3, columnspacing=1)
            axes[-1].set_xlabel(r'$x \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(join(path_plane_by_z_3ax,
                             'scatter-3ax_raw-z_by_z-nominal={}.png'.format(np.round(z_nominal, 2))))
            plt.close()

    if dict_plots['fit_plane']['fit_accuracy']:
        # analyze fit plane xyzc and rmse-z
        df_fit_plane = pd.DataFrame(data=np.vstack([z_nominals, i_fit_plane_img_xyzc_assert,
                                                    i_fit_plane_img_xyzc, i_fit_plane_rmsez]).T,
                                    columns=['z_nominal', 'iz_xyc_true', 'iz_xyc', 'irmsez'])

        df_fit_plane['iz_diff_true'] = df_fit_plane['iz_xyc_true'] - df_fit_plane['z_nominal']
        df_fit_plane['iz_diff'] = df_fit_plane['iz_xyc'] - df_fit_plane['z_nominal']
        df_fit_plane.to_excel(join(path_results, 'micrometer-step-variation_and_fit-rmse-z_by_z-nominal.xlsx'))

        # plot fit_plane_image_xyzc (the z-position at the center of the image) and rmse-z as a function of z_true

        # step-size variation
        dz_steps = np.diff(df_fit_plane['iz_xyc_true'].to_numpy())
        dzm = np.mean(dz_steps)
        dzstd = np.std(dz_steps)

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.plot(df_fit_plane['z_nominal'], df_fit_plane['iz_diff_true'], '-s', ms=5, color='k',
                 linewidth=0.5, label='Est. True')
        ax1.plot(df_fit_plane['z_nominal'], df_fit_plane['iz_diff'], '-o', ms=5, color=sciblue,
                 linewidth=1, label='IDPT')

        ax2.plot(df_fit_plane['z_nominal'].iloc[1:], dz_steps, '-o', color='k',
                 label=r'$\overline{\Delta z}=$' + str(np.round(dzm, 2)) + r'$\pm$' + str(
                     np.round(dzstd, 2)) + r' $\mu m$')

        ax1.set_ylabel(r'$z_{fit} - z_{nom} \: (\mu m)$')
        ax1.set_ylim([-1.95, 1.75])
        ax1.set_yticks([-1, 0, 1])
        ax1.legend(loc='lower left', fontsize='small')
        ax1.grid(alpha=0.125)
        ax2.set_ylabel(r'$\Delta z \: (\mu m)$')
        ax2.set_ylim([4, 6.45])
        ax2.set_yticks([5, 6])
        ax2.set_xlabel(r'$z_{nominal}$')
        ax2.legend(loc='upper left', fontsize='small')
        ax2.grid(alpha=0.125)
        plt.tight_layout()
        plt.savefig(join(path_results, 'micrometer-step-variation_and_fit-rmse-z_by_z-nominal.png'))
        plt.close()

        # ---

        dfisc = dfis.copy()
        dfssc = dfss.copy()
        dfgsc = dfgs.copy()

        fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1.2, size_y_inches))
        for ax, df, lbl, clr in zip(axes, [dfisc, dfssc, dfgsc], ['IDPT', 'SPCT', 'GDPT'],
                                    [sciblue, scigreen, sciorange]):
            ax.scatter(df['r'], df['error_rel_plane'], s=1, color=clr, alpha=0.3, label=lbl)
            ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
            ax.set_ylim([-out_of_plane_threshold, out_of_plane_threshold])
        axes[-1].set_xlabel(r'$R \: (pix)$')
        plt.tight_layout()
        plt.savefig(join(path_results, 'scatter-error_rel_plane_by_r.png'))
        plt.close()

        # ---

        # plot tilt per frame
        dfig = dfisc.groupby('z_nominal').mean().reset_index()
        xspan = 512 * microns_per_pixel
        ms = 4

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        ax1.plot(dfig['z_nominal'], dfig.tilt_x_degrees, '-o', ms=ms, label='x', color='r')
        ax1.plot(dfig['z_nominal'], dfig.tilt_y_degrees, '-s', ms=ms, label='y', color='k')
        ax2.plot(dfig['z_nominal'], np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_x_degrees))), '-o', ms=ms, label='x',
                 color='r')
        ax2.plot(dfig['z_nominal'], np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_y_degrees))), '-s', ms=ms, label='y',
                 color='k')
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
        ax2.plot(dfig['z_nominal'], np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_x_degrees))), '-o', ms=ms, label='x',
                 color='r')
        ax2.plot(dfig['z_nominal'], np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_y_degrees))), '-s', ms=ms, label='y',
                 color='k')
        ax3.plot(df_fit_plane['z_nominal'], df_fit_plane['irmsez'], '-o', ms=ms, label='IDPT')
        ax1.set_ylabel('Tilt ' + r'$(deg.)$')
        ax1.legend()
        ax2.set_ylabel(r'$\Delta z_{FoV} \: (\mu m)$')
        ax3.set_ylabel(r'$RMSE_{z}^{fit} \: (\mu m)$')
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
        ax.set_xlabel(r'$RMSE_{z}^{fit} \: (\mu m)$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(join(path_results, 'abs-sample-tilt_by_rmsez-fit-IDPT.png'))
        plt.close()

    return dict_data


def fit_rigid_transformations(method, dict_data, dict_inputs, dict_filters, dict_paths, dict_plots, xy_units):
    if xy_units == 'microns':
        microns_per_pixel = 1
    elif xy_units == 'pixels':
        microns_per_pixel = dict_inputs['microns_per_pixel']
    else:
        raise ValueError("xy_units must be: ['microns', 'pixels'].")

    path_results = dict_paths['rigid_transformations']
    make_dir(path=path_results)

    path_outliers = dict_paths['outliers']
    make_dir(path=path_outliers)

    if dict_data['dataset_rigid_transformations'] == 'aligned':
        df = dict_data['aligned'][method.upper()]
        # filter cm
        min_cm = dict_filters['min_cm']
        df = filter_cm(dfs=[df], labels=[method], cmin=min_cm, path_results=path_outliers)
        df = df[0]
    elif dict_data['dataset_rigid_transformations'].startswith('corrected'):
        if dict_data['dataset_rigid_transformations'] == 'corrected':
            path_ = join(dict_paths['fit_plane'], '{}_error_relative_plane.xlsx'.format(method))
        elif dict_data['dataset_rigid_transformations'] == 'corrected_all':
            path_ = join(dict_paths['fit_plane'], '{}_error_relative_plane_all.xlsx'.format(method))
        else:
            raise ValueError("Dataset not understood. Options are: ['corrected', 'corrected_all']")
        df = pd.read_excel(path_)
    else:
        raise ValueError("Dataset not understood. Options are: ['aligned', 'corrected', 'corrected_all']")

    in_plane_threshold = dict_filters['in_plane_threshold']
    min_counts_icp = dict_filters['min_counts_icp']

    # read coords: "true" in-plane positions of particles at focus (measured using ImageJ)
    dfxyzf = dict_data['aligned']['TRUE'].copy()

    # 3. convert x, y, r coordinates from units pixels to microns
    for pix2microns in ['x', 'y', 'r']:
        df[pix2microns] = df[pix2microns] * microns_per_pixel
        dfxyzf[pix2microns] = dfxyzf[pix2microns] * microns_per_pixel

    # 5. rigid transformations from focus using ICP
    dfBB_icp, df_icp, df_outliers = rigid_transforms_from_focus(df, dfxyzf, min_counts_icp, in_plane_threshold,
                                                                return_outliers=True)
    dfBB_icp.to_excel(join(path_results, 'dfBB_icp_{}.xlsx'.format(method)), index=False)
    df_icp.to_excel(join(path_results, 'df_icp_{}.xlsx'.format(method)), index=False)
    if len(df_outliers) > 0:
        df_outliers.to_excel(join(path_outliers, '{}_nneigh_errxy_relative_focus_invalid-only.xlsx'.format(method)),
                             index=False)

    # 6. depth-dependent r.m.s. error
    dfdz_icp = df_icp.groupby('z').mean().reset_index()
    dfdz_icp.to_excel(join(path_results, 'dfdz_icp_{}.xlsx'.format(method)), index=False)

    # 6. depth-averaged r.m.s. error
    dfBB_icp_mean = depth_averaged_rmse_rigid_transforms_from_focus(dfBB_icp)
    dfBB_icp_mean.to_excel(join(path_results, 'icp_mean-rmse_{}.xlsx'.format(method)))

    # plot accuracy of rigid transformations
    if dict_plots['fit_rt_accuracy']:
        ms = 3
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1.1, size_y_inches * 1.5))

        ax1.plot(dfdz_icp.z, dfdz_icp.dx, '-o', ms=ms, color='r', label=r'$\Delta x$')
        ax1.plot(dfdz_icp.z, dfdz_icp.dy, '-o', ms=ms, color='b', label=r'$\Delta y$')
        ax1.plot(dfdz_icp.z, dfdz_icp.dz, '-o', ms=ms, color='k', label=r'$\Delta z$')
        ax1.set_ylabel(r'Displacement $(\mu m)$')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax2.plot(dfdz_icp.z, dfdz_icp.rmse, '-o', color='k', ms=ms)
        ax2.set_ylabel(r'$RMSE_{fit} \: (\mu m)$')

        ax3.plot(dfdz_icp.z, dfdz_icp.rmse_x, '-o', ms=ms, color='r', label=r'$x$')
        ax3.plot(dfdz_icp.z, dfdz_icp.rmse_y, '-o', ms=ms, color='b', label=r'$y$')
        ax3.plot(dfdz_icp.z, dfdz_icp.rmse_z, '-o', ms=ms, color='k', label=r'$z$')
        ax3.set_ylabel(r'$RMSE \: (\mu m)$')
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(join(path_results, 'accuracy_of_rigid_transformations_{}.png'.format(method)))
        plt.close()


def evaluate_root_mean_square_error(dict_data, dict_inputs, dict_filters, dict_paths, dict_plots, xy_units):
    if xy_units == 'microns':
        microns_per_pixel = 1
    elif xy_units == 'pixels':
        microns_per_pixel = dict_inputs['microns_per_pixel']
    else:
        raise ValueError("xy_units must be: ['microns', 'pixels'].")

    # read coords
    if dict_data['dataset_rmse'].startswith('corrected'):
        if 'IDPT' in dict_data[dict_data['dataset_rmse']].keys():
            dfi = dict_data[dict_data['dataset_rmse']]['IDPT']
            dfs = dict_data[dict_data['dataset_rmse']]['SPCT']
            dfg = dict_data[dict_data['dataset_rmse']]['GDPT']
        else:
            if dict_data['dataset_rmse'] == 'corrected':
                path_idpt = join(dict_paths['fit_plane'], 'idpt_error_relative_plane.xlsx')
                path_spct = join(dict_paths['fit_plane'], 'spct_error_relative_plane.xlsx')
                path_gdpt = join(dict_paths['fit_plane'], 'gdpt_error_relative_plane.xlsx')
            elif dict_data['dataset_rmse'] == 'corrected_all':
                path_idpt = join(dict_paths['fit_plane'], 'idpt_error_relative_plane_all.xlsx')
                path_spct = join(dict_paths['fit_plane'], 'spct_error_relative_plane_all.xlsx')
                path_gdpt = join(dict_paths['fit_plane'], 'gdpt_error_relative_plane_all.xlsx')
            else:
                raise ValueError("Dataset not understood. Options are: ['corrected', 'corrected_all']")
            dfi = pd.read_excel(path_idpt)
            dfs = pd.read_excel(path_spct)
            dfg = pd.read_excel(path_gdpt)
    elif dict_data['dataset_rmse'] == 'rigid_transformations':
        dfi = pd.read_excel(join(dict_paths['rigid_transformations'], 'dfBB_icp_idpt.xlsx'))
        dfs = pd.read_excel(join(dict_paths['rigid_transformations'], 'dfBB_icp_spct.xlsx'))
        dfg = pd.read_excel(join(dict_paths['rigid_transformations'], 'dfBB_icp_gdpt.xlsx'))
    else:
        raise ValueError("Dataset not understood. Options are: ['corrected', 'corrected_all', 'rigid_transformations']")

    # read inputs
    col_error_z = dict_data['use_columns']['rmse_error_z']

    true_num_particles_per_z = dict_inputs['true_num_particles_per_z']

    min_counts_bin_rz = dict_filters['min_counts_bin_rz']
    min_counts_bin_r = dict_filters['min_counts_bin_r']
    min_counts_bin_z = dict_filters['min_counts_bin_z']

    path_results = dict_paths['rmse_z']
    make_dir(path=path_results)

    # --- PROCESSING

    # rmse-z

    # number of z-positions
    num_z_positions = len(dfi['z_nominal'].unique())
    true_total_num = true_num_particles_per_z * num_z_positions
    # scale to microns
    dfi['r_microns'] = dfi['r'] * microns_per_pixel
    dfs['r_microns'] = dfs['r'] * microns_per_pixel
    dfg['r_microns'] = dfg['r'] * microns_per_pixel
    # square all errors
    dfi['rmse_z'] = dfi[col_error_z] ** 2
    dfs['rmse_z'] = dfs[col_error_z] ** 2
    dfg['rmse_z'] = dfg[col_error_z] ** 2

    # compute rmse_z by z
    z_trues = dfi['z_nominal'].unique()
    dfim, dfim_global = bin_by_z_fp(df=dfi,
                                    z_bins=z_trues,
                                    true_num_particles_per_z=true_num_particles_per_z,
                                    true_total_num=true_total_num,
                                    path_results=path_results,
                                    method='idpt',
                                    return_global=True)
    dfsm, dfsm_global = bin_by_z_fp(df=dfs,
                                    z_bins=z_trues,
                                    true_num_particles_per_z=true_num_particles_per_z,
                                    true_total_num=true_total_num,
                                    path_results=path_results,
                                    method='spct',
                                    return_global=True)
    dfgm, dfgm_global = bin_by_z_fp(df=dfg,
                                    z_bins=z_trues,
                                    true_num_particles_per_z=true_num_particles_per_z,
                                    true_total_num=true_total_num,
                                    path_results=path_results,
                                    method='gdpt',
                                    return_global=True)

    # TODO:  I should add empty dictionaries to dict_data so I know what data does get put in there.
    dict_rmse_z = dict({'rmse_z': {'IDPT': dfim, 'SPCT': dfsm, 'GDPT': dfgm}})
    dict_data.update(dict_rmse_z)

    # ---

    path_pubfigs = dict_paths['pubfigs']
    make_dir(path=path_pubfigs)

    path_supfigs = dict_paths['supfigs']
    make_dir(path=path_supfigs)

    # mean rmse_z via fit-plane
    depth_averaged_fps = []
    global_averaged_fps = []
    dfzs = [dfim, dfsm, dfgm]
    dfms = [dfim_global, dfsm_global, dfgm_global]
    mtds = ['idpt', 'spct', 'gdpt']
    for dfz, dfm, mtd in zip(dfzs, dfms, mtds):
        # depth-averaged
        depth_averaged_fp = dfz[['cm', 'rmse_z', 'count_id', 'true_num_per_z', 'percent_meas']]
        depth_averaged_fp['binz'] = 1
        depth_averaged_fp = depth_averaged_fp.rename(columns={'rmse_z': 'fp_rmse_z',
                                                              'count_id': 'fp_count_id',
                                                              'true_num_per_z': 'fp_true_num',
                                                              'percent_meas': 'fp_percent_meas'})
        depth_averaged_fp = depth_averaged_fp.groupby('binz').mean().reset_index().drop(columns=['binz'])
        depth_averaged_fp = depth_averaged_fp[['cm', 'fp_rmse_z', 'fp_count_id', 'fp_true_num', 'fp_percent_meas']]
        depth_averaged_fp.insert(loc=0, column='method', value=[mtd.upper()])
        depth_averaged_fps.append(depth_averaged_fp)

        # global-averaged
        global_averaged_fp = dfm[['cm', 'rmse_z', 'count_id', 'true_num', 'percent_meas']]
        global_averaged_fp.insert(loc=0, column='method', value=[mtd.upper()])
        global_averaged_fps.append(global_averaged_fp)

    depth_averaged_fps = pd.concat(depth_averaged_fps)
    global_averaged_fps = pd.concat(global_averaged_fps)

    # mean rmse via rigid transformations
    if dict_data['dataset_rmse'] == 'rigid_transformations':
        depth_averaged_icps = []
        global_averaged_icps = []
        mtds = ['idpt', 'spct', 'gdpt']
        for mtd in mtds:
            # depth-averaged r.m.s. error
            depth_averaged_icp = pd.read_excel(
                join(dict_paths['rigid_transformations'], 'df_icp_{}.xlsx'.format(mtd))).groupby('zA').mean()
            depth_averaged_icp = depth_averaged_icp[['precision', 'rmse',
                                                     'rmse_x', 'rmse_y', 'rmse_xy', 'rmse_z',
                                                     'num_icp', 'numB', 'numA',
                                                     'dx', 'dy', 'dz',
                                                     ]]
            depth_averaged_icp = depth_averaged_icp.rename(columns={'precision': 'rt_precision',
                                                                    'rmse': 'rt_rmse',
                                                                    'rmse_x': 'rt_rmse_x',
                                                                    'rmse_y': 'rt_rmse_y',
                                                                    'rmse_xy': 'rt_rmse_xy',
                                                                    'rmse_z': 'rt_rmse_z',
                                                                    'num_icp': 'rt_num_icp',
                                                                    'numB': 'rt_numB',
                                                                    'numA': 'rt_numA',
                                                                    'dx': 'rt_dx',
                                                                    'dy': 'rt_dy',
                                                                    'dz': 'rt_dz', })
            depth_averaged_icp.insert(loc=0, column='method', value=[mtd.upper()])
            depth_averaged_icps.append(depth_averaged_icp)

            # global-average r.m.s. error
            global_averaged_icp = depth_averaged_rmse_rigid_transforms_from_focus(
                pd.read_excel(join(dict_paths['rigid_transformations'], 'dfBB_icp_{}.xlsx'.format(mtd))))
            global_averaged_icp = global_averaged_icp.drop(columns=['bin'])
            global_averaged_icp = global_averaged_icp.rename(columns={'rmse_errx': 'rt_rmse_x',
                                                                      'rmse_erry': 'rt_rmse_y',
                                                                      'rmse_errxy': 'rt_rmse_xy',
                                                                      'rmse_errz': 'rt_rmse_z',
                                                                      'rmse_errxyz': 'rt_rmse', })
            global_averaged_icp.insert(loc=0, column='method', value=[mtd.upper()])
            global_averaged_icps.append(global_averaged_icp)

        depth_averaged_icps = pd.concat(depth_averaged_icps)
        global_averaged_icps = pd.concat(global_averaged_icps)

        depth_averaged_fps = depth_averaged_fps.merge(depth_averaged_icps, how='inner', on='method')
        global_averaged_fps = global_averaged_fps.merge(global_averaged_icps, how='inner', on='method')

    depth_averaged_fps.to_excel(join(path_supfigs, 'depth-averaged_performance_w_GDPT.xlsx'), index=False)
    global_averaged_fps.to_excel(join(path_supfigs, 'global-averaged_performance_w_GDPT.xlsx'), index=False)

    depth_averaged_fps = depth_averaged_fps[depth_averaged_fps['method'].isin(['IDPT', 'SPCT'])]
    depth_averaged_fps.to_excel(join(path_pubfigs, 'depth-averaged_performance.xlsx'), index=False)
    global_averaged_fps = global_averaged_fps[global_averaged_fps['method'].isin(['IDPT', 'SPCT'])]
    global_averaged_fps.to_excel(join(path_pubfigs, 'global-averaged_performance.xlsx'), index=False)

    # ---

    # bin by z
    if dict_plots['local_rmse_z']['bin_z']:
        # filter before plotting
        dfim = dfim[dfim['count_id'] > min_counts_bin_z]
        dfsm = dfsm[dfsm['count_id'] > min_counts_bin_z]
        dfgm = dfgm[dfgm['count_id'] > min_counts_bin_z]

        # plot: rmse_z by z_nominal (i.e., bin)
        fig, ax = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.))

        ax.plot(dfim.bin, dfim['rmse_z'], '-o', label='IDPT')
        ax.plot(dfsm.bin, dfsm['rmse_z'], '-o', label='SPCT')
        ax.plot(dfgm.bin, dfgm['rmse_z'], '-o', label='GDPT')

        ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_xticks([-50, -25, 0, 25, 50])
        ax.legend(loc='upper left')  # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

        plt.tight_layout()
        plt.savefig(join(path_results, 'bin-z_rmse-z_by_z' + '.png'))
        plt.close()

        # ---

        # plot: local (1) correlation coefficient, (2) percent measure, and (3) rmse_z

        # setup
        zorder_i, zorder_s, zorder_g = 3.5, 3.3, 3.4
        ms = 5

        # plot
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                            figsize=(size_x_inches * 1.35, size_y_inches * 1.25))

        for df, lbl, clr, zord in zip([dfim, dfsm, dfgm], ['IDPT', 'SPCT', 'GDPT'],
                                      [sciblue, scigreen, sciorange], [zorder_i, zorder_s, zorder_g]):
            ax1.plot(df.bin, df['cm'], '-o', ms=ms, label=lbl, color=clr, zorder=zord)
            ax2.plot(df.bin, df['percent_meas'], '-o', ms=ms, color=clr, zorder=zord)
            ax3.plot(df.bin, df['rmse_z'], '-o', ms=ms, color=clr, zorder=zord)
        ax1.set_ylabel(r'$C_{m}^{\delta}$')
        ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
        ax2.set_ylabel(r'$\phi_{z}^{\delta}$')
        ax3.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax3.set_xlabel(r'$z \: (\mu m)$')
        ax3.set_xticks([-50, -25, 0, 25, 50])
        plt.tight_layout()
        plt.savefig(join(path_results, 'bin-z_local-cm-percent-meas-rmse-z_by_z' + '.png'))
        plt.close()

        # ---

    # bin by r
    if dict_plots['local_rmse_z']['bin_r']:
        # setup 2D binning
        r_bins = [100, 225, 350, 475]
        dfim = bin_by_r_fp(dfi, r_bins, path_results, method='idpt')
        dfsm = bin_by_r_fp(dfs, r_bins, path_results, method='spct')
        dfgm = bin_by_r_fp(dfg, r_bins, path_results, method='gdpt')
        # ---
        # filter before plotting
        dfim = dfim[dfim['count_id'] > min_counts_bin_r]
        dfsm = dfsm[dfsm['count_id'] > min_counts_bin_r]
        dfgm = dfgm[dfgm['count_id'] > min_counts_bin_r]

        # plot
        fig, ax = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.))
        ax.plot(dfim.bin, dfim['rmse_z'], '-o', color=sciblue, label='IDPT')
        ax.plot(dfsm.bin, dfsm['rmse_z'], '-o', color=scigreen, label='SPCT')
        ax.plot(dfgm.bin, dfgm['rmse_z'], '-o', color=sciorange, label='GDPT')
        ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax.set_xticks([100, 200, 300, 400, 500])
        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
        plt.tight_layout()
        plt.savefig(join(path_results, 'bin-r_rmse-z_by_r' + '.png'))
        plt.close()

        # ---

    # 2d-bin by r and z
    if dict_plots['local_rmse_z']['bin_r_z']:
        # setup 2D binning
        z_bins = dfi['z_nominal'].unique()
        r_bins = [150, 300, 450]
        dfim = bin_by_rz_fp(dfi, r_bins, z_bins, min_counts_bin_rz, path_results, method='idpt')
        dfsm = bin_by_rz_fp(dfs, r_bins, z_bins, min_counts_bin_rz, path_results, method='spct')
        dfgm = bin_by_rz_fp(dfg, r_bins, z_bins, min_counts_bin_rz, path_results, method='gdpt')
        # ---

        # plot
        clrs = ['black', 'blue', 'red']
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.75))

        for i, bin_r in enumerate(dfim.bin_tl.unique()):
            dfibr = dfim[dfim['bin_tl'] == bin_r]
            ax1.plot(dfibr.bin_ll, dfibr['rmse_z'], '-o', ms=4, color=clrs[i], label=int(np.round(bin_r, 0)))

            dfsbr = dfsm[dfsm['bin_tl'] == bin_r]
            ax2.plot(dfsbr.bin_ll, dfsbr['rmse_z'], '-o', ms=4, color=clrs[i])

            dfgbr = dfgm[dfgm['bin_tl'] == bin_r]
            ax3.plot(dfgbr.bin_ll, dfgbr['rmse_z'], '-o', ms=4, color=clrs[i])

        ax1.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax1.legend(loc='lower left', bbox_to_anchor=(0.2, 1.0), ncol=3, title=r'$r^{\delta} \: (\mu m)$')
        ax1.text(0.05, 0.925, 'IDPT', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)

        ax2.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax2.text(0.05, 0.925, 'SPCT', horizontalalignment='left', verticalalignment='top', transform=ax2.transAxes)

        ax3.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax3.set_xlabel(r'$z \: (\mu m)$')
        ax3.set_xticks([-50, -25, 0, 25, 50])
        ax3.text(0.05, 0.925, 'GDPT', horizontalalignment='left', verticalalignment='top', transform=ax3.transAxes)

        plt.tight_layout()
        plt.savefig(join(path_results, 'bin_r-z_rmse-z_by_r-z' + '.png'))
        plt.savefig(join(path_supfigs, 'Figure 4a-b - Compare with GDPT.png'),
                    dpi=300)
        plt.close()

        # ---


def evaluate_field_dependent_effects(method, dict_data, dict_inputs, dict_paths, dict_plots):
    path_results = dict_paths['field_dependent_effects']
    make_dir(path=path_results)

    plot_parabola_at_each_z = dict_plots['field_dependent_effects']['plot_each_z']
    plot_parabolas_overlayed = dict_plots['field_dependent_effects']['plot_overlay']
    plot_parabola_shape_across_z = dict_plots['field_dependent_effects']['plot_shape_change']
    mean_z_per_pid = dict_plots['field_dependent_effects']['plot_mean_z_per_pid']

    if dict_data['dataset_field_dependent_effects'] == 'corrected':
        path_read = join(dict_paths['fit_plane'], '{}_error_relative_plane.xlsx'.format(method))
        fn_write = '{}_error_relative_plane_with_parabola.xlsx'.format(method)
    elif dict_data['dataset_field_dependent_effects'] == 'corrected_all':
        path_read = join(dict_paths['fit_plane'], '{}_error_relative_plane_all.xlsx'.format(method))
        fn_write = '{}_error_relative_plane_all_with_parabola.xlsx'.format(method)
    elif dict_data['dataset_field_dependent_effects'] == 'rigid_transformations':
        path_read = join(dict_paths['rigid_transformations'], 'dfBB_icp_{}.xlsx'.format(method))
        fn_write = '{}_dfBB_icp_error_relative_plane_with_parabola.xlsx'.format(method)
    else:
        raise ValueError("Dataset not understood. Options are: ['corrected', 'corrected_all']")

    # get inputs
    microns_per_pixel = dict_inputs['microns_per_pixel']

    # get coords and define z_range
    df = pd.read_excel(path_read)
    zts = df['z_nominal'].unique()

    # ---

    # configure

    if plot_parabolas_overlayed:
        figg, axx = plt.subplots()
    clrs = iter(cm.Spectral_r(np.linspace(0, 1, len(zts))))
    norm = mpl.colors.Normalize(vmin=np.min(zts), vmax=np.max(zts))
    cmap = 'Spectral_r'

    if method == 'spct':
        mclr = scigreen
    elif method == 'gdpt':
        mclr = sciorange
    else:
        raise ValueError("Method not understood")

    if plot_parabola_at_each_z:
        path_results_p_by_z = join(path_results, 'parabola_by_z', method)
        make_dir(path=path_results_p_by_z)

    # evaluate at each z-position

    # setup
    fit_As = []
    fit_Bs = []
    fit_rmins = []
    fit_rmaxs = []
    fit_rmses = []
    fit_r_squareds = []
    fit_num = []

    dfzs = []

    store_fr = []
    store_fz = []

    # step through z's
    for zt in zts:
        dfz = df[df['z_nominal'] == zt]

        # define parabola relative to position of substrate
        z_substrate = dfz['z_calib'].mean()

        def fit_parabola(x, a):
            return a * x ** 2 + z_substrate

        pr = dfz['r'].to_numpy() * microns_per_pixel
        pz = dfz['z'].to_numpy()

        popt, pcov = curve_fit(fit_parabola, pr, pz)
        rmse, r_squared = fit.calculate_fit_error(fit_results=fit_parabola(pr, *popt), data_fit_to=pz)

        dfz['z_fit_parabola'] = fit_parabola(pr, *popt)

        fit_As.append(popt[0])
        fit_Bs.append(z_substrate)
        fit_rmins.append(np.min(pr))
        fit_rmaxs.append(np.max(pr))
        fit_rmses.append(rmse)
        fit_r_squareds.append(r_squared)
        fit_num.append(len(pz))

        dfzs.append(dfz)

        fr = np.linspace(0, np.max(pr))
        fz = fit_parabola(fr, *popt)

        store_fr.append(fr)
        store_fz.append(fz - z_substrate)

        # plot
        if plot_parabola_at_each_z:
            fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches * 0.75))
            if mean_z_per_pid:
                dfz = dfz.groupby('id').mean()
                dfz_std = dfz.groupby('id').std()
                ax.errorbar(dfz['r'] * microns_per_pixel, dfz['z'], yerr=dfz_std['z'] * 2, fmt='o',
                            ms=2, capsize=2, elinewidth=1,
                            color=mclr, label=method.upper() + ': ' + r'$z_{i}$')
            else:
                ax.plot(dfz['r'] * microns_per_pixel, dfz['z'], 'o', ms=2,
                        color=mclr, label=method.upper() + ': ' + r'$z_{i}$')
            # FIT
            ax.plot(fr, fz, color='k', label='Fit: ' + r'$\epsilon_{z}(r)$')
            ax.axhline(z_substrate, color='gray', linestyle='--', label='Substrate: ' + r'$z_{true}$')
            ax.set_xlabel(r'$r \: (\mu m)$')
            ax.set_ylabel(r'$z \: (\mu m)$')
            ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
            plt.tight_layout()
            plt.savefig(join(path_results_p_by_z,
                             'fit_parabola_z-nominal={}_rmse={}.png'.format(np.round(zt, 1), np.round(rmse, 2))))
            plt.close()

        # plot z by r
        if plot_parabolas_overlayed:
            p1, = axx.plot(fr, fz - z_substrate, color=next(clrs),  # norm(zt), #
                           label=int(np.round(zt, 0)))

    res_fit = np.vstack([zts, fit_As, fit_Bs, fit_rmins, fit_rmaxs, fit_rmses, fit_r_squareds, fit_num]).T
    df_res_fit = pd.DataFrame(res_fit, columns=['z', 'a', 'b', 'rmin', 'rmax', 'rmse', 'r_sq', 'num_pids'])
    df_res_fit.to_excel(join(path_results, '{}_fit_results.xlsx'.format(method)), index=False)

    dfzs = pd.concat(dfzs)
    dfzs.to_excel(join(path_results, fn_write))

    if plot_parabolas_overlayed:
        # outer figure
        axx.set_xlabel(r'$r \: (\mu m)$')
        axx.set_ylabel(r'$\epsilon_{z}(r) \: (\mu m)$')

        # color bar
        figg.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axx, label=r'$z \: (\mu m)$')

        plt.tight_layout()
        plt.savefig(join(path_results, '{}_fit-curvature_by_z_cbar.png'.format(method)))
        plt.close()

    if plot_parabola_shape_across_z:
        fig, ax = plt.subplots()
        ax.plot(zts, fit_As, '-o', color=mclr, label='Fit: ' + r'$ax^2+z_{substrate}$')
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_xticks([-50, -25, 0, 25, 50])
        ax.set_ylabel(r'$a$')
        ax.legend()
        plt.tight_layout()
        plt.savefig(join(path_results, '{}_fit-parabola_by_z.png'.format(method)))
        plt.close()


def plot_pubfigs(dict_data, dict_inputs, dict_paths, dict_plots):
    path_results = dict_paths['pubfigs']
    make_dir(path=path_results)

    plot_figure_3 = dict_plots['pubfigs']['Figure3']
    plot_figure_4 = dict_plots['pubfigs']['Figure4']

    if plot_figure_3:
        # read: rmse_z
        dfim = pd.read_excel(dict_paths['read_rmse_z']['IDPT'])
        dfsm = pd.read_excel(dict_paths['read_rmse_z']['SPCT'])

        # read: rmse_xy
        dfirt = pd.read_excel(dict_paths['read_rmse_xy']['IDPT'])
        dfsrt = pd.read_excel(dict_paths['read_rmse_xy']['SPCT'])

        # ---

        # plot local correlation coefficient

        # setup - general
        clr_i = sciblue
        clr_s = scigreen
        lgnd_i = 'IDPT'
        lgnd_s = 'SPCT'
        zorder_i, zorder_s = 3.5, 3.3

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
        py12 = 'rmse_xy'

        ylbl_cm = r'$\overline{c_{m}}(z)$'
        ylim_cm = [0.71, 1.02]  # data range: [0.7, 1.0]
        yticks_cm = [0.8, 0.9, 1.0]  # data ticks: np.arange(0.75, 1.01, 0.05)

        ylbl_phi = r"$\overline{N'_{p}}(z)/N_{p}$"
        ylim_phi = [0, 1.1]
        yticks_phi = [0, 0.5, 1]

        ylbl_rmse_xy = r'$\sigma_{xy}(z) \: (\mu m)$'
        ylbl_rmse_z = r'$\sigma_{z}(z) \: (\mu m)$'

        # plot
        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(size_x_inches * 2, size_y_inches * 1.05))

        ax1, ax3, ax2, ax4 = axs.ravel()

        ax1.plot(dfim[px], dfim[py], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax1.plot(dfsm[px], dfsm[py], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)
        ax1.set_ylabel(ylbl_cm)
        ax1.set_ylim(ylim_cm)
        ax1.set_yticks(yticks_cm)
        ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)

        ax2.plot(dfim[px], dfim[pyb], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax2.plot(dfsm[px], dfsm[pyb], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)
        ax2.set_ylabel(ylbl_phi)
        ax2.set_ylim(ylim_phi)
        ax2.set_yticks(yticks_phi)
        ax2.set_xlabel(xlbl)
        ax2.set_xticks(xticks)

        ax3.plot(dfirt[px1], dfirt[py12], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax3.plot(dfsrt[px1], dfsrt[py12], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)
        ax3.set_ylabel(ylbl_rmse_xy)

        ax4.plot(dfim[px], dfim[py4], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax4.plot(dfsm[px], dfsm[py4], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)
        ax4.set_xlabel(xlbl)
        ax4.set_xticks(xticks)
        ax4.set_ylabel(ylbl_rmse_z)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # hspace=0.175, wspace=0.25

        plt.savefig(join(path_results, 'Figure 3 - Comparison of measurement performance.png'), dpi=300)
        plt.close()

    if plot_figure_4:

        # get inputs
        correct_field_dependent_effects = False
        microns_per_pixel = dict_inputs['microns_per_pixel']

        # LOAD DATA

        # load coords
        if dict_data['dataset_field_dependent_effects'] == 'corrected':
            path_read_idpt = join(dict_paths['fit_plane'], 'idpt_error_relative_plane.xlsx')
            path_read_spct = join(dict_paths['field_dependent_effects'], 'spct_error_relative_plane_with_parabola.xlsx')
        elif dict_data['dataset_field_dependent_effects'] == 'corrected_all':
            path_read_idpt = join(dict_paths['fit_plane'], 'idpt_error_relative_plane_all.xlsx')
            path_read_spct = join(dict_paths['field_dependent_effects'],
                                  'spct_error_relative_plane_all_with_parabola.xlsx')
        elif dict_data['dataset_field_dependent_effects'] == 'rigid_transformations':
            path_read_idpt = join(dict_paths['rigid_transformations'], 'dfBB_icp_idpt.xlsx')
            path_read_spct = join(dict_paths['field_dependent_effects'],
                                  'spct_dfBB_icp_error_relative_plane_with_parabola.xlsx')
        else:
            raise ValueError("Dataset not understood. Options are: ['corrected', 'corrected_all']")
        dfi = pd.read_excel(path_read_idpt)
        dfs = pd.read_excel(path_read_spct)

        # load fit parabolas
        df_res_fit = pd.read_excel(join(dict_paths['field_dependent_effects'], 'spct_fit_results.xlsx'))

        # ---

        # setup

        def fit_parabola_ab(x, a, b):
            return a * x ** 2 + b

        zts = dfi['z_nominal'].unique()
        clrs_ex = iter(cm.Spectral_r(np.linspace(0, 1, len(zts))))
        # norm_ex = mpl.colors.Normalize(vmin=np.min(zts), vmax=np.max(zts))
        # cmap_ex = 'Spectral_r'

        # iterate
        for zt in zts:

            clr_ex_errz = next(clrs_ex)
            if zt < 35 or zt > 40:
                continue

            fig, axs = plt.subplots(2, 2,
                                    figsize=(size_x_inches * 2.175, size_y_inches * 1.25),
                                    sharex=False, layout='constrained',
                                    gridspec_kw={'wspace': 0.15, 'hspace': 0.125})
            ax1, ax2, ax3, ax4 = axs.ravel()
            ax_idpt = ax1
            ax_spct = ax3
            ax_z = ax2
            ax_parabola = ax4

            # plot

            # color bar
            clrs = iter(cm.Spectral_r(np.linspace(0, 1, len(zts))))
            norm = mpl.colors.Normalize(vmin=np.min(zts), vmax=np.max(zts))
            cmap = 'Spectral_r'
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_parabola, label=r'$z \: (\mu m)$')

            # --- plot particle positions and parabola for one z_nominal

            dfsz = dfs[dfs['z_nominal'] == zt]
            dfiz = dfi[dfi['z_nominal'] == zt]

            # Fit
            az = df_res_fit[df_res_fit['z'] == zt]['a'].iloc[0]
            bz = df_res_fit[df_res_fit['z'] == zt]['b'].iloc[0]

            # plot z by r
            pr = dfsz['r'].to_numpy() * microns_per_pixel
            pz = dfsz['z'].to_numpy()
            fr = np.linspace(0, np.max(pr))
            fz = fit_parabola_ab(fr, az, bz)

            # plot
            p_idpt, = ax_z.plot(dfiz['r'] * microns_per_pixel, dfiz['z'], 'o', ms=2, color=sciblue, label='IDPT')
            p_spct, = ax_z.plot(dfsz['r'] * microns_per_pixel, dfsz['z'], 'o', ms=2, color=scigreen, label='SPCT')
            p_sub = ax_z.axhline(bz, color='gray', linestyle='--', label=r'$z_{substrate}$')
            ax_z.set_ylabel(r'$z \: (\mu m)$')
            ax_z.tick_params(axis="x", labelbottom=False)
            ax_z_ylim = ax_z.get_ylim()

            # legend: z-coordinates
            """z_legend = ax_z.legend(handles=[p_idpt, p_spct, p_sub],
                                   ncol=3, loc='lower left', bbox_to_anchor=(0.0, 1.0), borderpad=0.05,
                                   handlelength=1.2, handletextpad=0.3, labelspacing=0.3, columnspacing=1)"""
            z_legend = ax_z.legend(handles=[p_idpt, p_spct, p_sub], loc='lower left',
                                   handlelength=1.5, handletextpad=0.4, labelspacing=0.4)
            ax_z.add_artist(z_legend)

            # legend: z-error-coordinates
            ax_zerr = ax_z.twinx()
            p_errz, = ax_zerr.plot(fr, fz - bz, color=clr_ex_errz, label=r'$\epsilon_{z}(r)$')
            ax_zerr.set_ylim([ax_z_ylim[0] - bz, ax_z_ylim[1] - bz])
            ax_zerr.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
            ax_zerr.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0))  # , borderpad=0.05, handlelength=1.2)

            # --- plot parabolas across all z's

            all_as = df_res_fit['a'].to_numpy()
            all_bs = df_res_fit['b'].to_numpy()
            all_rmaxs = df_res_fit['rmax'].to_numpy()
            for a_z, b_z, rmax_z in zip(all_as, all_bs, all_rmaxs):
                fr = np.linspace(0, rmax_z)
                fz = fit_parabola_ab(fr, a_z, 0)
                p1, = ax_parabola.plot(fr, fz, color=next(clrs), label=int(np.round(zt, 0)))
            ax_parabola.set_xlabel(r'$r \: (\mu m)$')
            ax_parabola.set_ylabel(r'$\epsilon_{z}(r) \: (\mu m)$')
            ax_parabola.text(0.05, 0.89, 'SPCT',
                             horizontalalignment='left', verticalalignment='top', transform=ax_parabola.transAxes)

            # --- 2d-bin by r and z

            # read
            dfim = pd.read_excel(join(dict_paths['rmse_z'], 'idpt_bin_r-z_rmse-z.xlsx'))
            dfsm = pd.read_excel(join(dict_paths['rmse_z'], 'spct_bin_r-z_rmse-z.xlsx'))

            # plot
            clrs = ['black', 'blue', 'red']

            for i, bin_r in enumerate(dfim.bin_tl.unique()):
                dfibr = dfim[dfim['bin_tl'] == bin_r]
                ax_idpt.plot(dfibr.bin_ll, dfibr['rmse_z'], '-o', ms=4, color=clrs[i], label=int(np.round(bin_r, 0)))

                dfsbr = dfsm[dfsm['bin_tl'] == bin_r]
                ax_spct.plot(dfsbr.bin_ll, dfsbr['rmse_z'], '-o', ms=4, color=clrs[i], label=int(np.round(bin_r, 0)))

            ax_idpt.set_ylabel(r'$\sigma_{z}^{\delta}(z) \: (\mu m)$')
            ax_idpt.set_ylim([0, 3.1])
            ax_idpt.set_yticks([0, 1, 2, 3])

            ax_idpt.legend(loc='lower left', bbox_to_anchor=(0.05, 1.0), ncol=3, title=r'$r^{\delta} \: (\mu m)$')
            """ax_idpt.legend(loc='upper right', title=r'$r^{\delta} \: (\mu m)$', # ncol=3,
                           markerscale=0.8, borderpad=0.3, handlelength=1.2, handletextpad=0.6,
                           labelspacing=0.25, columnspacing=1.5)"""
            ax_idpt.text(0.05, 0.89, 'IDPT',
                         horizontalalignment='left', verticalalignment='top', transform=ax_idpt.transAxes)

            ax_spct.set_ylabel(r'$\sigma_{z}^{\delta}(z) \: (\mu m)$')
            ax_spct.set_ylim([0, 3.1])
            ax_spct.set_yticks([0, 1, 2, 3])
            ax_spct.set_xlabel(r'$z \: (\mu m)$')
            ax_spct.set_xticks([-50, -25, 0, 25, 50])
            ax_spct.text(0.05, 0.89, 'SPCT',
                         horizontalalignment='left', verticalalignment='top', transform=ax_spct.transAxes)

            ax_idpt.tick_params(axis="x", labelbottom=False)

            # ---

            # ref: https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html
            # fig.set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2, wspace=0.2)
            # .get_layout_engine()

            plt.savefig(join(path_results, 'Figure 4 - Field-dependent effects at z={}.png'.format(np.round(zt, 2))),
                        dpi=300, facecolor='white')
            plt.close()


def plot_supfigs(method, dict_data, dict_inputs, dict_filters, dict_paths, dict_plots, xy_units):
    if xy_units == 'microns':
        microns_per_pixel = 1
    elif xy_units == 'pixels':
        microns_per_pixel = dict_inputs['microns_per_pixel']
    else:
        raise ValueError("xy_units must be: ['microns', 'pixels'].")

    path_results = dict_paths['supfigs']
    make_dir(path=path_results)
    lim_errxy = dict_filters['in_plane_threshold']
    lim_errz = dict_filters['out_of_plane_threshold']

    if dict_plots['supfigs']['Figure3_GDPT']:
        # read: rmse_z
        dfim = pd.read_excel(dict_paths['read_rmse_z']['IDPT'])
        dfsm = pd.read_excel(dict_paths['read_rmse_z']['SPCT'])
        dfgm = pd.read_excel(dict_paths['read_rmse_z']['GDPT'])

        # read: rmse_xy
        dfirt = pd.read_excel(dict_paths['read_rmse_xy']['IDPT'])
        dfsrt = pd.read_excel(dict_paths['read_rmse_xy']['SPCT'])
        dfgrt = pd.read_excel(dict_paths['read_rmse_xy']['GDPT'])

        # ---

        # plot local correlation coefficient

        # setup - general
        clr_i = sciblue
        clr_s = scigreen
        clr_g = sciorange
        lgnd_i = 'IDPT'
        lgnd_s = 'SPCT'
        lgnd_g = 'GDPT'
        zorder_i, zorder_s, zorder_g = 3.5, 3.3, 3.4

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
        py12 = 'rmse_xy'

        ylbl_cm = r'$\overline{c_{m}}(z)$'
        ylim_cm = [0.71, 1.02]  # data range: [0.7, 1.0]
        yticks_cm = [0.8, 0.9, 1.0]  # data ticks: np.arange(0.75, 1.01, 0.05)

        ylbl_phi = r"$\overline{N'_{p}}(z)/N_{p}$"
        ylim_phi = [0, 1.1]
        yticks_phi = [0, 0.5, 1]

        ylbl_rmse_xy = r'$\sigma_{xy}(z) \: (\mu m)$'
        ylbl_rmse_z = r'$\sigma_{z}(z) \: (\mu m)$'

        # plot
        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(size_x_inches * 2, size_y_inches * 1.05))

        ax2, ax3, ax1, ax4 = axs.ravel()

        ax1.plot(dfim[px], dfim[py], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax1.plot(dfsm[px], dfsm[py], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)
        ax1.plot(dfgm[px], dfgm[py], '-o', ms=ms, color=clr_g, label=lgnd_g, zorder=zorder_g)
        ax1.set_xlabel(xlbl)
        ax1.set_xticks(xticks)
        ax1.set_ylabel(ylbl_cm)
        ax1.set_ylim(ylim_cm)
        ax1.set_yticks(yticks_cm)

        ax2.plot(dfim[px], dfim[pyb], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax2.plot(dfsm[px], dfsm[pyb], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)
        ax2.plot(dfgm[px], dfgm[pyb], '-o', ms=ms, color=clr_g, label=lgnd_g, zorder=zorder_g)
        ax2.set_ylabel(ylbl_phi)
        ax2.set_ylim(ylim_phi)
        ax2.set_yticks(yticks_phi)
        ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.0),
                   ncol=3)  # loc='upper left', bbox_to_anchor=(1, 1)) , ncol=2

        ax3.plot(dfirt[px1], dfirt[py12], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax3.plot(dfsrt[px1], dfsrt[py12], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)
        ax3.plot(dfgrt[px1], dfgrt[py12], '-o', ms=ms, color=clr_g, label=lgnd_g, zorder=zorder_g)
        ax3.set_ylabel(ylbl_rmse_xy)

        ax4.plot(dfim[px], dfim[py4], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
        ax4.plot(dfsm[px], dfsm[py4], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)
        ax4.plot(dfgm[px], dfgm[py4], '-o', ms=ms, color=clr_g, label=lgnd_g, zorder=zorder_g)
        ax4.set_xlabel(xlbl)
        ax4.set_xticks(xticks)
        ax4.set_ylabel(ylbl_rmse_z)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # hspace=0.175, wspace=0.25

        plt.savefig(join(path_results, 'Figure 3 - Compare with GDPT.png'), dpi=300)
        plt.close()

    # ---

    if dict_plots['supfigs']['outliers']:
        path_cmin_outliers = join(dict_paths['outliers'], '{}_cm-invalid-only.xlsx'.format(method))
        path_errz_outliers = join(dict_paths['outliers'], '{}_error_relative_plane_invalid-only.xlsx'.format(method))
        path_errxy_outliers = join(dict_paths['outliers'],
                                   '{}_nneigh_errxy_relative_focus_invalid-only.xlsx'.format(method))

        dict_outliers = {}
        if os.path.exists(path_cmin_outliers):
            dfc = pd.read_excel(path_cmin_outliers)
            dfcg = dfc.groupby('z_nominal').count().reset_index()
            dict_outliers.update({'cmin': {'df': dfc, 'dfg': dfcg, 'label': r'$c_{m}$', 'color': 'k'}})

        if os.path.exists(path_errz_outliers):
            dfz = pd.read_excel(path_errz_outliers)
            dfzg = dfz.groupby('z_nominal').count().reset_index()
            dict_outliers.update({'errz': {'df': dfz, 'dfg': dfzg, 'label': r'$\epsilon_{z}$', 'color': 'r'}})

        if os.path.exists(path_errxy_outliers):
            dfxy = pd.read_excel(path_errxy_outliers)
            dfxyg = dfxy.groupby('z_nominal').count().reset_index()
            dict_outliers.update({'errxy': {'df': dfxy, 'dfg': dfxyg, 'label': r'$\epsilon_{xy}$', 'color': 'b'}})

        # plot setup
        xlbl = r'$z \: (\mu m)$'
        xticks = [-50, -25, 0, 25, 50]

        # plot outliers by z_nominal
        fig, ax = plt.subplots(figsize=(size_x_inches / 1.25, size_y_inches / 1.25))
        for k, v in dict_outliers.items():
            ax.plot(v['dfg']['z_nominal'], v['dfg']['id'], '-o', color=v['color'], label=v['label'])
        ax.set_xlabel(xlbl)
        ax.set_xticks(xticks)
        ax.set_ylabel('Counts')
        ax.legend()
        plt.tight_layout()
        plt.savefig(join(path_results, '{}_outliers_by_type-and-z.png'.format(method)))
        plt.close()

        # plot focal plane bias errors relative to z-errors
        if 'errz' in dict_outliers.keys():
            df = dict_outliers['errz']['df']
            dfpb = df[(df['z_calib'] > 0) & (df['z_no_corr'] < 0) | (df['z_calib'] < 0) & (df['z_no_corr'] > 0)]
            df = df[(df['z_calib'] > 0) & (df['z_no_corr'] > 0) | (df['z_calib'] < 0) & (df['z_no_corr'] < 0)]

            dfpbg = dfpb.groupby('z_nominal').count().reset_index()
            dfg = df.groupby('z_nominal').count().reset_index()

            fig, ax = plt.subplots(figsize=(size_x_inches / 1.25, size_y_inches / 1.25))
            ax.plot(dfpbg['z_nominal'], dfpbg['id'], '-o', color='r', label='f.p.b.')
            ax.plot(dfg['z_nominal'], dfg['id'], '-o', color='k', label='other')
            ax.set_xlabel(xlbl)
            ax.set_xticks(xticks)
            ax.set_ylabel('Counts')
            ax.legend()
            plt.tight_layout()
            plt.savefig(join(path_results, '{}_focal-plane-bias-errors_by_z.png'.format(method)))
            plt.close()

    # ---

    if dict_plots['supfigs']['hist_z']:
        if dict_data['dataset_rmse'] == 'corrected':
            path_ = join(dict_paths['fit_plane'], '{}_error_relative_plane.xlsx'.format(method))
        elif dict_data['dataset_rmse'] == 'corrected_all':
            path_ = join(dict_paths['fit_plane'], '{}_error_relative_plane_all.xlsx'.format(method))
        elif dict_data['dataset_rmse'] == 'rigid_transformations':
            path_ = join(dict_paths['rigid_transformations'], 'dfBB_icp_{}.xlsx'.format(method))
        else:
            raise ValueError("Dataset not understood. Options are: ['corrected', 'corrected_all', 'rigid_transformations']")

        # 1. read test coords
        df = pd.read_excel(path_)

        # histogram of z-errors
        error_col = 'error_rel_plane'
        binwidth_y = 0.2
        xlim = lim_errz * 1.05

        # hist-z
        y = df[error_col].to_numpy()

        # plot
        fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5))
        ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
        ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
        ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)
        ax.hist(y, bins=ybins, orientation='vertical', color='gray', zorder=2.5)
        ax.set_xlabel(r'$\epsilon_{z} \: (\mu m)$')
        ax.set_xlim([-xlim, xlim])
        ax.set_ylabel('Counts')
        ax.text(0.075, 0.925, method.upper(),
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(join(path_results, '{}_histogram_z-errors.png'.format(method)))
        plt.close()

        # ---

        # plot histogram of all non-focal plane bias z-errors
        path_errz_outliers = join(dict_paths['outliers'], '{}_error_relative_plane_invalid-only.xlsx'.format(method))
        if os.path.exists(path_errz_outliers):
            df2 = pd.read_excel(path_errz_outliers)
            df2 = df2[(df2['z_calib'] > 0) & (df2['z_no_corr'] > 0) | (df2['z_calib'] < 0) & (df2['z_no_corr'] < 0)]

            if len(df2) > 1:
                error_col = 'error_rel_plane'
                binwidth_y = 0.4
                xlim = lim_errz * 3.05

                # plot
                fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5))

                # histogram "valid" z-errors
                y = df[error_col].to_numpy()
                ylim_low = (int(np.min(y) / binwidth_y) - 1.5) * binwidth_y  # + binwidth_y
                ylim_high = (int(np.max(y) / binwidth_y) + 1.5) * binwidth_y - binwidth_y
                ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)
                ax.hist(y, bins=ybins, orientation='vertical', color='darkseagreen', zorder=2.5, label='valid')

                # histogram "invalid" z-errors
                y2 = df2[error_col].to_numpy()
                ylim_low = (int(np.min(y2) / binwidth_y) - 1.5) * binwidth_y  # + binwidth_y
                ylim_high = (int(np.max(y2) / binwidth_y) + 1.5) * binwidth_y - binwidth_y
                ybins2 = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)
                ax.hist(y2, bins=ybins2, orientation='vertical', color='rosybrown', zorder=2.5, label='invalid')
                ax.set_xlabel(r'$\epsilon_{z} \: (\mu m)$')
                ax.set_xlim([-xlim, xlim])
                ax.set_ylabel('Counts')
                ax.legend(loc='upper right', handlelength=1, handletextpad=0.4)
                ax.text(0.075, 0.925, method.upper(),
                        horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
                plt.tight_layout()
                plt.savefig(join(path_results, '{}_histogram_non-fpb-z-errors.png'.format(method)), dpi=300)
                plt.close()

    # ---

    if dict_plots['supfigs']['hist_xy']:
        # 1. read test coords
        if method == 'idpt':
            df = pd.read_excel(dict_paths['read_errors_xy']['IDPT'])
        elif method == 'spct':
            df = pd.read_excel(dict_paths['read_errors_xy']['SPCT'])
        elif method == 'gdpt':
            df = pd.read_excel(dict_paths['read_errors_xy']['GDPT'])
        else:
            raise ValueError("method not understood. Options are: ['idpt', 'spct', 'gdpt']")

        # plot histogram
        err_cols = ['errx', 'erry', 'errz']
        err_lbls = ['x-residuals', 'y-residuals', 'z-residuals']
        xlims = np.array([lim_errxy, lim_errxy, lim_errz]) * 1.05
        binwidth_y = 0.25
        bandwidth_y = 0.5

        for ycol, xlbl, xlim in zip(err_cols, err_lbls, xlims):
            y = df[ycol].to_numpy()
            fig, ax = scatter_and_kde_y(y, binwidth_y=binwidth_y, kde=False, bandwidth_y=bandwidth_y)
            ax.set_xlabel(xlbl + r'$(\mu m)$')
            ax.set_xlim([-xlim, xlim])
            ax.set_ylabel('Counts')
            ax.text(0.075, 0.925, method.upper(),
                    horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            plt.tight_layout()
            plt.savefig(join(path_results, '{}_hist_{}_of_rigid_transformations.png'.format(method, ycol)))

    # ---

    if dict_plots['supfigs']['rmse_z_by_cmin']:
        # read coords
        if dict_data['dataset_rmse'].startswith('corrected'):
            if dict_data['dataset_rmse'] == 'corrected':
                path_ = join(dict_paths['fit_plane'], '{}_error_relative_plane.xlsx'.format(method))
            elif dict_data['dataset_rmse'] == 'corrected_all':
                path_ = join(dict_paths['fit_plane'], '{}_error_relative_plane_all.xlsx'.format(method))
            else:
                raise ValueError("Dataset not understood. Options are: ['corrected', 'corrected_all']")
        elif dict_data['dataset_rmse'] == 'rigid_transformations':
            path_ = join(dict_paths['rigid_transformations'], 'dfBB_icp_{}.xlsx'.format(method))
        else:
            raise ValueError(
                "Dataset not understood. Options are: ['corrected', 'corrected_all', 'rigid_transformations']")
        df = pd.read_excel(path_)

        # setup
        num_frames_total = dict_inputs['num_frames_total']
        true_num_particles_per_frame = dict_inputs['true_num_particles_per_frame']
        num_particles_total = num_frames_total * true_num_particles_per_frame
        col_error_z = dict_data['use_columns']['rmse_error_z']
        c_min_filter = dict_filters['min_cm']

        column_to_bin = 'z_nominal'
        bins = df[column_to_bin].unique()
        column_to_count = 'id'
        round_to_decimal = 1
        return_groupby = True

        # evaluate rmse_z at each c_min
        c_mins = np.linspace(c_min_filter, 1, 50)

        num_particless = []
        rmse_depth_averageds = []
        rmse_global_averages = []
        for c_min in c_mins:
            # --- depth-averaged rmse-z
            # square all errors
            dfcmda = df[df['cm'] > c_min]
            dfcmda['rmse_z'] = dfcmda[col_error_z] ** 2
            # compute 1D bin (z)
            dfm, dfstd = bin.bin_generic(dfcmda, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)
            # compute rmse-z
            dfm['rmse_z'] = np.sqrt(dfm['rmse_z'])
            rmse_depth_averaged = dfm['rmse_z'].mean()

            # --- global-averaged rmse-z
            dfcm = df[df['cm'] > c_min]
            num_p = len(dfcm)
            errors = df[col_error_z].to_numpy()
            sq_errors = np.square(errors)
            mean_sq_errors = np.mean(sq_errors)
            rmse_global_average = np.sqrt(mean_sq_errors)

            num_particless.append(num_p)
            rmse_depth_averageds.append(rmse_depth_averaged)
            rmse_global_averages.append(rmse_global_average)

        res = pd.DataFrame(np.vstack([c_mins, rmse_depth_averageds, rmse_global_averages, num_particless]).T,
                           columns=['cmin', 'rmse_z_da', 'rmse_z_ga', 'num_meas'])
        res['percent_meas'] = res['num_meas'] / num_particles_total
        res['avg_num_meas_by_z'] = res['percent_meas'] / num_frames_total
        res.to_excel(join(path_results, '{}_rmse-z_by_c-min.xlsx'.format(method)))

        fig, ax = plt.subplots(figsize=(size_x_inches / 1.25, size_y_inches / 1.5))

        ax.plot(res['cmin'], res['percent_meas'], 'k-o', ms=2, zorder=3.1)
        ax.set_ylabel(r"$\overline{N'_{p}}(z)/N_{p}$", color='k')
        ax.set_xlabel(r'$c_{min}$')
        ax.text(0.1, 0.1, method.upper(), horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

        ax2 = ax.twinx()
        ax2.plot(res['cmin'], res['rmse_z_da'], 'r-o', ms=2, zorder=3.2)
        ax2.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$', color='r')

        plt.tight_layout()
        plt.savefig(join(path_results, '{}_rmse-z_by_c-min.png'.format(method)), dpi=300, facecolor='white')
        plt.close()

        # ---

        # plot methods on same figure
        if os.path.exists(join(path_results, 'idpt_rmse-z_by_c-min.xlsx')) and os.path.exists(join(path_results, 'spct_rmse-z_by_c-min.xlsx')):
            dfi = pd.read_excel(join(path_results, 'idpt_rmse-z_by_c-min.xlsx'))
            dfs = pd.read_excel(join(path_results, 'spct_rmse-z_by_c-min.xlsx'))

            results = [dfi, dfs]
            clrs = [sciblue, scigreen]
            lbls = ['IDPT', 'SPCT']
            save_id = 'compare_rmse-z_by_c-min'

            if os.path.exists(join(path_results, 'gdpt_rmse-z_by_c-min.xlsx')):
                dfg = pd.read_excel(join(path_results, 'gdpt_rmse-z_by_c-min.xlsx'))
                results.extend([dfg])
                clrs.extend([sciorange])
                lbls.extend(['GDPT'])
                save_id = save_id + '_w_GDPT'

            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))
            for res, clr, lbl in zip(results, clrs, lbls):
                ax1.plot(res['cmin'], res['percent_meas'], '-o', ms=2, color=clr, label=lbl)
                ax2.plot(res['cmin'], res['rmse_z_da'], '-o', ms=2, color=clr, label=lbl)

            ax1.set_ylabel(r"$\overline{N'_{p}}(z)/N_{p}$")
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

            ax2.set_xlabel(r'$c_{min}$')
            ax2.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')

            plt.tight_layout()
            plt.savefig(join(path_results, save_id + '.png'), dpi=300, facecolor='white')
            plt.close()

    # ---

    if dict_plots['supfigs']['compare_calibration_particle'] and method == 'idpt':
        # read coords
        if dict_data['dataset_rmse'].startswith('corrected'):
            if dict_data['dataset_rmse'] == 'corrected':
                path_idpt = join(dict_paths['fit_plane'], 'idpt_error_relative_plane.xlsx')
                path_spct = join(dict_paths['fit_plane'], 'spct_error_relative_plane.xlsx')
                path_gdpt = join(dict_paths['fit_plane'], 'gdpt_error_relative_plane.xlsx')
            elif dict_data['dataset_rmse'] == 'corrected_all':
                path_idpt = join(dict_paths['fit_plane'], 'idpt_error_relative_plane_all.xlsx')
                path_spct = join(dict_paths['fit_plane'], 'spct_error_relative_plane_all.xlsx')
                path_gdpt = join(dict_paths['fit_plane'], 'gdpt_error_relative_plane_all.xlsx')
            else:
                raise ValueError("Dataset not understood. Options are: ['corrected', 'corrected_all']")
            dfi = pd.read_excel(path_idpt)
            dfs = pd.read_excel(path_spct)
            dfg = pd.read_excel(path_gdpt)
        elif dict_data['dataset_rmse'] == 'rigid_transformations':
            dfi = pd.read_excel(join(dict_paths['rigid_transformations'], 'dfBB_icp_idpt.xlsx'))
            dfs = pd.read_excel(join(dict_paths['rigid_transformations'], 'dfBB_icp_spct.xlsx'))
            dfg = pd.read_excel(join(dict_paths['rigid_transformations'], 'dfBB_icp_gdpt.xlsx'))
        else:
            raise ValueError(
                "Dataset not understood. Options are: ['corrected', 'corrected_all', 'rigid_transformations']")

        # setup
        col_error_z = dict_data['use_columns']['rmse_error_z']
        cx, cy, cbb = 395 / microns_per_pixel, 370 / microns_per_pixel, 25 / microns_per_pixel

        dfi = dfi[(dfi['x'] > cx - cbb) & (dfi['x'] < cx + cbb) & (dfi['y'] > cy - cbb) & (dfi['y'] < cy + cbb)]
        dfs = dfs[(dfs['x'] > cx - cbb) & (dfs['x'] < cx + cbb) & (dfs['y'] > cy - cbb) & (dfs['y'] < cy + cbb)]
        dfg = dfg[(dfg['x'] > cx - cbb) & (dfg['x'] < cx + cbb) & (dfg['y'] > cy - cbb) & (dfg['y'] < cy + cbb)]

        rmsez_ga_i = np.sqrt(np.mean(np.square(dfi[col_error_z].to_numpy())))
        rmsez_ga_s = np.sqrt(np.mean(np.square(dfs[col_error_z].to_numpy())))
        rmsez_ga_g = np.sqrt(np.mean(np.square(dfg[col_error_z].to_numpy())))

        dfi = dfi.groupby('z_nominal').mean().reset_index()
        dfs = dfs.groupby('z_nominal').mean().reset_index()
        dfg = dfg.groupby('z_nominal').mean().reset_index()

        ms = 2
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.2, size_y_inches * 1.1))

        ax1.plot(dfi['z_nominal'], dfi['cm'], '-o', ms=ms, color=sciblue, label='IDPT')
        ax1.plot(dfs['z_nominal'], dfs['cm'], '-o', ms=ms, color=scigreen, label='SPCT')
        # ax1.plot(dfg['z_nominal'], dfg['cm'], '-o', ms=ms, color=sciorange, label='GDPT')

        ax2.plot(dfi['z_nominal'], dfi['z'] - dfi['z_calib'], '-o', ms=ms, color=sciblue, label='{}'.format(np.round(rmsez_ga_i, 2)))
        ax2.plot(dfs['z_nominal'], dfs['z'] - dfs['z_calib'], '-o', ms=ms, color=scigreen, label='{}'.format(np.round(rmsez_ga_s, 2)))
        # ax2.plot(dfg['z_nominal'], dfg['z'] - dfg['z_calib'], '-o', ms=ms, color=sciorange, label='{}'.format(np.round(rmsez_ga_g, 2)))

        ax1.set_ylabel(r'$c_m$')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax2.set_xlabel(r'$z \: (\mu m)$')
        ax2.set_xticks([-50, -25, 0, 25, 50])
        ax2.set_ylabel(r'$\epsilon_{i,z} \: (\mu m)$')
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$\overline{\sigma_{z}} \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(join(path_results, 'compare_calibration_particle.png'))

    # ---

    if dict_plots['supfigs']['asymmetric_similarity']:
        pass

if __name__ == '__main__':
    # TODO: limit images in dataset (that's on Github) to only those within this z-range

    # A. experimental details
    MAG_EFF = 10.01  # effective magnification (experimentally measured)
    NA_EFF = 0.45  # numerical aperture of objective lens
    MICRONS_PER_PIXEL = 1.6  # conversion ratio from pixels to microns (experimentally measured)
    SIZE_PIXELS = 16  # units: microns (size of the pixels on the CCD sensor)
    NUM_PIXELS = 512
    AREA_PIXELS = NUM_PIXELS ** 2
    NUM_FRAMES_PER_STEP = 3
    TRUE_NUM_PARTICLES_PER_FRAME = 88
    TRUE_NUM_PARTICLES_PER_Z = TRUE_NUM_PARTICLES_PER_FRAME * NUM_FRAMES_PER_STEP

    # dataset alignment
    Z_ZERO_OF_CALIB_ID_FROM_CALIBRATION = 49.6  # the in-focus z-position of calib particle in calib images (microns)
    Z_ZERO_OF_CALIB_ID_FROM_TEST = 68.1  # the in-focus z-position of calib particle in test images (microns)

    # inputs from test
    Z_RANGE = [-50, 55]
    MEASUREMENT_DEPTH = Z_RANGE[1] - Z_RANGE[0]
    NUM_FRAMES_TOTAL = 63
    BASELINE_FRAME = 39
    PADDING = 5  # units: pixels
    IMG_XC, IMG_YC = NUM_PIXELS / 2 + PADDING, NUM_PIXELS / 2 + PADDING

    # filters
    OUT_OF_PLANE_THRESHOLD = 5  # microns
    IN_PLANE_THRESHOLD_PIXELS = 2  # pixels
    IN_PLANE_THRESHOLD = IN_PLANE_THRESHOLD_PIXELS * MICRONS_PER_PIXEL  # units: microns
    MIN_CM = 0.5
    MIN_COUNTS = 1
    MIN_COUNTS_BIN_Z = 20
    MIN_COUNTS_BIN_R = 20
    MIN_COUNTS_BIN_RZ = 5
    MIN_COUNTS_ICP = 5
    Z_TILT_LIMIT = 3.25

    # ---
    # SETUP

    PATH_CWD = os.getcwd()
    PATH_IDPT_COORDS = join(PATH_CWD, 'results', 'test', 'test_test-coords.xlsx')
    PATH_SPCT_COORDS = join(PATH_CWD, 'analyses', 'ref', 'spct_test-coords.xlsx')
    PATH_GDPT_COORDS = join(PATH_CWD, 'analyses', 'ref', 'gdpt_test-coords.xlsx')
    PATH_TRUE_COORDS = join(PATH_CWD, 'analyses', 'ref', 'fiji_true-coords.xlsx')

    PATH_RESULTS = join(PATH_CWD, 'analyses', 'outputs')
    make_dir(path=PATH_RESULTS)
    PATH_FIT_PLANE = join(PATH_RESULTS, 'fit_plane')
    PATH_OUTLIERS = join(PATH_RESULTS, 'outliers')
    PATH_RMSE_Z = join(PATH_RESULTS, 'rmse_z')
    PATH_READ_RMSE_Z = dict({
        'IDPT': join(PATH_RMSE_Z, 'idpt_bin-z_rmse-z.xlsx'),
        'SPCT': join(PATH_RMSE_Z, 'spct_bin-z_rmse-z.xlsx'),
        'GDPT': join(PATH_RMSE_Z, 'gdpt_bin-z_rmse-z.xlsx'),
    })
    PATH_RIGID_TRANSFORMATIONS = join(PATH_RESULTS, 'rigid_transformations')
    PATH_READ_RMSE_XY = dict({
        'IDPT': join(PATH_RIGID_TRANSFORMATIONS, 'dfdz_icp_idpt.xlsx'),
        'SPCT': join(PATH_RIGID_TRANSFORMATIONS, 'dfdz_icp_spct.xlsx'),
        'GDPT': join(PATH_RIGID_TRANSFORMATIONS, 'dfdz_icp_gdpt.xlsx'),
    })
    PATH_READ_ERRORS_XY = dict({
        'IDPT': join(PATH_RIGID_TRANSFORMATIONS, 'dfBB_icp_idpt.xlsx'),
        'SPCT': join(PATH_RIGID_TRANSFORMATIONS, 'dfBB_icp_spct.xlsx'),
        'GDPT': join(PATH_RIGID_TRANSFORMATIONS, 'dfBB_icp_gdpt.xlsx'),
    })
    PATH_FIELD_DEP_EFFECTS = join(PATH_RESULTS, 'field_dependent_effects')
    PATH_PUBFIGS = join(PATH_RESULTS, 'pubfigs')
    PATH_SUPFIGS = join(PATH_RESULTS, 'supfigs')

    DICT_PATHS = {
        'results': PATH_RESULTS,
        'idpt': PATH_IDPT_COORDS,
        'spct': PATH_SPCT_COORDS,
        'gdpt': PATH_GDPT_COORDS,
        'true': PATH_TRUE_COORDS,
        'fit_plane': PATH_FIT_PLANE,
        'outliers': PATH_OUTLIERS,
        'rmse_z': PATH_RMSE_Z,
        'rigid_transformations': PATH_RIGID_TRANSFORMATIONS,
        'field_dependent_effects': PATH_FIELD_DEP_EFFECTS,
        'pubfigs': PATH_PUBFIGS,
        'supfigs': PATH_SUPFIGS,
        'read_rmse_z': PATH_READ_RMSE_Z,
        'read_rmse_xy': PATH_READ_RMSE_XY,
        'read_errors_xy': PATH_READ_ERRORS_XY,
    }

    DICT_INPUTS = {
        'num_pixels': NUM_PIXELS,
        'microns_per_pixel': MICRONS_PER_PIXEL,
        'padding': PADDING,
        'measurement_depth': MEASUREMENT_DEPTH,
        'num_frames_total': NUM_FRAMES_TOTAL,
        'baseline_frame': BASELINE_FRAME,
        'zf_calibration': Z_ZERO_OF_CALIB_ID_FROM_CALIBRATION,
        'zf_test': Z_ZERO_OF_CALIB_ID_FROM_TEST,
        'z_range': Z_RANGE,
        'img_xc': IMG_XC,
        'img_yc': IMG_YC,
        'r0': (IMG_XC, IMG_YC),
        'num_frames_per_step': NUM_FRAMES_PER_STEP,
        'true_num_particles_per_frame': TRUE_NUM_PARTICLES_PER_FRAME,
        'true_num_particles_per_z': TRUE_NUM_PARTICLES_PER_Z,
    }

    DICT_FILTERS = {
        'out_of_plane_threshold': OUT_OF_PLANE_THRESHOLD,
        'in_plane_threshold': IN_PLANE_THRESHOLD,
        'in_plane_threshold_pixels': IN_PLANE_THRESHOLD_PIXELS,
        'min_cm': MIN_CM,
        'min_counts': MIN_COUNTS,
        'min_counts_bin_z': MIN_COUNTS_BIN_Z,
        'min_counts_bin_r': MIN_COUNTS_BIN_R,
        'min_counts_bin_rz': MIN_COUNTS_BIN_RZ,
        'min_counts_icp': MIN_COUNTS_ICP,
        'z_tilt_limit': Z_TILT_LIMIT,
    }

    DICT_PLOTS = {
        'dataset_alignment': True,
        'fit_plane':
            {'z_corr_valid': True, 'z_raw_valid': True, 'z_raw_all': True, 'fit_accuracy': True},
        'local_rmse_z':
            {'bin_z': True, 'bin_r': True, 'bin_r_z': True},
        'fit_rt_accuracy': True,
        'field_dependent_effects':
            {'plot_each_z': True, 'plot_overlay': True, 'plot_shape_change': True, 'plot_mean_z_per_pid': False},
        'pubfigs': {'Figure3': True, 'Figure4': True},
        'supfigs':
            {'Figure3_GDPT': True, 'outliers': True, 'hist_z': True, 'hist_xy': True,
             'rmse_z_by_cmin': False, 'compare_calibration_particle': False, 'asymmetric_similarity': False,}
    }

    DICT_DATA = {
        'dataset_fit_plane': 'aligned',  # 'aligned' == fit plane to raw positions
        'dataset_rigid_transformations': 'corrected',  # 'corrected' == fit rigid transformations to fit-plane output
        'dataset_rmse': 'rigid_transformations',  # 'rigid_transformations' == evaluate after all outliers removed
        'dataset_field_dependent_effects': 'rigid_transformations',
        'dataset_performance': 'placeholder',
        'use_columns': {'rmse_error_z': 'error_rel_plane'},  # 'error_rel_plane' == plane; 'errz' == rigid transforms
    }

    USE_XY_UNITS = 'microns'

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ALIGN DATASETS IN X, Y, Z
    DICT_DATA = align_datasets(dict_data=DICT_DATA,
                               dict_inputs=DICT_INPUTS,
                               dict_paths=DICT_PATHS,
                               dict_plots=DICT_PLOTS,
                               make_xy_units=USE_XY_UNITS,
                               )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # FIT 3D PLANE TO CORRECT FOR STAGE TILT + VARIATIONS IN MICROMETER DISPLACEMENT
    DICT_DATA = fit_plane_analysis(dict_data=DICT_DATA,
                                   dict_inputs=DICT_INPUTS,
                                   dict_filters=DICT_FILTERS,
                                   dict_paths=DICT_PATHS,
                                   dict_plots=DICT_PLOTS,
                                   xy_units=USE_XY_UNITS,
                                   )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # FIT RIGID TRANSFORMATIONS FROM FOCUS
    for mtd in ['idpt', 'spct', 'gdpt']:
        fit_rigid_transformations(method=mtd,
                                  dict_data=DICT_DATA,
                                  dict_inputs=DICT_INPUTS,
                                  dict_filters=DICT_FILTERS,
                                  dict_paths=DICT_PATHS,
                                  dict_plots=DICT_PLOTS,
                                  xy_units=USE_XY_UNITS,
                                  )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # EVALUATE ROOT MEAN SQUARE ERROR POST OUTLIER REMOVAL
    evaluate_root_mean_square_error(dict_data=DICT_DATA,
                                    dict_inputs=DICT_INPUTS,
                                    dict_filters=DICT_FILTERS,
                                    dict_paths=DICT_PATHS,
                                    dict_plots=DICT_PLOTS,
                                    xy_units=USE_XY_UNITS,
                                    )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # FIT PARABOLAS TO EVALUATE FIELD-DEPENDENT EFFECTS
    for mtd in ['spct', 'gdpt']:
        evaluate_field_dependent_effects(method=mtd,
                                         dict_data=DICT_DATA,
                                         dict_inputs=DICT_INPUTS,
                                         dict_paths=DICT_PATHS,
                                         dict_plots=DICT_PLOTS,
                                         )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PLOT FIGURES RELATED TO PUBLICATION
    plot_pubfigs(dict_data=DICT_DATA,
                 dict_inputs=DICT_INPUTS,
                 dict_paths=DICT_PATHS,
                 dict_plots=DICT_PLOTS,
                 )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PLOT FIGURES RELATED TO SUPPLEMENTARY INFORMATION
    for mtd in ['idpt', 'spct', 'gdpt']:
        plot_supfigs(method=mtd,
                     dict_data=DICT_DATA,
                     dict_inputs=DICT_INPUTS,
                     dict_filters=DICT_FILTERS,
                     dict_paths=DICT_PATHS,
                     dict_plots=DICT_PLOTS,
                     xy_units=USE_XY_UNITS,
                     )

    print("completed without errors")
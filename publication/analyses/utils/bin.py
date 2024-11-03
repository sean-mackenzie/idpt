
import numpy as np
import pandas as pd

def bin_generic(df, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby):
    """
    dfm, dfstd = bin.bin_generic(df, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) mean, (2) standard deviation, (3) and counts for a single column
    """
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # returns an identical dataframe but adds a column named "bin"
    if isinstance(bins, int):
        df = bin_by_column(df, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=round_to_decimal)

    elif isinstance(bins, (list, tuple, np.ndarray)):
        df = bin_by_list(df, column_to_bin=column_to_bin, bins=bins, round_to_decimal=round_to_decimal)

    if return_groupby is False:
        return df

    else:
        # groupby.mean()
        dfm = df.groupby('bin').mean()

        # groupby.std()
        dfstd = df.groupby('bin').std()

        # groupby.count()
        if column_to_count is not None:
            dfc = df.groupby('bin').count()
            count_column = 'count_' + column_to_count
            dfc[count_column] = dfc[column_to_count]

            dfm = dfm.join([dfc[[count_column]]])

        dfm = dfm.reset_index()
        dfstd = dfstd.reset_index()

        return dfm, dfstd


def bin_generic_2d(df, columns_to_bin, column_to_count, bins, round_to_decimals, min_num_bin, return_groupby):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) mean, (2) standard deviation, (3) and counts for a single column
    """
    column_to_bin_top_level = columns_to_bin[0]
    column_to_bin_low_level = columns_to_bin[1]

    columns_to_count_low_level = column_to_count

    bins_top_level = bins[0]
    bins_low_level = bins[1]

    round_to_decimals_top_level = round_to_decimals[0]
    round_to_decimals_low_level = round_to_decimals[1]

    # drop NaNs in 'column_to_bin' which cause an Empty DataFrame
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin_top_level, column_to_bin_low_level])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # bin - top level (usually an axial spatial parameter: z)
    if isinstance(bins_top_level, int):
        df = bin_by_column(df,
                           column_to_bin=column_to_bin_top_level,
                           number_of_bins=bins_top_level,
                           round_to_decimal=round_to_decimals_top_level)

    elif isinstance(bins_top_level, (list, tuple, np.ndarray)):
        df = bin_by_list(df,
                         column_to_bin=column_to_bin_top_level,
                         bins=bins_top_level,
                         round_to_decimal=round_to_decimals_top_level)

    df = df.rename(columns={'bin': 'bin_tl'})

    # bin - low level (usually a lateral spatial parameter: x, y, r, dx, percent overlap diameter)
    if isinstance(bins_low_level, (int, float)):
        df = bin_by_column(df,
                           column_to_bin=column_to_bin_low_level,
                           number_of_bins=bins_low_level,
                           round_to_decimal=round_to_decimals_low_level)

    elif isinstance(bins_low_level, (list, tuple, np.ndarray)):
        df = bin_by_list(df,
                         column_to_bin=column_to_bin_low_level,
                         bins=bins_low_level,
                         round_to_decimal=round_to_decimals_low_level)

    df = df.rename(columns={'bin': 'bin_ll'})

    if return_groupby is False:
        return df

    else:

        dfms = []
        dfstds = []

        # for each bin (z)
        for bntl in df.bin_tl.unique():
            # get the dataframe for this bin only
            dfbtl = df[df['bin_tl'] == bntl]
            bins_tl_ll = dfbtl.bin_ll.unique()

            # for each bin (x, y, r)
            dfmll, dfstdll = bin_generic(dfbtl,
                                         column_to_bin_low_level,
                                         column_to_count=columns_to_count_low_level,
                                         bins=bins_tl_ll,
                                         round_to_decimal=round_to_decimals_low_level,
                                         return_groupby=return_groupby,
                                         )

            # filter dfm
            dfmll = dfmll.dropna(subset=columns_to_bin)
            dfmll = dfmll[dfmll['count_' + column_to_count] >= min_num_bin]
            dfms.append(dfmll)

            # re-organize dfstd
            dfstdll['bin_tl'] = bntl
            dfstdll['bin_ll'] = dfstdll['bin']
            dfstdll = dfstdll.dropna(subset=columns_to_bin)
            dfstdll = dfstdll[dfstdll['bin'].isin(dfmll.bin.unique())]
            dfstds.append(dfstdll)

        df_means = pd.concat(dfms, ignore_index=True)
        df_stds = pd.concat(dfstds, ignore_index=True)

        df_means = df_means.drop(columns=['bin'])
        df_stds = df_stds.drop(columns=['bin'])

        return df_means, df_stds


def bin_by_column(df, column_to_bin='z_true', number_of_bins=25, round_to_decimal=2):
    """
    Creates a new column "bin" of which maps column_to_bin to equi-spaced bins. Note, that this does not change the
    original dataframe in any way. It only adds a new column to enable grouping.

    # rename column if mis-named
    if 'true_z' in df.columns:
        df = df.rename(columns={"true_z": "z_true"})

    """

    # turn off pandas warning to try using .loc[row_indexer, col_indexer] = value
    pd.options.mode.chained_assignment = None  # default='warn'

    # round the column_to_bin to integer for easier mapping
    temp_column = 'temp_' + column_to_bin
    df[temp_column] = np.round(df[column_to_bin].values, round_to_decimal)

    # copy the column_to_bin to 'mapped' for mapping
    df.loc[:, 'bin'] = df.loc[:, temp_column]

    # get unique values
    unique_vals = df[temp_column].astype(float).unique()

    # drop temp column
    df = df.drop(columns=[temp_column])

    # calculate the equi-width stepsize
    stepsize = (np.max(unique_vals) - np.min(unique_vals)) / number_of_bins

    # re-interpolate the space
    new_vals = np.linspace(np.min(unique_vals) + stepsize / 2, np.max(unique_vals) - stepsize / 2, number_of_bins)

    # round to reasonable decimal place
    new_vals = np.around(new_vals, decimals=round_to_decimal)

    # create the mapping list
    mappping = map_lists_a_to_b(unique_vals, new_vals)

    # create the mapping dictionary
    mapping_dict = {unique_vals[i]: mappping[i] for i in range(len(unique_vals))}

    # insert the mapped values
    df.loc[:, 'bin'] = df.loc[:, 'bin'].map(mapping_dict)

    return df


def bin_by_list(df, column_to_bin, bins, round_to_decimal=0):
    """
    Creates a new column "bin" of which maps column_to_bin to the specified values in bins [type: list, ndarray, tupe].
    Note, that this does not change the original dataframe in any way. It only adds a new column to enable grouping.
    """

    # rename column if mis-named
    if 'true_z' in df.columns:
        df = df.rename(columns={"true_z": "z_true"})

    # round the column_to_bin to integer for easier mapping
    df = df.round({'z_true': 4})

    if column_to_bin in ['x', 'y']:
        df = df.round({'x': round_to_decimal, 'y': round_to_decimal})

    # copy the column_to_bin to 'mapped' for mapping
    df['bin'] = df[column_to_bin].copy()

    # get unique values and round to reasonable decimal place
    unique_vals = df[column_to_bin].unique()

    # create the mapping list
    mappping = map_lists_a_to_b(unique_vals, bins)

    # create the mapping dictionary
    mapping_dict = {unique_vals[i]: mappping[i] for i in range(len(unique_vals))}

    # insert the mapped values
    df['bin'] = df['bin'].map(mapping_dict)

    return df


def map_lists_a_to_b(a, b):
    """
    returns a new list which is a mapping of a onto b.
    """
    mapped_vals = []
    for val in a:
        # get the distance of val from every element in our list to map to
        dist = np.abs(np.ones_like(b) * val - b)

        # append the value of minimum distance to our mapping list
        mapped_vals.append(b[np.argmin(dist)])

    return mapped_vals


def sample_array_at_intervals(arr_to_sample, bins, bin_width, nearest_sample_to_bin=True):
    """

    :param arr_to_sample:
    :param bins:
    :param bin_width:
    :param nearest_sample_to_bin:
    :return:
    """
    # ensure numpy array
    if isinstance(arr_to_sample, list):
        arr_to_sample = np.array(arr_to_sample)

    # get unique values
    arr_to_sample = np.unique(arr_to_sample)

    arr_sampled_at_bins = []
    for i in range(len(bins) - 1):

        arr_values_at_bin = arr_to_sample[(arr_to_sample > bins[i] - bin_width) * (arr_to_sample < bins[i] + bin_width)]

        if len(arr_values_at_bin) == 0:
            continue
        elif nearest_sample_to_bin is True:
            arr_value = arr_values_at_bin[np.argmin(np.abs(arr_values_at_bin - bins[i]))]
        else:
            arr_value = arr_values_at_bin[0]

        arr_sampled_at_bins.append(arr_value)

    return arr_sampled_at_bins
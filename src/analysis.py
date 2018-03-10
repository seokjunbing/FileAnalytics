"""
What are some things that we would want to possibly do?
Modeling:
* Correlation between block size and file lifetime
*

FILETYPE
EXTENSION
BLOCKS
BIRTHTIME
ALIVE_FOR_PERIODS
IS_DEAD
"""

import time
from typing import List
from os import path

import matplotlib.pyplot as plt
import pandas as pd

START_TIME = time.time()
DATASET = 'jun'  # 'juan' or 'jun'
FILENAME = '../data/test_%s.tsv' % DATASET

# we will split our dataset into smaller datasets to generate descriptive statistics
BUCKET_BOUNDARIES = [0, 1, 10, 100, 500, 1000, 10000, 100000]
BUCKET_LABELS = ['0', '1-10', '11-100', '101-500', '501-1000', '1001-10000', '10001-100000']

DATAFRAME_COLUMNS = ['filetype', 'extension', 'blocks', 'birthtime', 'alive_for_periods', 'is_dead']
EXTENSION_COL = DATAFRAME_COLUMNS[1]
BIRTHTIME_COL = DATAFRAME_COLUMNS[3]
ALIVE_FOR_COL = DATAFRAME_COLUMNS[4]
IS_DEAD_COL = DATAFRAME_COLUMNS[5]

# not included in the list because this one is not present in the original .tsv and the primary purpose of the list is
# to name the columns in that .tsv. In other words, these are columns we generate for later
# processing
BLOCKSIZE_CATEGORY_COL = 'blocksize_category'
TOTAL_LIFETIME_COL = 'lifetime'
TOTAL_LIFETIME_DAYS_COL = 'lifetime_days'
ALIVE_OBSERVED_MINUTES_COL = 'alive_for_minutes'

PREPROCESSED_SUFFIX = '_preprocessed.tsv'


# def create_block_vs_lifetime_table(df):
# df = pd.read_csv('../data/sample.tsv', sep='\t')
# df = pd.read_csv(FILENAME, sep='\t', header=None, names=DF_COLUMNS)
# print(df)
# cut_buckets = {0: '0', 1: '1', 10: '10', 100: '100', 500: '500', 1000: '1000', inf: 'inf'}
# cut_buckets = {cut_buckets[i]: '%d-%d' % (cut_buckets[i - 1], cut_buckets[i]) for i in range(1, len(cut_buckets))}
# cuts: pd.Series = pd.cut(
#     df['blocks'],
#     bins=BUCKET_BOUNDARIES,
#     labels=BUCKET_LABELS
# )
# df['blocksize_category'] = cuts  # add a new column, with the blocksize category
# return df, cuts.value_counts()

# table = []  # the entries will go here
# for entry in table:
#     filename, extension, is_alive, birth_time, periods_alive = entry
# if file_in_range(is_alive, birth_time, periods_alive):
#     pass

def get_blocksize_dataframes(df: pd.DataFrame, state: str, column='blocksize_category'):
    """
    :param df: data frame
    :param state: read or process; first read or subsequent processes
    :param column:
    :return:
    """
    df_blocksize_list = []

    if state == state_read:
        for i in range(len(BUCKET_LABELS)):
            df_filtered = df[df['%s' % BLOCKSIZE_CATEGORY_COL] == BUCKET_LABELS[i]]
            df_blocksize_list.append(df_filtered)
            filename = '../data/block%s.tsv' % BUCKET_LABELS[i]
            df_filtered.to_csv(filename, sep='\t')

    elif state == state_process:
        for i in range(len(BUCKET_LABELS)):
            filename = '../data/block%s.tsv' % BUCKET_LABELS[i]
            print('about to read %s' % filename)
            df_filtered = pd.read_csv(filename, sep='\t')
            df_blocksize_list.append(df_filtered)

    return df_blocksize_list


def preprocess_data(end_time):
    index_label = 'num'
    is_dead_index = 5

    df = pd.read_csv(FILENAME, sep='\t', header=None, names=DATAFRAME_COLUMNS, dtype={is_dead_index: bool})

    # add a column with the category (by blocksize)
    cuts = pd.cut(df['blocks'], bins=BUCKET_BOUNDARIES, labels=BUCKET_LABELS)
    df['blocksize_category'] = cuts

    # add a column with the lifetime of the file
    df[TOTAL_LIFETIME_COL] = df[BIRTHTIME_COL]

    # using loc because assigning a new column to the dataframe as we did creates a copy of the data.
    df[TOTAL_LIFETIME_COL] = df.loc[0:, TOTAL_LIFETIME_COL].apply(lambda birthtime: end_time - birthtime)
    df[TOTAL_LIFETIME_DAYS_COL] = df[TOTAL_LIFETIME_COL].apply(lambda lifetime: to_days(lifetime))

    # convert periods monitored to minutes: 24 hours divided by how many periods were seen by the script
    minutes_per_period = (24 * 60) / df[ALIVE_FOR_COL].max()
    df[ALIVE_OBSERVED_MINUTES_COL] = df[ALIVE_FOR_COL].apply(lambda periods_alive: periods_alive * minutes_per_period)

    df.to_csv('%s%s' % (FILENAME, PREPROCESSED_SUFFIX), sep='\t', index_label=index_label)
    return df


def get_dataframe(state, filename, end_time=None):
    """
    First read of the raw data.
        Do light processing to add a blocksize category (1-10, 11-100, etc.)
    Subsequent runs
        Read the tables from the .tsv file in disk
    """
    index_label = 'num'

    # headers already formatted and included
    df = pd.read_csv(filename, sep='\t')
    # Some files have characters that are written as newlines in the .csv file.
    # We originally thought this would happen only at the end of filenames and are cleaning those characters
    # there, but it turns out this also happens at the beginning of filenames. Data collection makes it unfeasible
    # to fix our data collection at this stage, so this is a patch.
    # The weird characters make the pandas library think the row index is not numeric because the newline
    # results in one of the lines start with a string. This dirty solution consists of ignoring the lines
    # that do not start with an int. The previous line containing the filename that had a new line in
    # the middle becomes unusable (the columns become NaN after the numeric conversion, but this is fine for now,
    # since we are only looking at statistics for now and the NaN get ignored for those.

    # df[IS_DEAD_COLUMN] = df[IS_DEAD_COLUMN].astype(bool)
    # print(df[IS_DEAD_COLUMN].dtype)
    df2 = df[pd.to_numeric(df[index_label], errors='coerce').notnull()]
    df2.set_index(index_label)
    df2 = df2.dropna(subset=['birthtime', 'alive_for_periods'])
    df2['alive_for_periods'] = df2['alive_for_periods'].apply(lambda x: pd.to_numeric(x, errors='coerce'))

    return df2


def mean(df: pd.DataFrame, column):
    return df[column].mean()


def mean_lifetime(df: pd.DataFrame):
    return mean(df, ALIVE_FOR_COL)


def stddev(df: pd.DataFrame, column):
    return df[column].std()


state_read = 'read'
state_process = 'process'
STATE_ALL = {state_read, state_process}
STATE_CURRENT = state_process


def blocksize_stats(df, label, verbose_mode):
    print('--BLOCKSIZE STATS--')
    print('-%s-' % label)

    df_new = pd.DataFrame()
    df_new['category'] = BUCKET_LABELS
    mean_lifetime_list = []
    std_lifetime_list = []
    mean_alive_list = []
    std_alive_list = []
    count_list = []

    for i in range(len(BUCKET_LABELS)):
        df_filtered = df[df[BLOCKSIZE_CATEGORY_COL] == BUCKET_LABELS[i]]
        mean_alive_list.append(mean(df_filtered, ALIVE_FOR_COL))
        std_alive_list.append(stddev(df_filtered, ALIVE_FOR_COL))
        mean_lifetime_list.append(to_days(mean(df_filtered, TOTAL_LIFETIME_COL)))
        std_lifetime_list.append(to_days(stddev(df_filtered, TOTAL_LIFETIME_COL)))
        count_list.append(len(df_filtered))

        if verbose_mode:
            print('Block size: %s' % BUCKET_LABELS[i])
            print('count: %d' % len(df_filtered))

            print('mean for alive periods monitored (periods): ' + str(mean(df_filtered, ALIVE_FOR_COL)))
            print('stddev for alive periods monitored (periods): ' + str(stddev(df_filtered, ALIVE_FOR_COL)))
            print(
                '\tmin: %s, max: %s' % (
                    to_days(df_filtered[ALIVE_FOR_COL].min()), to_days(df_filtered[ALIVE_FOR_COL].max())
                )
            )

            print('mean for lifetime approximation (days): ' + str(to_days(mean(df_filtered, TOTAL_LIFETIME_COL))))
            print(
                'stddev for lifetime approximation (days): ' + str(to_days(stddev(df_filtered, TOTAL_LIFETIME_COL))))
            print(
                '\tmin: %s, max: %s' % (
                    to_days(df_filtered[TOTAL_LIFETIME_COL].min()), to_days(df_filtered[TOTAL_LIFETIME_COL].max())
                )
            )
            print('\n')

    df_new['alive_mean_periods'] = mean_alive_list
    df_new['alive_std_periods'] = std_alive_list
    df_new['lifetime_mean_days'] = mean_lifetime_list
    df_new['lifetime_std_days'] = std_lifetime_list
    df_new['count'] = count_list
    print('%s\n' % df_new)
    df_new.to_csv('../data/blocksize_stats_%s_%s.csv' % (DATASET, label), index=False, )


def to_days(seconds):
    """
    Seconds -> days
    """
    return seconds / 3600 / 24


def filetype_stats(df: pd.DataFrame, label: str):
    """
    Cluster files by type and print age stats about each
    """
    f_tmp = ['tmp']
    f_config = ['plist']
    f_data = ['dat', 'xml', 'json', 'bson', 'zip']
    f_img = ['jpg', 'jpeg', 'raw', 'tiff', 'png', 'svg']
    f_doc = ['txt', 'doc', 'docx', 'xls', 'xlsx', 'pdf', 'md']
    f_dev = ['c', 'cpp', 'h', 'py', 'kt', 'java', 'ipynb', 'html', 'css', 'js']
    f_devtmp = ['out', 'pyc']
    f_db = ['sqlite', 'sql']

    filetypes_all = [f_tmp, f_data, f_config, f_img, f_doc, f_dev, f_devtmp, f_db]
    filetype_names = [
        'temporary', 'data', 'config', 'image', 'document', 'developer', 'temporary developer', 'database'
    ]
    for i in range(len(filetypes_all)):
        extension_group = filetypes_all[i]
        # df_extension = df[df[EXTENSION_COLUMN].isin(extension_group)]
        # for extension_group in filetypes_all:
        df_extension = df[df[EXTENSION_COL].isin(extension_group)]
        print('%s files: %s' % (filetype_names[i], str(extension_group)))
        print('count: %d' % len(df_extension))
        t = [type(x) for x in df_extension[ALIVE_FOR_COL]]
        print('stddev for lifetime: ' + str(stddev(df_extension, ALIVE_FOR_COL)))
        print('mean for lifetime: ' + str(mean(df_extension, ALIVE_FOR_COL)))
        print('\n')


def filetype_histogram(df: pd.DataFrame, mode: str, label: str):
    """
    Instead of clustering the files by type, get the top 20 file types
    by count and do stats on those
    """
    assert mode in ['minutes', 'days']

    print('--HISTOGRAM--')
    print('-%s-' % label)
    dist: pd.Series = df[EXTENSION_COL].value_counts(dropna=False).head(10)
    ext_list, count_list = zip(*[(ext, count) for ext, count in dist.iteritems()])
    mean_list = []
    stddev_list = []

    for ext, count in dist.iteritems():
        # str(ext) so that nan (which is a float) gets converted to a string, just like all the other values in the column
        if str(ext).lower() == 'nan':
            df_filtered = df[df[EXTENSION_COL].isna()]
        else:
            df_filtered = df[df[EXTENSION_COL] == ext]

        selection_col = TOTAL_LIFETIME_DAYS_COL if mode == 'days' else ALIVE_OBSERVED_MINUTES_COL

        mean_val = df_filtered[selection_col].mean()
        stddev_val = df_filtered[selection_col].std()
        mean_list.append(mean_val)
        stddev_list.append(stddev_val)

    # if mode == 'days':
    #     for ext, count in dist.iteritems():
    #         # ext_list.append(ext)
    #         # count_list.append(count)
    #         df_filtered = df[df[EXTENSION_COL] == ext]
    #         mean_val = df_filtered[TOTAL_LIFETIME_DAYS_COL].mean()
    #         stddev_val = df_filtered[TOTAL_LIFETIME_DAYS_COL].std()
    #         mean_list.append(mean_val)
    #         stddev_list.append(stddev_val)
    #
    # elif mode == 'minutes':
    #     for ext, count in dist.iteritems():
    #         print(ext)
    #         # ext_list.append(ext)
    #         # count_list.append(count)
    #         df_filtered = df[df[EXTENSION_COL] == ext]
    #         mean_val = df_filtered[ALIVE_OBSERVED_MINUTES_COL].mean()
    #         stddev_val = df_filtered[ALIVE_OBSERVED_MINUTES_COL].std()
    #         mean_list.append(mean_val)
    #         stddev_list.append(stddev_val)

    df_stats = pd.DataFrame()
    df_stats['extension'] = ext_list
    df_stats['count'] = count_list
    df_stats['age_mean_%s' % mode] = mean_list
    df_stats['age_stddev_%s' % mode] = stddev_list

    print(df_stats)
    print('\n')


if __name__ == '__main__':

    verbose = False

    start_time = 1520451300  # 03/07/2018 2:30 PM
    end_time = 1520550000  # 03/08/2018 06:00 PM
    years_twoandahalf = int(365 * 2.5)  # for filtering old files

    preprocessed_fname = FILENAME + PREPROCESSED_SUFFIX
    if not path.isfile(preprocessed_fname):
        preprocess_data(end_time)

    df_all = get_dataframe(state_process, preprocessed_fname)
    # exclude files older than two and a half years
    df_all = df_all[df_all[TOTAL_LIFETIME_DAYS_COL] <= years_twoandahalf]
    print('There are %d files in this dataset' % len(df_all))

    # files deleted during the monitoring period
    df_deleted = df_all[df_all[IS_DEAD_COL] == True]
    # files created and deleted during the monitoring period
    df_created_and_deleted = df_deleted[df_deleted[BIRTHTIME_COL] > start_time]
    # the difference of the two previous sets, i.e., files created _before_ the monitoring period
    # that died _during_ the monitoring period
    df_not_created_and_deleted = df_deleted[~df_deleted.index.isin(df_created_and_deleted.index)]

    # matplotlib is giving us a lot of trouble on standalone python scripts.
    # See the attached notebook graphs.py.ipynb for our graph creation process
    # df_created_and_deleted.hist(column='alive_for_periods')
    # plt.plot()

    print(len(df_not_created_and_deleted))

    # blocksize
    blocksize_stats(df_all, 'ALL FILES', verbose)
    blocksize_stats(df_deleted, 'FILES CREATED AND DELETED WHILE BEING MONITORED', verbose)
    blocksize_stats(df_not_created_and_deleted, 'FILES CREATED BEFORE MONITORING, DELETED DURING MONITORING', verbose)
    # # age stats per filetype group (image files, developer files, temporary files, etc.)
    filetype_stats(df_all, 'ALL FILES')
    filetype_stats(df_created_and_deleted, 'FILES CREATED AND DELETED WHILE BEING MONITORED')
    filetype_stats(df_not_created_and_deleted, 'FILES CREATED BEFORE MONITORING, DELETED DURING MONITORING')
    # most common filetypes
    filetype_histogram(df_all, 'days', 'ALL FILES')
    filetype_histogram(df_not_created_and_deleted, 'days', 'FILES CREATED BEFORE MONITORING, DELETED DURING MONITORING')
    filetype_histogram(df_created_and_deleted, 'minutes', 'FILES CREATED AND DELETED WHILE BEING MONITORED')

    print('Count of deleted files: %d' % len(df_deleted))
    print('Count of created and deleted files: %d' % len(df_created_and_deleted))
    print('Count of pre-existing files that were deleted: %d' % len(df_not_created_and_deleted))

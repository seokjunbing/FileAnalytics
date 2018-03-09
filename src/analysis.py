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
import csv
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

START_TIME = time.time()
# FILENAME = '../data/test_out_copy.tsv'
# FILENAME = '../data/test_out_FINAL.tsv'
FILENAME = '../data/test_jun.tsv'

BUCKET_BOUNDARIES = [0, 1, 10, 100, 500, 1000, 10000, 100000]
BUCKET_LABELS = ['%s-%s' % (BUCKET_BOUNDARIES[i - 1], BUCKET_BOUNDARIES[i]) for i in range(1, len(BUCKET_BOUNDARIES))]

DF_COLUMNS = ['filetype', 'extension', 'blocks', 'birthtime', 'alive_for_periods', 'is_dead']
EXTENSION_COLUMN = DF_COLUMNS[1]
BIRTHTIME_COLUMN = DF_COLUMNS[3]
LIFETIME_COLUMN = DF_COLUMNS[4]
IS_DEAD_COLUMN = DF_COLUMNS[5]
# not included in the list because this one is not present in the original .tsv and the primary purpose of the list is
# to name the columns in that .tsv
DF_BLOCKSIZE_CATEGORY = 'blocksize_category'


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


def file_in_range(is_alive, birth_time, periods_alive):
    return birth_time > START_TIME and not is_alive


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
            df_filtered = df[df['%s' % DF_BLOCKSIZE_CATEGORY] == BUCKET_LABELS[i]]
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


def get_dataframe(state):
    """
    First read of the raw data.
        Do light processing to add a blocksize category (1-10, 11-100, etc.)
    Subsequent runs
        Read the tables from the .tsv file in disk
    """
    index_label = 'num'
    is_dead_index = 5

    if state == state_read:
        # initial creation of the table does not contain column names. The pandas library works a lot better
        # with column names, so add them and then save the file to disk
        df = pd.read_csv(FILENAME, sep='\t', header=None, names=DF_COLUMNS, dtype={is_dead_index: bool})

        # TEST
        # df.index = df[pd.to_numeric(df.index, errors='coerce').notnull()]
        # TEST

        # Create one table for each blocksize category, i.e., separate the files that fall in the 1-10 blocks
        # category in a separate table and so on. Helps when calculating descriptive statistics
        cuts: pd.Series = pd.cut(
            df['blocks'],
            bins=BUCKET_BOUNDARIES,
            labels=BUCKET_LABELS
        )
        df['blocksize_category'] = cuts  # add a new column with the blocksize category of each file
        df.to_csv('%s_processed.tsv' % FILENAME, sep='\t', index_label=index_label)  # save to disk
        return df
    elif state_process == state_process:
        # headers already formatted and included
        df = pd.read_csv('%s_processed.tsv' % FILENAME, sep='\t')
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

        # print(df2[IS_DEAD_COLUMN].dtype)
        # df2.to_csv('../data/test_test.tsv', sep='\t', index=0)
        # print(df2[IS_DEAD_COLUMN].dtype)
        return df2
        # return df2


def mean(df: pd.DataFrame, column):
    return df[column].mean()


def mean_lifetime(df: pd.DataFrame):
    return mean(df, LIFETIME_COLUMN)


def stddev(df: pd.DataFrame, column):
    return df[column].std()


state_read = 'read'
state_process = 'process'
STATE_ALL = {state_read, state_process}
STATE_CURRENT = state_process


def blocksize_stats(df):
    df_blocksize_list: List[pd.DataFrame] = get_blocksize_dataframes(df, state=STATE_CURRENT)

    for i in range(len(df_blocksize_list)):
        print('Block size: %s' % BUCKET_LABELS[i])
        df_filtered = df_blocksize_list[i]
        print('count: %d' % len(df_filtered))
        print('stddev for lifetime: ' + str(stddev(df_filtered, LIFETIME_COLUMN)))
        print('mean for lifetime: ' + str(mean(df_filtered, LIFETIME_COLUMN)))
        print('\n')


def filetype_stats(df):
    """
    Cluster files by type
    :param df:
    :return:
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
        df_extension = df[df[EXTENSION_COLUMN].isin(extension_group)]
        print('%s files: %s' % (filetype_names[i], str(extension_group)))
        print('count: %d' % len(df_extension))
        t = [type(x) for x in df_extension[LIFETIME_COLUMN]]
        print('stddev for lifetime: ' + str(stddev(df_extension, LIFETIME_COLUMN)))
        print('mean for lifetime: ' + str(mean(df_extension, LIFETIME_COLUMN)))
        print('\n')


def filetype_histogram(df: pd.DataFrame):
    # print(group)
    # df_grouped = df.groupby(EXTENSION_COLUMN).count()
    # print(df_grouped)
    print('--HISTOGRAM--')
    print('COUNT: %s' % len(df))
    print(len(df))
    dist = df[EXTENSION_COLUMN].value_counts(dropna=False).head(20)
    print(dist)
    print('\n')
    plt.hist(dist)
    plt.show()


if __name__ == '__main__':
    start_time = 1520451300

    if STATE_CURRENT == state_read:
        df_all_files = get_dataframe(state_read)
        # df_blocksize_list = get_blocksize_dataframes(df_all_files, state=STATE_CURRENT)
        # for i in range(len(df_blocksize_list)):
        #     print(i)
        #     df_filtered = df_blocksize_list[i]
        #     print('stddev for lifetime: ' + str(stddev(df_filtered, LIFETIME_COLUMN)))
        #     print('mean for lifetime: ' + str(mean(df_filtered, LIFETIME_COLUMN)))


    elif STATE_CURRENT == state_process:
        df_all_files = get_dataframe(state_process)
        df_dead_files = df_all_files[df_all_files[IS_DEAD_COLUMN] == True]
        df_alive_and_dead_files = df_dead_files[df_dead_files[BIRTHTIME_COLUMN] > start_time]

        # blocksize
        blocksize_stats(df_all_files)
        blocksize_stats(df_dead_files)
        # and filetype
        filetype_stats(df_all_files)
        filetype_stats(df_dead_files)
        # most common filetypes
        filetype_histogram(df_all_files)
        filetype_histogram(df_dead_files)
        print('ALIVE AND DEAD')
        print(df_dead_files[BIRTHTIME_COLUMN].dtype)
        filetype_histogram(df_alive_and_dead_files)

        print('Count of dead files: %d' % len(df_dead_files))
        print('Count of alive and dead files: %d' % len(df_alive_and_dead_files))

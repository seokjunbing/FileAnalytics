import os
import csv
import time


def get_all_files(path: str):
    flist = []

    counter = 0

    for root, subdirs, files in os.walk(path):

        for file in os.listdir(root):

            file_path = os.path.join(root, file)

            if os.path.isdir(file_path):
                pass
            else:
                if counter % 10000 == 0:
                    print("scanning: %d\n" % counter)

                try:
                    s = os.stat(file_path)

                except PermissionError:
                    pass
                except FileNotFoundError:
                    pass
                except Exception:
                    pass

                ext = file.split('.')

                if len(ext) < 2:  # catch file names without an extension
                    ext = ''
                elif ext[0] == '':  # catch file names that start with a dot
                    ext = ''
                else:
                    ext = ext[-1]

                # file_path, extension, block size, birthtime, lifetime, death
                flist.append((file_path.strip().rstrip(), ext, str(s.st_blocks), str(s.st_birthtime), '0', '0'))
                # print(flist[-1])

                counter += 1

    return flist


def write_list_to_csv(file_list: list, out_dir: str, file_name: str):
    counter = 0
    print("about to write list to file")
    with open(out_dir + file_name, "w") as fout:
        fwriter = csv.writer(fout, delimiter='\t', quoting=csv.QUOTE_ALL)
        for f in file_list:
            if counter % 10000 == 0:
                print("writing: %d\n" % counter)
            fwriter.writerow(f);
            counter += 1


def write_dict_to_csv(file_dict: dict, out_dir: str, file_name: str):
    counter = 0
    print("about to write dict to file")
    with open(out_dir + file_name, "w") as fout:
        fwriter = csv.writer(fout, delimiter='\t', quoting=csv.QUOTE_ALL)
        for v in file_dict.values():
            if counter % 10000 == 0:
                print("writing: %d\n" % counter)
            fwriter.writerow(v);
            counter += 1


def read_from_csv(in_dir: str, file_name: str):
    counter = 0
    flist = []
    with open(in_dir + file_name, "r") as fin:
        freader = csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_ALL)
        for row in freader:
            if counter % 10000 == 0:
                print("reading: %d\n" % counter)

            # file_path = row[0]
            # ext = row[1]
            # blocks = row[2]
            # birthtime = int(row[3])
            # lifetime = int(row[4])
            # death = int(row[5])

            flist.append(row)

            counter += 1
    return flist


def get_path_birth_pair(dat: list):
    path_ind = 0
    birth_ind = 3
    return dat[path_ind], dat[birth_ind]


def get_updated_file_data(curr_files: list, prev_files: list):
    assert (len(curr_files[0]) == len(prev_files[0]))

    dead_ind = 5
    life_ind = 4

    prev_files_dict = dict()
    for f in prev_files:
        prev_files_dict[get_path_birth_pair(f)] = f

    curr_files_dict = dict()
    for f in curr_files:
        curr_files_dict[get_path_birth_pair(f)] = f

    # find dead files (i.e. prev files that aren't in curr files)
    # and update them as being dead
    # update alive files by adding +1 to their lifetime
    for f in prev_files:
        if get_path_birth_pair(f) in curr_files_dict:  # files here are still alive
            # increase their lifetime
            prev_files_dict[get_path_birth_pair(f)][life_ind] = \
                str(int(prev_files_dict[get_path_birth_pair(f)][life_ind]) + 1)
        else:  # files here are dead
            # mark dead files as dead
            prev_files_dict[get_path_birth_pair(f)][dead_ind] = '1'

    # find new file (i,e, curr files that aren't in prev files)
    # and add to prev files dict
    for f in curr_files:
        if get_path_birth_pair(f) in prev_files_dict:  # files here are not new
            # intentionally empty
            pass
        else:  # files here are new
            prev_files_dict[get_path_birth_pair(f)] = f

    return prev_files_dict


if __name__ == '__main__':
    # UPDATE_FILE_DATA = True
    #
    # WRITE_FILE = True
    #
    # scan_path = "/Users/jun"
    #
    # output_dir = "/Users/jun/PycharmProjects/258"
    # ofile = "/test_out.tsv"
    #
    # input_dir = output_dir
    # ifile = ofile
    #
    # if UPDATE_FILE_DATA:
    #     curr_files = get_all_files(scan_path)
    #     # print(curr_files)
    #     files_from_csv = read_from_csv(input_dir, ifile)
    #     # print(files_from_csv)
    #     updated_dict = get_updated_file_data(curr_files, files_from_csv)
    #     # print(updated_dict)
    #     if WRITE_FILE:
    #         write_dict_to_csv(updated_dict, output_dir, ofile)
    #
    # else:
    #     curr_files = get_all_files(scan_path)
    #
    #     if WRITE_FILE:
    #         write_list_to_csv(curr_files, output_dir, ofile)

    # continuously run for....

    scan_path = "/Users/juantorres"
    output_dir = "/Users/juantorres/Developer/cs258/src/FileMonitor/data"
    ofile = "/test_out.tsv"
    input_dir = output_dir
    ifile = ofile

    # initial scan
    # curr_files_dict = get_all_files(scan_path)
    # write_list_to_csv(curr_files, output_dir, ofile)

    # updates
    count = 0
    while True:
        print("iter: %d" % count)
        curr_files = get_all_files(scan_path)
        files_from_csv = read_from_csv(input_dir, ifile)
        updated_dict = get_updated_file_data(curr_files, files_from_csv)
        write_dict_to_csv(updated_dict, output_dir, ofile)
        time.sleep(60)
        count += 1

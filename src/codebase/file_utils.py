import sys
import tarfile
import os
import pickle
import shutil
import datetime
import numpy as np

def save_obj(obj, name, dir ):
    with open(dir+ name + '.p', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, dir ):
    with open(dir+ name + '.p', 'rb') as f:
        return pickle.load(f)

def make_folder(task_handle):
    nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # ISO 8601 format
    log_dir = "./log/%s_%s/"%(nowstr, task_handle)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def path_backslash(log_dir):
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir + "/"
    return log_dir

def make_tarfile(source_dir, output_filename,):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def delete_folder_contents(folder):
    """
    Delete contents of folder.
    """
    if 'log' not in folder.split('/'):
        print("Folder should start with `log`")
        return 0
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def extract_tarfile(source_dir, output_dir):
    """
    Plot posterior sampple histograms and traces
    Inputs
    ============
    - source_dir: path to tar file (ends in ".tar.gz")
    - output_dir: name of extract_to file, together
        with the path. It has to end in "/"
        E.g. /foo/bar/
    Output
    ============
    -  plot figure (optionally saved in addition
    to specified directory)
    """
    if output_dir[-1] != "/":
        print("Needs a final / character")
        output_dir = output_dir+ "/"
    tar = tarfile.open(source_dir)
    tar.extractall(output_dir)
    tar.close()





#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
~Gros
'''


from os import listdir, path
from math import floor
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys
import csv
import pickle

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from emotion_predictor.config import PATH_DATA
from emotion_predictor.config import PATH_PICTURES
from emotion_predictor.config import PICKLED_DATA_RESTING
from emotion_predictor.config import PICKLED_DATA_EMOTIONIZED
from emotion_predictor.config import PICKLED_DATA_PICTURES

TOOLS = {'BITalino': {'BPM.csv':'BVP', 'GSR.csv':'GSR'}}
FREQ = 128

MIN_RESTING_TIME = 5  # in seconds, min time before first picture to count as valid experiment

DO_PLOTTING = 1
DO_LOGS = 1
PREPROCESS_ONLY = ['B357']  # empty list to plot all


def filter_experiments(all_experiments_path):
    '''
    The script will preprocess only dirs(experiments/persons) returned by this function
    Returns: list of paths to experiments

    As for now we use experiments which used BITalino and contains both signals
    '''
    experiments = []
    for one_experiment in listdir(all_experiments_path):
        one_experiment_path = path.join(all_experiments_path, one_experiment)

        if not path.isdir(one_experiment_path):
            continue

        files_in_one_experiment = listdir(one_experiment_path)
        is_experiment_ok = True

        # check if all required tools were used in this experiment
        tool_id = 0
        while tool_id < len(TOOLS) and is_experiment_ok:
            tool = list(TOOLS.keys())[tool_id]
            tool_id += 1
            if tool not in files_in_one_experiment:
                is_experiment_ok = False
                break

            # check if all tools contains required data files (signals)
            files_in_one_tool = listdir(path.join(one_experiment_path, tool))
            for required_data_file in TOOLS[tool]:
                if required_data_file not in files_in_one_tool:
                    is_experiment_ok = False
                    break

        # check if file with pictures exists:
        if len(glob(path.join(one_experiment_path, '*_timestamp.csv'))) != 1:
            is_experiment_ok = False

        if is_experiment_ok:
            experiments.append(one_experiment_path)

    print('Found {} valid experiments'.format(len(experiments)))
    return experiments


def micro_siemens_to_ohm(value):
        return (10**6)/value


def is_trail_picture(picture_name):
    '''
    Skip measurements for pictures for which this func returns True
    In other words, skip "trail" pictures
    '''
    return 'trail' in picture_name.lower()


def normalize_picture_name(picture_name):
    '''
    Rename "pictures\\day1\\Landscapes_120_v.jpg" -> "Landscapes_120_v"
           "pictures\\trail\\5.jpg" -> "trail5.jpg"
    '''
    if is_trail_picture(picture_name):
        return ''.join(picture_name.split('\\')[-2:]).rsplit('.', 1)[0]    
    return picture_name.split('\\')[-1].rsplit('.', 1)[0]


def preprocess_one_experiment(one_experiment_path, do_plotting=False, do_logs=False):
    '''
    Preprocess one experiments
    Args:
        one_experiment_path(str) - path to directory, should include TOOLS dirs
                                        and in each TOOLS dir there should be
                                        correctly named files with signals
        do_plotting(Bool) - guess what
        do_logs(Bool) - same

    Returns:
        tuple(emotionized, resting)
            emotionized: dict[picture_names][signal] = [signal_value, signal_value2,...]
            resting: dict[signal] = [signal_value, signal_value2,...]
    '''
    def log(x):
        if do_logs:
            print(x)

    print('Preprocessing {}'.format(one_experiment_path))

    # read csv with pictures display times
    pictures_path = glob(path.join(one_experiment_path, '*_timestamp.csv'))[0]
    pictures = pd.read_csv(pictures_path, header=None, names=['picture', 'timestamp'], index_col=1, parse_dates=True)

    pictures.picture = pictures.picture.apply(normalize_picture_name)
    pictures_start_time = pictures.index.min()

    log(pictures)

    emotionized = {}
    resting = {}
    for picture_name, _ in pictures.groupby('picture'):
        emotionized[picture_name] = {}

    # preprocess signals from every tool
    pictures_timezone_changed = False
    for tool in TOOLS:
        for signal_file in TOOLS[tool]:
            signal = TOOLS[tool][signal_file]
            data_path = path.join(one_experiment_path, tool, signal_file)

            log('Signal {}'.format(signal))

            # read measurements
            data = pd.read_csv(data_path)
            data.rename(columns=lambda x: x.strip(), inplace=True)
            data.timestamp = pd.to_datetime(data.timestamp, unit='ms')
            data.set_index('timestamp', inplace=True)

            # one hour mismatch (probably time zones)
            if pictures_start_time > data.index.min():
                timezone_to_change = pictures
                timezone_base = data
            else:
                timezone_to_change = data
                timezone_base = pictures

            timezone_difference = (timezone_to_change.index.min() - timezone_base.index.min()).seconds
            if timezone_difference >= 1*60*60:  # >= 1h
                log('    Timezone mismatch:\n{} - pictures starts\n{} - data (signal) starts'.format(
                        pictures_start_time, data.index.min()))

                if timezone_difference >= 2*60*60:  # >= 2
                    print('    Broken time (more than hour mismatch), skipping')
                    return None

                if timezone_to_change is pictures:
                    if pictures_timezone_changed:
                        # well, both signals may have wrong timezone
                        print('    Broken time (signals times are strange), skipping')
                        return None
                    pictures_timezone_changed = True

                print('    Fixing timezone')
                timezone_to_change.index = timezone_to_change.index.map(lambda v: v-pd.Timedelta(hours=1))
                pictures_start_time = pictures.index.min()

            # skip if there is (almost) no resting data
            if (pictures_start_time - data.index.min()).seconds < MIN_RESTING_TIME:
                print('    There is not resting data, skipping')
                return None

            # convert units
            # GSR microSiemens -> Ohm
            if tool == 'BITalino' and signal == 'GSR':
                log('    Converting GSR microSiemens -> Ohm')
                data.value = micro_siemens_to_ohm(data.value)

            # BVP microVolt -> nanoWatt?
            # seems to be ok, no need for conversion
            if tool == 'BITalino' and signal == 'BVP':
                pass

            # sample with correct frequency
            log('    Sampling frequency')
            freq_div = floor(data.size / (data.index.max() - data.index.min()).seconds / FREQ)
            data = data[::freq_div]

            # check if the signal is not constant (f.e. was not measured)
            if len(data[data.value != data.values[0][0]]) == 0:
                print('    {} signal was not measured - skipping'.format(signal))
                return None

            # get resting
            resting[signal] = data[data.index < pictures_start_time]

            # get emotionized
            data = data[data.index >= pictures_start_time]
            sum_of_measurements = len(data)

            # split emotionized by picture
            data = pd.merge(data, pictures.reindex(data.index,
                                        method='pad'), left_index=True, right_index=True)

            for picture_name, data_for_one_picture in data.groupby('picture'):
                emotionized[picture_name][signal] = data_for_one_picture

            # cut values for the last picture
            last_picture = pictures.tail(1).values[0][0]
            one_picture_measurements_mean = sum_of_measurements - len(emotionized[last_picture][signal])
            one_picture_measurements_mean /= len(emotionized) - 1
            emotionized[last_picture][signal] = emotionized[last_picture][signal].head(floor(one_picture_measurements_mean))

            if do_logs:
                print('    {} - {} | resting '.format(resting[signal].index.min(), resting[signal].index.max()))
                for picture_name in pictures.picture:
                    print('    {} - {} | {}'.format(emotionized[picture_name][signal].index.min(),
                        emotionized[picture_name][signal].index.max(),
                        picture_name))

            if do_plotting:
                print('    do plotting {}'.format(signal))
                prev_plot = plt.plot(resting[signal].index, resting[signal].value, label='resting')
                for picture_name in pictures.picture:
                    label = None
                    if is_trail_picture(picture_name): label = 'trail'
                    plt.plot(emotionized[picture_name][signal].index, emotionized[picture_name][signal].value,
                                                                        label=label)
                plt.legend(loc='best')
                plt.show()

            # transform DataFrame to list
            resting[signal] = resting[signal].values.reshape(1,-1).tolist()[0]
            for picture_name in pictures.picture:
                emotionized[picture_name][signal] = emotionized[picture_name][signal].iloc[:,0].values.reshape(1,-1).tolist()[0]

        # skip trail pictures
        for picture_name in list(emotionized):
            if is_trail_picture(picture_name):
                if picture_name in emotionized:
                    del(emotionized[picture_name])

        # use just one tool for now 
        break

    print('-'*40)
    return emotionized, resting


def preprocess_pictures(pictures_file_path):
    '''
    Valence and arousal for pictures

    Returns:
        pictures: dict[picture_name] = {'valence': 1.82, 'arousal': 7.05}
    '''
    print('Preprocessing pictures')

    with open(pictures_file_path, newline='') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        pictures = pd.read_csv(csvfile, dialect=dialect, decimal=",")

    # normalize picture names
    pictures.rename(columns={'ID': 'picture'}, inplace=True)
    pictures.picture = pictures.picture.apply(normalize_picture_name)

    # cols to lowercase
    pictures.columns = map(str.lower, pictures.columns)

    for col in pictures.columns:
        if col not in ['picture', 'valence', 'arousal']:
            del pictures[col]

    if DO_LOGS:
        print(pictures)

    pictures_dict = pictures.set_index('picture').to_dict('index')
    return pictures_dict


def main():
    # paths sanity check
    if not path.isfile(PATH_PICTURES) or not path.isdir(PATH_DATA):
        print('Configure paths in the script first')
        print('Now they are: "{}" and "{}"'.format(PATH_PICTURES, PATH_DATA))
        sys.exit(1)

    # get valence/arousal for pictures
    pictures = preprocess_pictures(PATH_PICTURES)
    with open(PICKLED_DATA_PICTURES, 'wb') as f:
        pickle.dump(pictures, f)

    # try to open previously preprocessed data
    all_data_emotionized = {}
    if path.isfile(PICKLED_DATA_EMOTIONIZED):
        try:
            with open(PICKLED_DATA_EMOTIONIZED, 'rb') as f:
                all_data_emotionized = pickle.load(f)
        except:
            pass

    all_data_resting = {}
    if path.isfile(PICKLED_DATA_RESTING):
        try:
            with open(PICKLED_DATA_RESTING, 'rb') as f:
                all_data_resting = pickle.load(f)
        except:
            pass

    # preprocess more data
    failures = 0
    preprocessed = 0
    skipped = 0
    experiments = filter_experiments(PATH_DATA)

    for i, one_experiment_path in enumerate(experiments):

        preprocess_it = True
        if len(PREPROCESS_ONLY) == 0:
            # preprocess all data
            if one_experiment_path in all_data_emotionized:
                # except already preprocessed
                preprocess_it = False
        else:
            # preprocess only if specified in PREPROCESS_ONLY
            if not any([po in one_experiment_path for po in PREPROCESS_ONLY]):
                preprocess_it = False

        if not preprocess_it:
            skipped += 1
            continue

        one_experiment_path = path.join(PATH_DATA, one_experiment_path)
        result = preprocess_one_experiment(one_experiment_path, do_plotting=DO_PLOTTING, do_logs=DO_LOGS)
        if result is None:
            failures += 1
            continue

        all_data_emotionized[one_experiment_path] = result[0]
        all_data_resting[one_experiment_path] = result[1]
        preprocessed += 1

        with open(PICKLED_DATA_EMOTIONIZED, 'wb') as f:
            pickle.dump(all_data_emotionized, f)
        with open(PICKLED_DATA_RESTING, 'wb') as f:
            pickle.dump(all_data_resting, f)

    print('failures: {}\npreprocessed: {}\nskipped: {}'.format(failures, preprocessed, skipped))
        

if __name__ == "__main__":
    main()

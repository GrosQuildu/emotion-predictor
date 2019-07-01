#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
~Gros
'''


from os import listdir, path
from math import floor
from glob import glob
import pandas as pd
# import matplotlib.pyplot as plt
import sys
import csv
import pickle

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


PATH = '/home/gros/studia/eaiib_5/wshop/data/2018-afcai-spring'
PATH_PICTURES = '/home/gros/studia/eaiib_5/wshop/data/NAPS_valence_arousal_2014.csv'

TOOLS = {'BITalino': {'BPM.csv':'BVP', 'GSR.csv':'GSR'}}
FREQ = 128

DO_PLOTTING = 0
DO_LOGS = 0

PICKLED_DATA_RESTING = 'preprocessed_geist_resting.pickle'
PICKLED_DATA_EMOTIONIZED = 'preprocessed_geist_emotionized.pickle'
PICKLED_DATA_PICTURES = 'preprocessed_geist_pictures.pickle'


def filter_experiments(all_experiments_path):
    '''
    Use only dirs/experiments/persons returned by this function
    Returns: list of paths to experiments

    As for now we use experiments with BITalino with two signals
    '''
    experiments = []
    for one_experiment in listdir(all_experiments_path):
        one_experiment_path = path.join(all_experiments_path, one_experiment)

        if not path.isdir(one_experiment_path):
            continue

        files_in_one_experiment = listdir(one_experiment_path)
        is_experiment_ok = True

        # check if all required tools were used in this experiment
        for tool in TOOLS:
            if tool not in files_in_one_experiment:
                is_experiment_ok = False
                break

            files_in_one_tool = listdir(path.join(one_experiment_path, tool))
            # check if all tools contains required data files
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
    '''
    return 'trail' in picture_name.lower()


def normalize_picture_name(picture_name):
    '''
    rename "pictures\\day1\\Landscapes_120_v.jpg" -> "Landscapes_120_v"
    '''
    if is_trail_picture(picture_name):
        return ''.join(picture_name.split('\\')[-2:]).rsplit('.', 1)[0]    
    return picture_name.split('\\')[-1].rsplit('.', 1)[0]


def preprocess_one_experiment(one_experiment_path, do_plotting=False, do_logs=False):
    '''
    Preprocess one experiments
    Args:
        one_experiment_path(str) - path to directory, should include TOOLS dirs
                                        and in each TOOLS should be files with signals
        plot(Bool) - guess what

    Returns:
        tuple(emotionized, resting)
            emotionized - dict[picture_names][signal] = [signal_value, signal_value2,...]
            resting - dict[signal] = [signal_value, signal_value2,...]
    '''
    print('preprocessing {}'.format(one_experiment_path))

    pictures_path = glob(path.join(one_experiment_path, '*_timestamp.csv'))[0]
    pictures = pd.read_csv(pictures_path, header=None, names=['picture', 'timestamp'], index_col=1, parse_dates=True)

    pictures.picture = pictures.picture.apply(normalize_picture_name)
    pictures_start_time = pictures.index.min()

    if do_logs:
        print(pictures)

    emotionized = {}
    resting = {}
    for picture_name, _ in pictures.groupby('picture'):
        emotionized[picture_name] = {}

    pictures_timezonde_changed = False
    for tool in TOOLS:
        for signal_file in TOOLS[tool]:
            signal = TOOLS[tool][signal_file]
            data_path = path.join(one_experiment_path, tool, signal_file)

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

            if (timezone_to_change.index.min() - timezone_base.index.min()).seconds >= 1*60*60:
                if timezone_to_change is pictures:
                    if pictures_timezonde_changed:
                        print('Broken timezones, skipping')
                        return None
                    pictures_timezonde_changed = True
                timezone_to_change.index = timezone_to_change.index.map(lambda v: v-pd.Timedelta(hours=1))
                pictures_start_time = pictures.index.min()

            # skip if there is (almost) no resting data
            if (pictures_start_time - data.index.min()).seconds < 5:
                print('There is not resting data, skipping')
                return None

            # convert units
            # GSR microSiemens -> Ohm
            if tool == 'BITalino' and signal == 'GSR':
                data.value = (10**6) / data.value

            # BVP microVolt -> nanoWatt?: todo
            if tool == 'BITalino' and signal == 'BVP':
                pass

            # sample with correct frequency
            freq_div = floor(data.size / (data.index.max() - data.index.min()).seconds / FREQ)
            data = data[::freq_div]

            # check if the signal is not constant (f.e. was not measured)
            if len(data[data.value != data.values[0][0]]) == 0:
                print('{} signal was not measured - skipping'.format(signal))
                return None

            # get resting
            resting[signal] = data[data.index < pictures.index.min()]

            # get emotionized
            data = data[data.index >= pictures_start_time]
            sum_of_measurements = len(data)

            # split emotionized by picture
            data = pd.merge(data, pictures.reindex(data.index,
                                        method='pad'), left_index=True, right_index=True)

            # todo: emozje wzbudzaja sie dobiero po jakims czasie
            # tj po kilku sekundach od pokazania obrazka
            # gosc wycinal jakies kawalki, chyba 15 sekund?
            # trzeba by to tez tutaj zrobic

            for picture_name, data_for_one_picture in data.groupby('picture'):
                emotionized[picture_name][signal] = data_for_one_picture#.iloc[:,0].values

            # cut values for the last picture
            last_picture = pictures.tail(1).values[0][0]
            # print(last_picture)
            # print(emotionized)
            one_picture_measurements_mean = sum_of_measurements - len(emotionized[last_picture][signal])
            one_picture_measurements_mean /= len(emotionized)-1
            emotionized[last_picture][signal] = emotionized[last_picture][signal].head(floor(one_picture_measurements_mean))

            if do_logs:
                print(resting[signal].index.min(), resting[signal].index.max(), 'resting')
                for picture_name in pictures.picture:
                    print(emotionized[picture_name][signal].index.min(),
                        emotionized[picture_name][signal].index.max(),
                        picture_name)
                print('-'*40)

            if do_plotting:
                print('do plotting {}'.format(signal))
                prev_plot = plt.plot(resting[signal].index, resting[signal].value, label='resting')
                for picture_name in pictures.picture:
                    plt.plot(emotionized[picture_name][signal].index, emotionized[picture_name][signal].value,
                                                                        label=None)
                plt.show()

        # skip trail pictures
        for picture_name in list(emotionized):
            if is_trail_picture(picture_name):
                if picture_name in emotionized:
                    del(emotionized[picture_name])

        # use just one tool for now 
        break

    return emotionized, resting


def preprocess_pictures(pictures_file_path):
    '''
    Valence and arousal for pictures

    Returns:
        pictures - dict[picture_name] = (valence, arousal)
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

    # print(pictures)
    return pictures


if __name__ == "__main__":
    # get valence/arousal for pictures
    pictures = preprocess_pictures(PATH_PICTURES)
    with open(PICKLED_DATA_PICTURES, 'wb') as f:
        pickle.dump(pictures, f)

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

    failures = 0
    preprocessed = 0
    skipped = 0
    for i, one_experiment_path in enumerate(filter_experiments(PATH)):
        if one_experiment_path in all_data_emotionized:
            skipped += 1
            continue

        one_experiment_path = path.join(PATH, one_experiment_path)
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

    print('failures: {}\npreprocessed: {}'.format(failures, preprocessed))
        
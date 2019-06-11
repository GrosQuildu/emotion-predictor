#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
~Gros
'''


from os import listdir, path
from math import floor
from glob import glob
import pandas as pd


PATH = '/home/gros/studia/eaiib_5/wshop/data/2018-afcai-spring'
TOOLS = {'BITalino': {'BPM.csv':'BVP', 'GSR.csv':'GSR'}}
FREQ = 128


def filter_experiments(all_experiments_path):
    '''
    Use only dirs/experiments/persons returned by this function
    Returns: list of paths to experiments 
    '''
    experiments = []
    for one_experiment in listdir(all_experiments_path):
        one_experiment_path = path.join(all_experiments_path, one_experiment)

        if not path.isdir(one_experiment_path):
            continue

        files_in_one_experiment = listdir(one_experiment_path)
        is_experiment_ok = True

        # check if all required tools were used in this experiment
        for tool in TOOLS.keys():
            if tool not in files_in_one_experiment:
                is_experiment_ok = False
                break

            files_in_one_tool = listdir(path.join(one_experiment_path, tool))
            # check if all tools contains required data files
            for required_data_file in TOOLS[tool].keys():
                if required_data_file not in files_in_one_tool:
                    is_experiment_ok = False
                    break

        # check if file with pictures exists:
        if len(glob(path.join(one_experiment_path, '*_timestamp.csv'))) != 1:
            is_experiment_ok = False

        if is_experiment_ok:
            experiments.append(one_experiment_path)

    return experiments


def micro_siemens_to_ohm(value):
        return (10**6)/value


def preprocess_one_experiment(one_experiment_path)
    pictures_path = glob(path.join(one_experiment_path, '*_timestamp.csv'))[0]
    pictures = pd.read_csv(pictures_path, header=None, names=['picture', 'timestamp'], index_col=1, parse_dates=True)
    print(pictures.index.min(), pictures.index.max())
    # print(pictures)

    emotionized = {}
    resting = {}
    pictures_timezonde_changed = False
    for tool in TOOLS.keys():
        for signal_file in TOOLS[tool].keys():
            signal = TOOLS[tool][signal_file]
            data_path = path.join(one_experiment_path, tool, signal_file)

            # read measurements
            data = pd.read_csv(data_path)
            data.rename(columns=lambda x: x.strip(), inplace=True)
            data.timestamp = pd.to_datetime(data.timestamp, unit='ms')
            data.set_index('timestamp', inplace=True)

            # one hour mismatch (probably time zones)
            if pictures.index.min() > data.index.min():
                timezone_to_change = pictures
            else:
                timezone_to_change = data

            if (timezone_to_change.index.min() - timezone_base.index.min()).seconds >= 1*60*60:
                if timezone_to_change is pictures:
                    if pictures_timezonde_changed:
                        print('Broken timezones')
                        return None
                    pictures_timezonde_changed = True
                timezone_to_change.index = timezone_to_change.index.map(lambda v: v-pd.Timedelta(hours=1))

            # skip if there is no resting data
            if (data.index.min() - pictures.index.min()).seconds < 5:
                return None

            # convert units
            # GSR microSiemens -> Ohm
            if tool == 'BITalino' and signal == 'GSR':
                # print(data)
                data['value'] = (10**6)/data['value']

            # BVP microVolt -> nanoWatt?: todo
            if tool == 'BITalino' and signal == 'BVP':
                pass

            # sample frequency
            freq_div = floor(data.size / (data.index.max() - data.index.min()).seconds / FREQ)
            data = data[::freq_div]
            print(data.index.min(), data.index.max())

            # get resting
            resting[signal] = data[data.index < pictures.index.min()]

            # get emotionized and merge with pictures
            data = pictures.reindex(data.index, method='nearest')

            # skip trail pictures
            

            emotionized[signal] = data
            # print(emotionized[signal][:3])

        # use just one tool for now 
        break

    return emotionized, resting

if __name__ == "__main__":
    skip_exp = 2
    for i, one_experiment_path in enumerate(filter_experiments(PATH)):
        if i < skip_exp:
            continue

        print('Preprocess {}'.format(one_experiment_path))
        one_experiment_path = path.join(PATH, one_experiment_path)
        preprocess_one_experiment(one_experiment_path)

        break
        
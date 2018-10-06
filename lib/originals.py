import numpy as np
import pyedflib as bdf
import sys
import matplotlib.pyplot as plt
import datetime
import struct

ORIGINALS_PATH = 'F:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_original_bdf'

GSR = 40
BVP = 45
STATUS = 47


class Originals:
    def get_person_resting_values(self, filename, number):
        reader = bdf.EdfReader(filename)

        gsr_signal = reader.readSignal(GSR)
        bvp_signal = reader.readSignal(BVP)
        begin, end = self._get_resting_markers(reader.readSignal(STATUS))

        rest_bvp = self._get_resting_data(bvp_signal, begin, end)
        # rest_gsr = self._get_resting_data(gsr_signal, begin, end)
        rest_gsr = gsr_signal[begin:end]
        if int(number) < 23:
            rest_gsr = self._convert_gsr_values(rest_gsr)

        return rest_bvp, rest_gsr

    def _get_resting_markers(self, status):
        markers = self._get_resting_markers_by_signal(status)
        if not markers:
            markers = self._get_resting_markers_by_time(status)
            if not markers:
                raise Exception("No resting time found")

        return markers

    def _get_resting_markers_by_signal(self, status):
        current = None
        last_index = 0
        sum = 0

        begin = None
        end = None
        mode = 0

        # print(f"0:00:00: start of the experiment")
        for index, value in enumerate(status):
            value = self.get_least_byte(int(value))
            if value != current:
                if value == 6:
                    continue

                change = (index - last_index) / 512
                sum += change
                # time = datetime.timedelta(seconds=sum)
                # print(f"{# time}: Value {current} length {change}s , mode={mode}")
                if current == 1:
                    mode += 1

                if current == 0 and mode == 2:
                    begin = last_index
                    end = index
                    break
                current = value
                last_index = index

        seconds = (end - begin) / 512

        if (seconds < 30) or not begin or not end:
            return None

        return begin, end

    def _get_resting_markers_by_time(self, status):
        current = None
        last_index = 0

        begin = None
        end = None

        for index, value in enumerate(status):
            value = self.get_least_byte(int(value))
            if value != current:
                if value == 6:
                    continue

                change = (index - last_index) / 512
                if 100 < change < 130:
                    begin = last_index
                    end = index
                    break

                current = value
                last_index = index

        if begin is None or end is None:
            return None

        return begin, end

    def _get_resting_data(self, signal, begin, end):
        resting = []
        for index, value in enumerate(signal):
            if index < begin:
                continue
            if index > end:
                break

            resting.append(value)

        return resting[::4]

    def _get_absolute_file_name(self, number):
        return "{}s{}.bdf".format(ORIGINALS_PATH, number)

    def _convert_gsr_values(self, values):
        new_values = []
        for v in values:
            converted = (10**9)/v
            new_values.append(converted)

        return new_values

    def get_least_byte(self, value):
        result = []

        for i in range(0, 1):
            result.append(value >> (i * 8) & 0xff)

        result.reverse()

        return result[0]

    def plot(self, data):
        x = list(range(0, len(data)))
        plt.close('all')
        plt.figure(figsize=(32, 6))
        plt.plot(
            x,
            data,
            'r-'
        )
        plt.grid()
        plt.show()


import numpy as np
import pyedflib as bdf
import sys
import matplotlib.pyplot as plt


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
        rest_gsr = self._get_resting_data(gsr_signal, begin, end)
        if number > 22:
            rest_gsr = self._convert_gsr_values(rest_gsr)

        return rest_bvp, rest_gsr

    def _get_resting_markers(self, status):
        current = None
        last_index = 0
        for index, value in enumerate(status):
            if value != current:
                current = value
                change = (index - last_index) / 512
                if 119 < change < 121:
                    return last_index, index
                last_index = index

        raise Exception("No resting time found")

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
        return "{}s{}.bdf".format(BASE_PATH, number)

    def _convert_gsr_values(self, values):
        # TODO: implement converter
        return values


originals = Originals()
originals.get_base_values()

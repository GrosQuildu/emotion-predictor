import numpy
from lib.bvp import BVP
from lib.gsr import GSR
import sys


CHANNELS = {
    'bvp': 38,
    'gsr': 36
}

DATA_FREQUENCY = 128


class Preprocessing:
    def load_data_from_file(self, file_path):
        loaded = numpy.load(file_path, allow_pickle=True, encoding='bytes')

        return {
            'data': loaded[b'data'],
            'labels': loaded[b'labels']
        }

    def process_person(self, file):
        data = self.load_data_from_file(file)
        for i in range(len(data['data'])):

            bpm = self._run_bvp(data['data'][i])
            timestamps = [el[0] for el in bpm]
            gsr = self._run_gsr(data['data'][i], timestamps)

            tuples = self._create_data_tuples(bpm, gsr, data['labels'][i])
            print(tuples)
            sys.exit(0)

    def _run_bvp(self, data):
        num = len(data[CHANNELS['bvp']])
        bvp = BVP(
            list(range(0, num)),
            data[CHANNELS['bvp']],
            DATA_FREQUENCY
        )
        return bvp.convert_to_bpm(avg=True)

    def _run_gsr(self, data, timestamps):
        gsr = GSR(
            data[CHANNELS['gsr']],
            timestamps
        )
        return gsr.match_timestamps()

    def _create_data_tuples(self, bpm, gsr, label):
        results = []
        for i in range(len(bpm)):
            results.append((bpm[i][1], gsr[i][1], label[0], label[1]))

        return results
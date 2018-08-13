import numpy
from lib.bvp import BVP


CHANNELS = {
    'bvp': 38,
    'gsr': 36
}


class Preprocessing:
    def load_data_from_file(self, file_path):
        loaded = numpy.load(file_path, allow_pickle=True, encoding='bytes')

        return {
            'data': loaded[b'data'],
            'labels': loaded[b'labels']
        }

    def process_person(self, file):
        data = self.load_data_from_file(file)
        self._run_bvp(data)

    def _run_bvp(self, data):
        num = len(data['data'][0][CHANNELS['bvp']])
        bvp = BVP(
            list(range(0, num)),
            data['data'][0][CHANNELS['bvp']],
            128
        )
        bpm = bvp.convert_to_bpm(avg=True)
        print(bpm)

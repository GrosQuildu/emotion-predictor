import numpy
import matplotlib.pyplot as plt
from lib.bvp import BVP
from lib.gsr import GSR
from statistics import mean, StatisticsError
import sys


CHANNELS = {
    'bvp': 38,
    'gsr': 36
}
#todo move
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
        person_result = []
        for i in range(len(data['data'])):
            video_result = {}
            bpm = self._run_bvp(data['data'][i])
            timestamps = [el[0] for el in bpm]
            gsr = self._run_gsr(data['data'][i], timestamps)

            video_result = {
                'bpm': bpm,
                'gsr': gsr,
                'valence': data['labels'][i][0],
                'arousal': data['labels'][i][1]
            }
            person_result.append(video_result)

        person_result = self._get_person_data_avg(person_result)
        # self._show_bpm_plot(person_result, 0)
        # self._show_bpm_plot(person_result, 10)
        return person_result

    def _run_bvp(self, data):
        num = len(data[CHANNELS['bvp']])
        bvp = BVP(
            list(range(0, num)),
            data[CHANNELS['bvp']],
            DATA_FREQUENCY
        )
        return bvp.convert_to_bpm(show_plot=False, show_output_plot=False)

    def _run_gsr(self, data, timestamps):
        gsr = GSR(
            data[CHANNELS['gsr']],
            timestamps,
            DATA_FREQUENCY
        )
        return gsr.match_timestamps(show_plot=False)

    def _get_person_data_avg(self, person_data):
        bpm_list = []
        gsr_list = []

        for i in person_data:
            for j in i['bpm']:
                bpm_list.append(j[1])

            for j in i['gsr']:
                gsr_list.append(j[1])

        bpm_avg = mean(bpm_list)
        gsr_avg = mean(gsr_list)

        for i in person_data:
            new_bpm = []
            for j in i['bpm']:
                new_bpm.append((j[0], j[1] - bpm_avg))
            i['bpm'] = new_bpm

            new_gsr = []
            for j in i['gsr']:
                new_gsr.append((j[0], j[1] - gsr_avg))
            i['gsr'] = new_gsr

        return person_data

    def _create_data_tuples(self, bpm, gsr, label):
        results = []
        for i in range(len(bpm)):
            results.append((bpm[i][1], gsr[i][1], label[0], label[1]))

        return results

    def _show_bpm_plot(self, data, video_id):
        x = [i[0]/DATA_FREQUENCY for i in data[video_id]['bpm']]
        y = [i[1] for i in data[video_id]['bpm']]

        plt.figure(figsize=(32, 6))
        plt.plot(x, y, 'b-')
        plt.title("BPM plot (valence={} , arousal={})".format(
            data[video_id]['valence'],
            data[video_id]['arousal']
        ))
        plt.grid()
        plt.show()

    def _show_gsr_plot(self, data, video_id):
        x = [i[0] / DATA_FREQUENCY for i in data[video_id]['gsr']]
        y = [i[1] for i in data[video_id]['gsr']]

        plt.figure(figsize=(32, 6))
        plt.plot(x, y, 'b-')
        plt.title("BPM plot (valence={} , arousal={})".format(
            data[video_id]['valence'],
            data[video_id]['arousal']
        ))
        plt.grid()
        plt.show()

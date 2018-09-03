import numpy
import matplotlib.pyplot as plt
from lib.bvp import BVP
from lib.gsr import GSR
from lib.originals import Originals
from statistics import mean, StatisticsError
from sklearn.preprocessing import MinMaxScaler
import sys


CHANNELS = {
    'bvp': 38,
    'gsr': 36
}


class Preprocessing:
    def __init__(self, data_frequency):
        self._data_frequency = data_frequency
        self._org = Originals()

    def load_data_from_file(self, file_path):
        loaded = numpy.load(file_path, allow_pickle=True, encoding='bytes')

        return {
            'data': loaded[b'data'],
            'labels': loaded[b'labels']
        }

    def process_person(self, file, original_file, file_number):
        base_bvp, base_gsr = self._org.get_person_resting_values(original_file, file_number)
        # todo convert bvp to bpm, get avg values
        data = self.load_data_from_file(file)
        person_result = []
        for i in range(len(data['data'])):
            try:
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
            except StatisticsError:
                print("Malformed data")
                continue

        person_result = self._get_person_data_avg(person_result)
        # self._show_combined_plot(
        #     person_result[0]['bpm'],
        #     person_result[0]['gsr'],
        #     person_result[0]['valence'],
        #     person_result[0]['arousal'],
        #     normalize=True
        # )
        # sys.exit(0)

        # self._show_bpm_plot(person_result, 0)
        # self._show_bpm_plot(person_result, 10)
        return person_result

    def _run_bvp(self, data):
        num = len(data[CHANNELS['bvp']])
        bvp = BVP(
            list(range(0, num)),
            data[CHANNELS['bvp']],
            self._data_frequency
        )
        return bvp.convert_to_bpm(show_plot=True, show_output_plot=True)

    def _run_gsr(self, data, timestamps):
        gsr = GSR(
            data[CHANNELS['gsr']],
            timestamps,
            self._data_frequency
        )
        return gsr.match_timestamps(show_plot=False, avg=False)

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
                new_bpm.append((j[0], self._get_percentage_diff(j[1], bpm_avg)))

            i['bpm'] = new_bpm

            new_gsr = []
            for j in i['gsr']:
                new_gsr.append((j[0], self._get_percentage_diff(j[1], gsr_avg)))
            i['gsr'] = new_gsr

        return person_data

    def _create_data_tuples(self, bpm, gsr, label):
        results = []
        for i in range(len(bpm)):
            results.append((bpm[i][1], gsr[i][1], label[0], label[1]))

        return results

    def _show_bpm_plot(self, data, video_id):
        x = [i[0] / self._data_frequency for i in data[video_id]['bpm']]
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
        x = [i[0] / self._data_frequency for i in data[video_id]['gsr']]
        y = [i[1] for i in data[video_id]['gsr']]

        plt.figure(figsize=(32, 6))
        plt.plot(x, y, 'b-')
        plt.title("BPM plot (valence={} , arousal={})".format(
            data[video_id]['valence'],
            data[video_id]['arousal']
        ))
        plt.grid()
        plt.show()

    def _show_combined_plot(self, bpm, gsr, valence, arousal, normalize=False):
        x_bpm = []
        y_bpm = []
        for i in bpm:
            x_bpm.append(i[0])
            y_bpm.append(i[1])

        x_gsr = []
        y_gsr = []
        for i in gsr:
            x_gsr.append(i[0])
            if normalize:
                y_gsr.append(i[1]/1000)
            else:
                y_gsr.append(i[1])

        plt.figure(figsize=(32, 6))
        plt.plot(x_bpm, y_bpm, 'r-', x_gsr, y_gsr, 'b-')
        plt.title(f"Combined plot (valence={valence} , arousal={arousal})")
        plt.grid()
        plt.show()

    def _get_percentage_diff(self, value, avg):
        diff = value - avg
        return (diff/avg)*100

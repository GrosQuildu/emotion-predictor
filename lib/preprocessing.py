import numpy
import matplotlib.pyplot as plt
import scipy.fftpack
from lib.bvp import BVP
from lib.new_bvp import NewBVP
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
        avg_bpm = self._get_avg_bpm(base_bvp)
        avg_gsr = self._get_avg_gsr(base_gsr)
        data = self.load_data_from_file(file)
        person_result = []
        for i in range(len(data['data'])):
            try:
                bpm = self._run_bvp(data['data'][i])
                if bpm == 0.0:
                    print("Fatal error")
                    print(file)
                    print(i)
                    sys.exit(0)
                # timestamps = [el[0] for el in bpm]
                gsr = self._run_gsr(data['data'][i], None)

                video_result = {
                    'bpm': bpm,
                    'gsr': gsr,
                    'valence': data['labels'][i][0],
                    'arousal': data['labels'][i][1]
                }
                person_result.append(video_result)
            except Exception:
                print(f"Malformed data when processing video {i}")
                continue

        person_result = self._get_person_data_diff(person_result, avg_bpm, avg_gsr)
        return person_result

    def _run_bvp(self, data):
        num = len(data[CHANNELS['bvp']])
        bvp = NewBVP(
            list(range(0, num)),
            data[CHANNELS['bvp']],
            self._data_frequency
        )
        return bvp.convert_to_bpm(show_plot=False, show_output_plot=False)

    def _run_gsr(self, data, timestamps):
        gsr = GSR(
            data[CHANNELS['gsr']],
            timestamps,
            self._data_frequency
        )
        # return gsr.get_dropping_time()
        return gsr.get_avg_resistance()

    def _get_person_data_diff(self, person_data, bpm_avg, gsr_avg):
        for i in person_data:
            i['bpm'] = bpm_avg - i['bpm']
            i['gsr'] = gsr_avg - i['gsr']

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

    def _get_avg_bpm(self, bvp):
        bvp = NewBVP(
            list(range(len(bvp))),
            bvp,
            self._data_frequency
        )
        return bvp.convert_to_bpm(show_plot=False, show_output_plot=False)

    def _get_avg_gsr(self, gsr):
        return mean(gsr)

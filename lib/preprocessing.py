import numpy
import matplotlib.pyplot as plt
from lib.signals.new_bvp import NewBVP
from lib.signals.new_gsr import NewGSR
from lib.originals import Originals
import sys


CHANNELS = {
    'bvp': 38,
    'gsr': 36
}


class Preprocessing:
    def __init__(self, data_frequency, extract_all_features=False, need_preprocessing=False):
        self._data_frequency = data_frequency
        self._org = Originals()
        self._extract_all_features = extract_all_features
        self._need_preprocessing = need_preprocessing

    def load_data_from_file(self, file_path):
        loaded = numpy.load(file_path, allow_pickle=True, encoding='bytes')

        return {
            'data': loaded[b'data'],
            'labels': loaded[b'labels']
        }

    def process_person(self, file, original_file, file_number):
        try:
            base_bvp, base_gsr = self._org.get_person_resting_values(original_file, file_number)
            heart_avg = self._get_avg_bpm(base_bvp)
            gsr_avg = self._get_avg_gsr(base_gsr, "avg_{}".format(file_number))
        except Exception:
            print(f"Malformed data when processing baseline of {file_number}")
            raise

        data = self.load_data_from_file(file)
        person_result = []
        for i in range(len(data['data'])):
            try:
                heart = self._run_bvp(data['data'][i])
                gsr = self._run_gsr(data['data'][i], "video_{}_{}".format(file_number, i))

                video_result = {
                    'heart': heart,
                    'gsr': gsr,
                    'valence': data['labels'][i][0],
                    'arousal': data['labels'][i][1]
                }
                person_result.append(video_result)
            except Exception:
                print(f"Malformed data when processing video {i}")
                continue

        person_result = self._get_person_data_diff(person_result, heart_avg, gsr_avg)
        return person_result

    # start=2, stop=9
    def _run_bvp(self, data):
        bvp_signal = self._trim_signal(data[CHANNELS['bvp']], self._data_frequency, start=2, stop=17)
        bvp = NewBVP(
            None,
            bvp_signal,
            self._data_frequency
        )
        return bvp.get_features(extract_all_features=self._extract_all_features)

    def _run_gsr(self, data, filename):
        gsr_signal = self._trim_signal(data[CHANNELS['gsr']], self._data_frequency, start=2, stop=17)
        gsr = NewGSR(
            gsr_signal,
            self._data_frequency,
            filename=filename,
            file=self._need_preprocessing
        )
        return gsr.get_features(extract_all_features=self._extract_all_features)

    def _get_person_data_diff(self, person_data, heart_avg, gsr_avg):
        for i in person_data:
            for key in i['heart']:
                i['heart'][key] = heart_avg[key] - i['heart'][key]

            for key in i['gsr']:
                i['gsr'][key] = gsr_avg[key] - i['gsr'][key]

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
        return bvp.get_features(extract_all_features=self._extract_all_features)

    def _get_avg_gsr(self, gsr, filename):
        gsr = NewGSR(gsr, 512, filename=filename, file=self._need_preprocessing)
        return gsr.get_features(extract_all_features=self._extract_all_features)

    def _trim_signal(self, signal, frequency, start=0, stop=0):
        if not start and not stop:
            return signal

        start *= frequency
        stop *= frequency

        return signal[start:stop]

    @staticmethod
    def get_labels():
        return NewBVP.labels + NewGSR.labels

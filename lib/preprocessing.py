import numpy
import matplotlib.pyplot as plt
from lib.signals.new_bvp import NewBVP
from lib.signals.new_gsr import NewGSR
from lib.originals import Originals
from config import EXTRACT_ALL_FEATURES, NEED_PREPROCESSING, SIGNAL_BEGIN, SIGNAL_END


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
        try:
            # BVP comes downsampled while GSR comes with original frequency (512 Hz)
            base_bvp, base_gsr = self._org.get_person_resting_values(original_file, file_number)
            heart_avg = self._get_base_bvp_features(base_bvp)
            gsr_avg = self._get_base_gsr_features(base_gsr, "avg_{}".format(file_number))
        except Exception:
            print(f"Malformed data when processing baseline of {file_number}")
            raise

        data = self.load_data_from_file(file)
        person_result = []
        for i in range(len(data['data'])):
            try:
                heart = self._get_bvp_features(data['data'][i], start=SIGNAL_BEGIN, stop=SIGNAL_END)
                gsr = self._get_gsr_features(data['data'][i], "video_{}_{}".format(file_number, i), start=SIGNAL_BEGIN, stop=SIGNAL_END)

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

        person_result = self._convert_features_to_diffs(person_result, heart_avg, gsr_avg)
        return person_result

    def _get_bvp_features(self, data, start=0, stop=0):
        bvp_signal = self._trim_signal(data[CHANNELS['bvp']], self._data_frequency, start=start, stop=stop)
        bvp = NewBVP(
            bvp_signal,
            self._data_frequency,
            show_plot=False
        )
        return bvp.get_features(extract_all_features=EXTRACT_ALL_FEATURES)

    def _get_gsr_features(self, data, filename, start=0, stop=0):
        gsr_signal = self._trim_signal(data[CHANNELS['gsr']], self._data_frequency, start=start, stop=stop)
        gsr = NewGSR(
            gsr_signal,
            self._data_frequency,
            filename=filename,
            file=NEED_PREPROCESSING,
            plot=False
        )
        return gsr.get_features(extract_all_features=EXTRACT_ALL_FEATURES)

    def _get_base_bvp_features(self, bvp):
        bvp = NewBVP(
            bvp,
            self._data_frequency
        )
        return bvp.get_features(extract_all_features=EXTRACT_ALL_FEATURES)

    def _get_base_gsr_features(self, gsr, filename):
        gsr = NewGSR(gsr, 512, filename=filename, file=NEED_PREPROCESSING)
        return gsr.get_features(extract_all_features=EXTRACT_ALL_FEATURES)

    def _convert_features_to_diffs(self, person_data, heart_avg, gsr_avg):
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

    def _trim_signal(self, signal, frequency, start=0, stop=0):
        if not start and not stop:
            return signal

        start *= frequency
        stop *= frequency

        return signal[start:stop]

    ############ LEGACY METHODS ############

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

    @staticmethod
    def get_labels():
        return NewBVP.labels + NewGSR.labels

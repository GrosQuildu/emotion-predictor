import numpy

from config import EXTRACT_ALL_FEATURES, NEED_PREPROCESSING, SIGNAL_BEGIN, SIGNAL_END, SHOW_PLOTS
from lib.originals import Originals
from lib.signals.new_bvp import NewBVP
from lib.signals.new_gsr import NewGSR


# channel numbers in initially preprocessed files
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
            heart_avg = self.get_base_bvp_features(base_bvp)
            gsr_avg = self.get_base_gsr_features(base_gsr, filename="avg_{}".format(file_number))
        except Exception:
            print(f"Malformed data when processing baseline of {file_number}")
            raise

        data = self.load_data_from_file(file)
        person_result = []
        for i in range(len(data['data'])):
            try:
                heart = self._get_bvp_features(data['data'][i])
                gsr = self._get_gsr_features(
                    data['data'][i],
                    filename="video_{}_{}".format(file_number, i)
                )

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

        person_result = self._calculate_feature_diffs(person_result, heart_avg, gsr_avg)
        return person_result

    def get_diffed_values(self, bvp, gsr, base_bvp_features, base_gsr_features):
        bvp_features = self._get_bvp_features(bvp)
        gsr_features = self._get_gsr_features(gsr, None)

        features = []

        for key, value in bvp_features:
            bvp_features[key] = base_bvp_features[key] - bvp_features[key]
            features.append(bvp_features[key])

        for key, value in gsr_features:
            gsr_features[key] = base_gsr_features[key] - gsr_features[key]
            features.append(gsr_features[key])

        return features

    def get_base_bvp_features(self, bvp):
        bvp = NewBVP(
            bvp,
            self._data_frequency
        )
        return bvp.get_features(extract_all_features=EXTRACT_ALL_FEATURES)

    def get_base_gsr_features(self, gsr, filename=None):
        gsr = NewGSR(gsr, 512, filename=filename, file=NEED_PREPROCESSING)
        return gsr.get_features(extract_all_features=EXTRACT_ALL_FEATURES)

    def _get_bvp_features(self, data):
        bvp_signal = self._trim_signal(data[CHANNELS['bvp']], self._data_frequency, start=SIGNAL_BEGIN, stop=SIGNAL_END)
        bvp = NewBVP(
            bvp_signal,
            self._data_frequency,
            show_plot=SHOW_PLOTS
        )
        return bvp.get_features(extract_all_features=EXTRACT_ALL_FEATURES)

    def _get_gsr_features(self, data, filename=None):
        gsr_signal = self._trim_signal(data[CHANNELS['gsr']], self._data_frequency, start=SIGNAL_BEGIN, stop=SIGNAL_END)
        gsr = NewGSR(
            gsr_signal,
            self._data_frequency,
            filename=filename,
            file=NEED_PREPROCESSING,
            show_plot=SHOW_PLOTS
        )
        return gsr.get_features(extract_all_features=EXTRACT_ALL_FEATURES)

    def _calculate_feature_diffs(self, person_data, heart_avg, gsr_avg):
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

    @staticmethod
    def get_labels():
        return NewBVP.labels + NewGSR.labels

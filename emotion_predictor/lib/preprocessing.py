import numpy

from emotion_predictor.config import EXTRACT_ALL_FEATURES, NEED_PREPROCESSING, SIGNAL_BEGIN, SIGNAL_END, SHOW_PLOTS
from emotion_predictor.lib.signals.new_bvp import NewBVP
from emotion_predictor.lib.signals.new_gsr import NewGSR

import pickle


# channel numbers in initially preprocessed files
CHANNELS = {
    'bvp': 38,
    'gsr': 36
}


class Preprocessing:
    def __init__(self, data_frequency):
        self._data_frequency = data_frequency

    def load_data_from_file(self, file_path):
        """
        Loads initially preprocessed files
        :param file_path: Path to file to load
        :return: dict containing data and labels
        """
        loaded = numpy.load(file_path, allow_pickle=True, encoding='bytes')

        return {
            'data': loaded[b'data'],
            'labels': loaded[b'labels']
        }

    def process_person(self, emotionized_file, resting_file, pictures_file):
        """
        Processes all trials showed to a single person
        :param file: Path to initially preprocessed file
        :param resting_file: Path to resting file
        :param file_number: Person No
        :return: dict containing information about all trials
        """

        signals = ['BVP', 'GSR']

        with open(emotionized_file, 'rb') as f:
            data_emotionized = pickle.load(f)
        with open(resting_file, 'rb') as f:
            data_resting = pickle.load(f)
        with open(pictures_file, 'rb') as f:
            data_pictures = pickle.load(f)

        # lecimy po ekperymentach
        for experiment_path in data_emotionized.keys():
            # bierzemy dane spoczynkowe (przed rozpoaczeciem eksperymentu)
            try:
                base_bvp, base_gsr = data_resting[experiment_path]['BVP'], data_resting[experiment_path]['GSR']
                heart_avg = self.get_base_bvp_features(base_bvp)
                gsr_avg = self.get_base_gsr_features(base_gsr, filename="avg_{}".format(file_number))
            except Exception:
                print(f"Malformed data when processing baseline of {file_number}")
                continue

            person_result = []
            # lecimy po obrazkach
            for picture_name, one_image_data in data_emotionized[experiment_path].iteritems():
                try:
                    heart = self._get_bvp_features(one_image_data['BVP'])
                    gsr = self._get_gsr_features(
                        one_image_data['GSR'],
                        # that filename should be escaped (it may contains / which may break the function)
                        filename="video_{}_{}".format(experiment_path, picture_name)
                    )

                    video_result = {
                        'heart': heart,
                        'gsr': gsr,
                        'valence': data_pictures[picture_name][0],
                        'arousal': data_pictures[picture_name][1]
                    }
                    person_result.append(video_result)
                except Exception:
                    print(f"Malformed data when processing image {picture_name}")
                    continue

            # zwracamy roznice miedzy danymi z emocjami a spoczynkowymi (bez emocji)
            person_result = self._calculate_feature_diffs(person_result, heart_avg, gsr_avg)
            yield person_result
        


    def process_person_old(self, file, original_file, file_number):
        """
        Processes all trials showed to a single person
        :param file: Path to initially preprocessed file
        :param original_file: Path to original file
        :param file_number: Person No
        :return: dict containing information about all trials
        """
        try:
            # BVP comes downsampled while GSR comes with original frequency (512 Hz)
            # this can be processed ahead of time
            # because is strongly data-dependent (channels numbers etc)
            base_bvp, base_gsr = self._org.get_person_resting_values(original_file, file_number)
            # base_bvp: [val1, val2,...], len = X
            # base_gsr: [val1, val2,...], len = 4*X

            heart_avg = self.get_base_bvp_features(base_bvp)
            # NewBVP -> biosppy.process -> heartbeat
            # {'feature_name': value, 'feature2_name': value2}

            gsr_avg = self.get_base_gsr_features(base_gsr, filename="avg_{}".format(file_number))
            # NewGSR -> neurokit.eda_process -> statistics
            # {'feature_name': value, 'feature2_name': value2}
        except Exception:
            print(f"Malformed data when processing baseline of {file_number}")
            raise

        data = self.load_data_from_file(file)
        person_result = []
        # iterate over videos/trials
        for i in range(len(data['data'])):
            try:
                # data['data'][i] -> array 40x8064 of channelXdata
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
        # person_result = [{'heart':{'feature_name': value,...}, 'gsr':{'feature_name': value,...}, 'valence':X, 'arousal':X2}, ...]

        person_result = self._calculate_feature_diffs(person_result, heart_avg, gsr_avg)
        # person_result = [{'heart':{'feature_name': value_avg-value,...}, 'gsr':{'feature_name': value_avg-value,...}, 'valence':X, 'arousal':X2}, ...]
        return person_result

    def get_diffed_values(self, bvp, gsr, base_bvp_features, base_gsr_features):
        """
        Calculates features on BVP and GSR features and returns their difference with features from base signals
        """
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
        """
        Returns features from base BVP signal
        :param bvp:
        :return:
        """
        bvp = NewBVP(
            bvp,
            self._data_frequency
        )
        return bvp.get_features(extract_all_features=EXTRACT_ALL_FEATURES)

    def get_base_gsr_features(self, gsr, filename=None):
        """
        Returns features from base GSR signal
        :param gsr:
        :param filename:
        :return:
        """
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
        """
        Returns labels from both BVP and GSR
        """
        return NewBVP.labels + NewGSR.labels

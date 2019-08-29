from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import emotion_predictor.lib.emotion as em


class Postprocessing:
    """
    This class contains methods used to transform data to be ready to use in machine learning
    """
    def __init__(self, freq):
        self._freq = freq

    def make_data_tuples(self, people):
        """
        Converts preprocessing result to X, Y lists of tuples
        :param people: list of dicts, preprocessing result
        :return: X, Y lists of tuples
        """
        x = []
        y = []

        for person in people:
            for image in person:
                if not image['heart'] and not image['gsr']:
                    continue
                emotion = em.get_class_for_values(image['valence'], image['arousal'])

                bpm_attr_list = []
                for key in image['heart']:
                    bpm_attr_list.append(image['heart'][key])

                gsr_attr_list = []
                for key in image['gsr']:
                    gsr_attr_list.append(image['gsr'][key])

                all_features = bpm_attr_list + gsr_attr_list
                if self._has_missing_feature(all_features):
                    continue

                x.append(all_features)
                y.append(emotion)

        return x, y

    def normalize(self, data):
        """
        Normalizes given matrix
        """
        mms = MinMaxScaler()
        return mms.fit_transform(data)

    def standarize(self, data):
        """
        Standarizes given matrix
        """
        stdsc = StandardScaler()
        return stdsc.fit_transform(data)

    def _has_missing_feature(self, features):
        features = [features]
        df = DataFrame(features)
        return len(df.dropna()) == 0

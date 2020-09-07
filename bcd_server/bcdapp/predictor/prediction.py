import pickle
import re
import librosa
import numpy as np
import json
import os
import math
from .predictor_configs import CRYING_BABY_LABEL


class FeatureEngineer:
    RATE = 44100   # All recordings in ESC are 44.1 kHz
    FRAME = 512    # Frame size in samples

    def __init__(self):
        pass

    def feature_engineer(self, audio_data):
        zcr_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='zero_crossing_rate')
        rmse_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='rmse')
        mfcc_feat = self.compute_librosa_features(audio_data=audio_data, feat_name= 'mfcc')
        spectral_centroid_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='spectral_centroid')
        spectral_rolloff_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='spectral_rolloff')
        spectral_bandwidth_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='spectral_bandwidth')

        concat_feat = np.concatenate((zcr_feat,
                                      rmse_feat,
                                      mfcc_feat,
                                      spectral_centroid_feat,
                                      spectral_rolloff_feat,
                                      spectral_bandwidth_feat
                                      ), axis=0)

        return np.mean(concat_feat, axis=1, keepdims=True).transpose()

    def compute_librosa_features(self, audio_data, feat_name):
        if feat_name == 'zero_crossing_rate':
            return librosa.feature.zero_crossing_rate(y=audio_data, hop_length=self.FRAME)
        elif feat_name == 'rmse':
            return librosa.feature.rms(y=audio_data, hop_length=self.FRAME)
        elif feat_name == 'mfcc':
            return librosa.feature.mfcc(y=audio_data, sr=self.RATE, n_mfcc=13)
        elif feat_name == 'spectral_centroid':
            return librosa.feature.spectral_centroid(y=audio_data, sr=self.RATE, hop_length=self.FRAME)
        elif feat_name == 'spectral_rolloff':
            return librosa.feature.spectral_rolloff(y=audio_data, sr=self.RATE, hop_length=self.FRAME, roll_percent=0.90)
        elif feat_name == 'spectral_bandwidth':
            return librosa.feature.spectral_bandwidth(y=audio_data, sr=self.RATE, hop_length=self.FRAME)


class Reader:
    def __init__(self, file_name):
        self.file_name = file_name
        pass

    def read_audio_file(self):
        play_list = list()
        header_duration = librosa.get_duration(filename=self.file_name)

        if header_duration < 5:
            audio_data, _ = librosa.load(self.file_name, sr=44100, mono=True)
            play_list.append(audio_data)
            return play_list

        splits_number = math.ceil(header_duration - 4)

        for offset in range(splits_number):
            audio_data, _ = librosa.load(self.file_name, sr=44100, mono=True, offset=offset, duration=5.0)
            play_list.append(audio_data)

        return play_list


class Prediction:
    def __init__(self, prediction, accuracy, confidence):
        self.prediction = prediction
        self.accuracy = accuracy
        self.confidence = confidence


class MajorityVoter:
    def __init__(self, prediction_list):
        self.predictions = prediction_list
        self.result = None
        self.positive_weight = None
        self.sum_weights = None

    def vote(self):
        self.positive_weight = 0
        self.sum_weights = 0
        for pred in self.predictions:
            self.positive_weight += pred.prediction * pred.accuracy * pred.confidence
            self.sum_weights += pred.accuracy

        if self.positive_weight > self.sum_weights/2.0:
            self.result = 1
        else:
            self.result = 0

        return self

    def get_dictionary(self):
        return {
            'prediction': self.result,
            'cry': self.positive_weight / self.sum_weights
        }


class BabyCryPredictor:
    def __init__(self, model):
        self.model = model

    def classify(self, new_signal):
        confidence = 1
        category = self.model.predict(new_signal)

        try:
            confidence = self.model.predict_proba(new_signal)[0][0]
        except AttributeError as error:
            pass

        return self._is_baby_cry(category[0]), confidence

    @staticmethod
    def _is_baby_cry(string):
        match = re.search(CRYING_BABY_LABEL, string)
        if match:
            return 1
        else:
            return 0


def predict(file_name):
    final_models = [
        {'model_path': "svc_model.pkl", 'perf_path': "svc_performance.json",
         'preds': [], 'name': "svc", 'voter': {}},
        {'model_path': "svc_v1_model.pkl", 'perf_path': "svc_v1_performance.json",
         'preds': [], 'name': "svc_v1", 'voter': {}},
        {'model_path': "linsvc_model.pkl", 'perf_path': "linsvc_performance.json",
         'preds': [], 'name': "linsvc", 'voter': {}},
        {'model_path': "mlp_model.pkl", 'perf_path': "mlp_performance.json",
         'preds': [], 'name': "mlp", 'voter': {}}
    ]

    file_reader = Reader(file_name)
    play_list = file_reader.read_audio_file()

    # Feature extraction
    engineer = FeatureEngineer()
    play_list_processed = list()
    for signal in play_list:
        tmp = engineer.feature_engineer(signal)
        proc_signal = {"data": tmp, "preds": list()}
        play_list_processed.append(proc_signal)

    # MAKE PREDICTION
    predictions = list()

    save_path = os.path.normpath(
        '{}/models/'.format(os.path.dirname(os.path.abspath(__file__)))) + "/"

    for final_model in final_models:
        with open(save_path + final_model['model_path'], 'rb') as fp:
            model = pickle.load(fp)

        with open(save_path + final_model['perf_path'], 'rb') as fp:
            perf = json.load(fp)

        predictor = BabyCryPredictor(model)
        sig_num = 0
        for signal in play_list_processed:
            tmp, confidence = predictor.classify(signal['data'])
            prediction = Prediction(tmp, perf['f1'], confidence)
            predictions.append(prediction)
            signal['preds'].append(prediction)

            final_model['preds'].append(prediction)
            final_model['voter'][sig_num] = {
                'prediction': prediction.prediction,
                'confidence': prediction.confidence
            }

            sig_num += 1

        fm_voter = MajorityVoter(final_model['preds'])
        fm_voter.vote()
        final_model['voter']['ovr'] = fm_voter.get_dictionary()

    majority_voter = MajorityVoter(predictions)
    majority_voter.vote()

    signals_result = []
    for signal in play_list_processed:
        voter = MajorityVoter(signal['preds'])
        voter.vote()
        signals_result.append(voter.get_dictionary())

    return {
        "overall": majority_voter.get_dictionary(),
        "signals": signals_result,
        final_models[0]['name']: final_models[0]['voter'],
        final_models[1]['name']: final_models[1]['voter'],
        final_models[2]['name']: final_models[2]['voter'],
        final_models[3]['name']: final_models[3]['voter'],
    }

import pickle
import re
import librosa
import numpy as np
import json
import os
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

        for offset in range(5):
            audio_data, _ = librosa.load(self.file_name, sr=44100, mono=True, offset=offset, duration=5.0)
            play_list.append(audio_data)

        return play_list


class Prediction:
    def __init__(self, accuracy, prediction):
        self.accuracy = accuracy
        self.prediction = prediction


class MajorityVoter:
    def __init__(self, prediction_list):
        self.predictions = prediction_list

    def vote(self):
        pr_sum = 0
        abs_sum = 0
        for pred in self.predictions:
            pr_sum += pred.prediction
            abs_sum += pred.accuracy

        if pr_sum > abs_sum/2.0:
            return 1
        else:
            return 0


class BabyCryPredictor:
    def __init__(self, model):
        self.model = model

    def classify(self, new_signal):
        category = self.model.predict(new_signal)
        return self._is_baby_cry(category[0])

    @staticmethod
    def _is_baby_cry(string):
        match = re.search(CRYING_BABY_LABEL, string)
        if match:
            return 1
        else:
            return 0


final_models = [
    {'model_path': "svc_model.pkl", 'perf_path': "svc_performance.json"},
    {'model_path': "linsvc_model.pkl", 'perf_path': "linsvc_performance.json"},
    {'model_path': "mlp_model.pkl", 'perf_path': "mlp_performance.json"}
]


def predict(file_name):
    file_reader = Reader(file_name)
    play_list = file_reader.read_audio_file()

    # Feature extraction
    engineer = FeatureEngineer()
    play_list_processed = list()
    for signal in play_list:
        tmp = engineer.feature_engineer(signal)
        play_list_processed.append(tmp)

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
        for signal in play_list_processed:
            tmp = predictor.classify(signal)
            prediction = Prediction(perf['f1'], tmp)
            predictions.append(prediction)

    # MAJORITY VOTE
    majority_voter = MajorityVoter(predictions)
    majority_vote = majority_voter.vote()
    return majority_vote



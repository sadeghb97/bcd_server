from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import numpy as np
import json
import os
import pickle
import sklearn2pmml
from sklearn2pmml import PMMLPipeline
import config


class TrainClassifier:
    SVC_CLASSIFIER = {'id': 1, 'prefix': "svc"}
    LINEAR_SVC_CLASSIFIER = {'id': 2, 'prefix': "linsvc"}
    MLP_CLASSIFIER = {'id': 3, 'prefix': "mlp"}

    def __init__(self, x, y, cc):
        self.X = x
        self.Y = y
        self.current_classifier = cc
        self.perf = None
        self.best_estimator = None

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y,
                                                            test_size=0.25,
                                                            random_state=0,
                                                            stratify=self.Y)

        estimator = None
        if self.current_classifier['id'] == TrainClassifier.SVC_CLASSIFIER['id']:
            pipeline = PMMLPipeline([
                ('scl', StandardScaler()), #dade haro miare tu meghiase sefro yek
                ('clf', SVC(probability=True)) #ye clusteringe kolli ke khas manzure nabude va baraye dade haye ba bode kam estefade mishe.
            ])

            param_grid = [
                {
                    'clf__kernel': ['linear', 'rbf'],
                    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'clf__gamma': np.logspace(-2, 2, 5),
                    # 'lda__n_components': range(2, 17)
                }
            ]

            estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')

        elif self.current_classifier['id'] == TrainClassifier.LINEAR_SVC_CLASSIFIER['id']:
            pipeline = PMMLPipeline([
                ('scl', StandardScaler()),
                ('clf', LinearSVC())
            ])

            param_grid = [{
                    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]
                }]

            estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')

        elif self.current_classifier['id'] == TrainClassifier.MLP_CLASSIFIER['id']:
            classifier = MLPClassifier(
                solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1
            )

            pipeline = PMMLPipeline([
                ('scl', StandardScaler()), #dade haro miare tu meghiase sefro yek
                ('clf', classifier) #ye clusteringe kolli ke khas manzure nabude va baraye dade haye ba bode kam estefade mishe.
            ])

            estimator = pipeline

        model = estimator.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        perf = {'accuracy': accuracy_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred, average='macro'),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'f1': f1_score(y_test, y_pred, average='macro'),
                # 'summary': classification_report(y_test, y_pred)
                }

        print(perf)
        self.perf = perf
        self.best_estimator = model
        if hasattr(model, 'best_estimator_'):
            self.best_estimator = model.best_estimator_

    def save(self):
        save_path = os.path.normpath(
            '{}/models/'.format(os.path.dirname(os.path.abspath(__file__))))
        perf_sp = self.current_classifier['prefix'] + '_performance' + ".json"
        model_sp = self.current_classifier['prefix'] + '_model' + ".pkl"
        pmml_model_sp = self.current_classifier['prefix'] + '_pmml_model' + ".pmml"

        # Save performances
        with open(os.path.join(save_path, perf_sp), 'w') as fp:
            json.dump(self.perf, fp)

        # Save model
        with open(os.path.join(save_path, model_sp), 'wb') as fp:
            pickle.dump(self.best_estimator, fp)

        sklearn2pmml.sklearn2pmml(
            self.best_estimator,
            os.path.join(save_path, pmml_model_sp),
            with_repr=True)


# TRAIN MODEL
X = np.load(config.DERIVED_DATASET_PATH)
Y = np.load(config.DERIVED_LABELS_PATH)

train_classifier = TrainClassifier(
    X, Y,
    TrainClassifier.LINEAR_SVC_CLASSIFIER
)
train_classifier.train()
train_classifier.save()





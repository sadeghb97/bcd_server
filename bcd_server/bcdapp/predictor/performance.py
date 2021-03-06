import re
import os
import prediction

RAW_DATASET_PATH = '/home/sadegh/Desktop/final_project/baby_cry_detection/data/'
CRYING_BABY_RAW_DIR_NAME = '301 - Crying baby'

load_path = RAW_DATASET_PATH
regex = re.compile(r'^[0-9]')
directory_list = [i for i in os.listdir(load_path) if regex.search(i)]

all_records = 0
correct_predictions = 0

for directory in directory_list:
    file_list = os.listdir(os.path.join(load_path, directory))

    for audio_file in file_list:
        file_name = os.path.join(load_path, directory, audio_file)
        pred = prediction.predict(file_name)['overall']['prediction']

        all_records += 1
        if directory == CRYING_BABY_RAW_DIR_NAME:
            if pred == 1:
                correct_predictions += 1
        else:
            if pred == 0:
                correct_predictions += 1

        print(str(correct_predictions) + "/" + str(all_records) +
              " (" + str(correct_predictions / all_records) + ")")


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
from PIL import Image

root_dir = "D:\\Github\\Final-year-project\\Datasets\\EMOTIW-Audio_sets"
output_dir = "D:\\Github\\Final-year-project\\Datasets\\EMOTIW-Mel_spectrum"

def preprocess():
    for set in ['train','val']:
        set_path = os.path.join(root_dir, set)
        for catg in os.listdir(set_path):
            catg_path = os.path.join(set_path,catg)
            video_files = [name for name in os.listdir(catg_path)]

            train_dir = os.path.join(output_dir, 'train', catg)
            val_dir = os.path.join(output_dir, 'val', catg)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)

            for i in video_files:
                video_path = os.path.join(root_dir,set,catg,i)
                print(video_path)
                file_name = i.split('.')[0]
                samples, sample_rate = librosa.load(video_path)
                melspectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=2048, hop_length=1024,
                                                                n_mels=60)
                logmelspec = librosa.power_to_db(melspectrogram)
                mfccs = sklearn.preprocessing.scale(logmelspec, axis=1)
                plt.figure(figsize=(12, 4))
                librosa.display.specshow(mfccs, sr=sample_rate)
                plt.savefig(os.path.join(output_dir,set,catg,file_name +'.jpg'))
                plt.close()
                img = Image.open(os.path.join(output_dir,set,catg,file_name +'.jpg'))
                new_img = img.resize((224, 224), Image.BILINEAR)
                new_img.save(os.path.join(output_dir, set, catg, file_name + '.jpg'))
    print('Preprocessing finished.')

preprocess()








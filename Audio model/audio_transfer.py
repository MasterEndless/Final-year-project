from moviepy.editor import *
import os

root_dir = "D:\\Github\\Final-year-project\\Datasets\\EMOTIW"
output_dir = "D:\\Github\\Final-year-project\\Datasets\\EMOTIW-Audio_sets"

def preprocess():
    for set in os.listdir(root_dir):
        set_path = os.path.join(root_dir, set)
        for catg in os.listdir(set_path):
            catg_path = os.path.join(set_path,catg)
            print(catg_path)
            video_files = [name for name in os.listdir(catg_path)]

            train_dir = os.path.join(output_dir, 'train', catg)
            val_dir = os.path.join(output_dir, 'val', catg)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)

            for i in video_files:
                video_path = os.path.join(root_dir,set,catg,i)
                file_name = i.split('.')[0]
                video = VideoFileClip(video_path)
                audio = video.audio
                audio.write_audiofile(os.path.join(output_dir,set,catg,file_name +'.wav'))
                video.reader.close()
                video.audio.reader.close_proc()
    print('Preprocessing finished.')

preprocess()

import os
from sklearn.model_selection import train_test_split
import cv2

root_dir= 'D:/Github/Final-year-project/Datasets/CAER-1'
save_dir='D:/Github/Final-year-project/Datasets/CAER-2'

resize_height = 256
resize_width = 342


def preprocess():
    # Split train/val/test sets
    for set in os.listdir(root_dir):
        set_path = os.path.join(root_dir, set)
        for catg in os.listdir(set_path):
            catg_path = os.path.join(set_path,catg)
            video_files = [name for name in os.listdir(catg_path)]
            for video in video_files:
                process_video(video, set, catg)

    print('Preprocessing finished.')

def process_video(video, set_name, catg_name):
    # Initialize a VideoCapture object to read video data into a numpy array
    video_filename = video.split('.')[0]
    video_filename = set_name+'_'+catg_name+'_'+video_filename
    if not os.path.exists(os.path.join(save_dir, video_filename)):
        os.mkdir(os.path.join(save_dir, video_filename))
    capture = cv2.VideoCapture(os.path.join(root_dir, set_name, catg_name, video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Make sure splited video has at least 16 frames
    if frame_count > 28:
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 28:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 28:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 28:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue
            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

preprocess()
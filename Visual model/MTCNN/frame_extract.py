
import os
from sklearn.model_selection import train_test_split
import cv2
import Face_capture

root_dir= 'D:\\Github\\Final-year-project\\Datasets\\CAER'
output_dir='D:\\Github\\Final-year-project\\Datasets\\CAER_face'

resize_height = 128
resize_width = 171
crop_size = 112

def preprocess():
    # Split train/val/test sets
    set_path = os.path.join(root_dir, 'val')
    for catg in os.listdir(set_path):
        val_dir = os.path.join(output_dir, 'val', catg)
        catg_path = os.path.join(root_dir,'val',catg)
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)

        video_files = [name for name in os.listdir(catg_path)]
        for video in video_files:
            process_video(video, 'val', catg)
            print(os.path.join(output_dir,'val',catg,video))

    print('Preprocessing finished.')

def process_video(video, set, catg):
    # Initialize a VideoCapture object to read video data into a numpy array
    video_filename = video.split('.')[0]
    if not os.path.exists(os.path.join(output_dir, set, catg, video_filename)):
        os.mkdir(os.path.join(output_dir, set, catg, video_filename))
    capture = cv2.VideoCapture(os.path.join(root_dir, set, catg, video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Make sure splited video has at least 16 frames
    EXTRACT_FREQUENCY = 4
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
    count = 0
    i = 0
    retaining = True
    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue
        if count % EXTRACT_FREQUENCY == 0:
            index, img = Face_capture.face_capture(frame)
            if index == 1:
                cv2.imwrite(filename=os.path.join(output_dir, set, catg, video_filename, '0000{}.jpg'.format(str(i))), img=img)
                i += 1
        count += 1

    # Release the VideoCapture once it is no longer needed
    capture.release()

preprocess()
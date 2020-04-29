import cv2
import Face_capture
import os

capture = cv2.VideoCapture('D:\\Github\\Final-year-project\\MTCNN_face_detection\\Test\\01196.avi')
count=0
frame_count=50
retaining=True
resize_height = 128
resize_width = 171
i=0
save_dir = 'D:\\Github\\Final-year-project\\MTCNN_face_detection\\Test'
video_filename = '01196'

while (count < frame_count and retaining):
    retaining, frame = capture.read()
    if frame is None:
        continue
    index, img = Face_capture.face_capture(frame)
    print(index)
    if index == 1:
        frame = cv2.resize(img, (resize_width, resize_height))
        cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
        i += 1
    count += 1

        # Release the VideoCapture once it is no longer needed
capture.release()
import os
import shutil


root_dir = 'D:\\Github\\Final-year-project\\Datasets\\EMOTIW_face'
output_dir = 'D:\\Github\\Final-year-project\\Datasets\\EMOTIW_face_full'


count = 0
count_list = []
for set in os.listdir(root_dir):
    set_path = os.path.join(root_dir, set)
    for catg in os.listdir(set_path):
        catg_path = os.path.join(set_path,catg)
        catg_out_path = os.path.join(output_dir,set,catg)
        if not os.path.exists(catg_out_path):
            os.mkdir(catg_out_path)
        for video in os.listdir(catg_path):
            video_path = os.path.join(catg_path,video)
            video_out_path = os.path.join(catg_out_path,video)
            if not os.path.exists(video_out_path):
                os.mkdir(video_out_path)

            for img in os.listdir(video_path):
                img_path = os.path.join(video_path,img)
                count = count + 1
            new_path = os.path.join(output_dir,set,catg,video)
            count_2 = count
            for i in range(17-count):
                shutil.copy2(img_path, new_path)
                new_img = os.path.join(new_path,img)
                os.rename(new_img,os.path.join(new_path, '0000{}.jpg'.format(str(count_2))))
                count_2 = count_2 + 1


            '''
            for img in os.listdir(video_path):
                img_path = os.path.join(video_path,img)
                shutil.move(img_path, video_out_path)
            '''

            count = 0

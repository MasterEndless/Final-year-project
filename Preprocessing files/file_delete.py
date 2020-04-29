import os

root_dir = 'D:\\Github\\Final-year-project\\Datasets\\full_face_resize'
count = 0
for set in os.listdir(root_dir):
    set_path = os.path.join(root_dir, set)
    for catg in os.listdir(set_path):
        catg_path = os.path.join(set_path,catg)
        for video in os.listdir(catg_path):
            video_path = os.path.join(catg_path,video)
            for img in os.listdir(video_path):
                count = count + 1
            if count < 17:
                for img in os.listdir(video_path):
                    os.remove(os.path.join(video_path,img))
                os.rmdir(video_path)
            count = 0 

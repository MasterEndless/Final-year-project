import os

root_dir = 'D:\\Github\\Final-year-project\\Datasets\\full_face_resize'
count = 0
count_list = []
for set in os.listdir(root_dir):
    set_path = os.path.join(root_dir, set)
    for catg in os.listdir(set_path):
        catg_path = os.path.join(set_path,catg)
        for video in os.listdir(catg_path):
            video_path = os.path.join(catg_path,video)
            for img in os.listdir(video_path):
                count = count + 1
            count_list.append(count)
            count = 0
print(sorted(count_list))
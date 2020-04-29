from PIL import Image
import os


root_dir = 'D:\\Github\\Final-year-project\\Datasets\\Full_face'
output_dir = 'D:\\Github\\Final-year-project\\Datasets\\full_face_resize'


for set in os.listdir(root_dir):
    set_path = os.path.join(root_dir, set)
    for catg in os.listdir(set_path):
        catg_path = os.path.join(set_path,catg)
        if not os.path.exists(os.path.join(output_dir,set,catg)):
                os.mkdir(os.path.join(output_dir,set,catg))
        for video in os.listdir(catg_path):
            video_path = os.path.join(catg_path,video)
            print(video_path)
            if not os.path.exists(os.path.join(output_dir,set,catg,video)):
                os.mkdir(os.path.join(output_dir,set,catg,video))
            for img in os.listdir(video_path):
                img_filename = img.split('.')[0]
                img_path = os.path.join(root_dir,set,catg,video,img)
                try:
                    img=Image.open(img_path)
                    new_img=img.resize((171,128),Image.BILINEAR)   
                    new_img.save(os.path.join(output_dir,set,catg,video,img_filename+'.jpg'))
                except Exception as e:
                    pass
                continue
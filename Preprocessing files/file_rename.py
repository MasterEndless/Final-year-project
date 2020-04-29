import os
#Sequence:

emo_list = ['Anger','Disgust','Fear','Happy','Neutral','Sad','Surprise']
set_list=['train','validation','test']
input_source_path = 'D:/Github/Final-year-project/Datasets/CAER_old/CAER10/'
output_source_path='D:\\Github\\Final-year-project\\Datasets\\CAER_video'

count=[5191, 2778, 1993, 12580, 28414, 10733, 11960]   #follow exactly the sequence of emo_list
for set in set_list:
    index = 0
    for emo in emo_list:
        input_path=input_source_path+set+'/'+emo+'/'
        f = os.listdir(input_path)
        n=0
        for i in f:
            oldname=input_path+f[n]
            count_str=str(count[index]+1)
            count_zf=count_str.zfill(5)
            newname = output_source_path +emo+'/'+ count_zf + '.avi'
            n+=1
            count[index]+=1
            os.rename(oldname, newname)
            print(newname)
        index += 1
print(count)







#f=os.listdir(path)

'''
n=0
for i in f:

    #设置旧文件名（就是路径+文件名）
    oldname=path+f[n]

    #设置新文件名
    newname=path+'机场'+str(n+1)+'.jpg'

    #用os模块中的rename方法对文件改名
    os.rename(oldname,newname)
    print(oldname,'======>',newname)

    n+=1
'''
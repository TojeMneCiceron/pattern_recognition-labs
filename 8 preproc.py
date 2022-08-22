import numpy as np
import os
import cv2
import shutil
import pprint
import time

time_ = time.time()

dir_train = r"C:\Users\Пользователь\python\pics_dataset\train"
dir_test = r"C:\Users\Пользователь\python\pics_dataset\test"
N = 128


def Read(dir_name):
    dict = {}
    for root, dirs, files in os.walk(dir_name, topdown = False):
        for name in files:
            # a = root[-1] in dict
            if not root[-1] in dict:
                dict[root[-1]] = []
            dict[root[-1]] += [(root, name)]
    # pprint.pprint(dict)
    return dict

# os.path.join(root, name)

def Open_image(im_name):
    image = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
    return image

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Remove_transparency(image):
    if len(image.shape) == 3:
        channels = image.shape[-1]

    if channels == 4:
        trans_mask = image[:,:,3] == 0        
        image[trans_mask] = [250, 250, 250, 255]
    return image

def Preproc_image(image, N):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, image = cv2.threshold(image, 127, 255, 0)
    a, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if image[0][0] == 0:
        image = 255 - image
    # viewImage(image, "heh")

    points = np.argwhere(image==0)
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points) 
    image = image[y:y+h, x:x+w]
    # viewImage(image, "heh")

    bordersize = int((max(w, h) - min(w, h))/2)

    tb = 0
    lr = 0
    if max(w, h) == h:
        lr = bordersize
    else:
        tb = bordersize

    image = cv2.copyMakeBorder(
        image,
        top=tb,
        bottom=tb,
        left=lr,
        right=lr,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    # print(image.shape)
    # viewImage(image, "oww")
    
    image = cv2.resize(image, (N, N), interpolation = cv2.INTER_AREA)
    # print(image.shape)
    # viewImage(image, "wow")

    return image
    # os.chdir(r"C:\Users\Пользователь\Desktop")
    # save = 'a.png'
    # print(cv2.imwrite(save, image))

def Preproc_n_Save(dict, dir_name, N):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for key in dict.keys():
        dir_key = os.path.join(dir_name, key)
        if not os.path.exists(dir_key):
            os.mkdir(dir_key)

        images = dict[key]

        for i in range(len(images)):
            # print('===================================')
            # print(images[i])
            os.chdir(images[i][0])
            image = Open_image(images[i][1])
            # print(os.path.join(images[i][0], images[i][1]))
            image = Remove_transparency(image)
            image = Preproc_image(image, N)

            os.chdir(dir_key)
            cv2.imwrite(str(i) + ".png", image)
                

prep_dir = r"C:\Users\Пользователь\Desktop\preprocessed_pics"
if os.path.exists(prep_dir):
    shutil.rmtree(prep_dir, ignore_errors=True)
os.mkdir(prep_dir)

dict_train = Read(dir_train)
dict_test = Read(dir_test)

Preproc_n_Save(dict_train, os.path.join(prep_dir, "train"), N)
Preproc_n_Save(dict_test, os.path.join(prep_dir, "test"), N)

print("--- %s seconds ---" % (time.time() - time_))

# norm = r"C:\Users\Пользователь\python\pics_dataset\train\45e13cda-3651-4208-8cf1-6b49a0c5e859\5\_0012_Слой-18.png"
# ue = r"C:\Users\Пользователь\python\pics_dataset\train\84029aea-b710-4455-b14a-e099050bf556\0\Baoh1.png"
# hm = r"C:\Users\Пользователь\Desktop\uebskii dataset\train\84029aea-b710-4455-b14a-e099050bf556\3\Stardust Crusaders.png"

# aaaaaaaa = r"C:\Users\Пользователь\python\pics_dataset\train\45e13cda-3651-4208-8cf1-6b49a0c5e859\1"

# # os.chdir(aaaaaaaa)

# image = cv2.imread('_0024_Sloi 6.jpg', -1)
# # image = Preproc_image(image, N)
# viewImage(image, "ubeite")
# image = Remove_transparency(image)
# viewImage(image, "ubeite")
# image = Preproc_image(image, N)
# viewImage(image, "ubeite")


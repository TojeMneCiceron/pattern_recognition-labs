import time
import numpy as np
import cv2
import os
import json

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Potential_Method:
    def __init__(self):
        self.N = 0
        self.model = {}
        self.model_path = r"C:\Users\Пользователь\Desktop\models"
        self.model_file_name = "model.json"
        
    def Open_image(self, im_name):
        image = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
        return image

    def Remove_transparency(self, image):
        if len(image.shape) == 3:
            channels = image.shape[-1]

        if channels == 4:
            trans_mask = image[:,:,3] == 0        
            image[trans_mask] = [250, 250, 250, 255]
        return image

    def Preproc_image(self, image, N):
        image = self.Remove_transparency(image)

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
            value=[255, 255, 255])    
        image = cv2.resize(image, (N, N), interpolation = cv2.INTER_AREA)
        return image

    def P(self, image):
        self.N = image.shape[0]

        card = np.zeros(image.shape) 
        a, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        for x,y in np.ndindex(card.shape):
            if image[x,y]==0:
                card[x,y] += 1

            if x-1 >= 0 and image[x-1,y] == 0:
                card[x,y] += 1/6
            if y+1 < self.N and image[x,y+1] == 0:
                card[x,y] += 1/6
            if x+1 < self.N and image[x+1,y] == 0:
                card[x,y] += 1/6
            if y-1 >= 0 and image[x,y-1] == 0:
                card[x,y] += 1/6

            if x-1 >= 0 and y-1 >= 0 and image[x-1,y-1] == 0:
                card[x,y] += 1/12
            if x-1 >= 0 and y+1 < self.N and image[x-1,y+1] == 0:
                card[x,y] += 1/12
            if x+1 < self.N and y+1 < self.N and image[x+1,y+1] == 0:
                card[x,y] += 1/12
            if x+1 < self.N and y-1 >= 0 and image[x+1,y-1] == 0:
                card[x,y] += 1/12
        return card
 
    def Dist(self, card1, card2):
        return np.sum(np.abs(card1 - card2))

    def Read(self, dir_name):
        dict = {}
        for root, dirs, files in os.walk(dir_name, topdown = False):
            for name in files:
                # a = root[-1] in dict
                if not root[-1] in dict:
                    dict[root[-1]] = []
                dict[root[-1]] += [(root, name)]
        # pprint.pprint(dict)
        return dict

    def Fit(self, dir_name):
        # print(dir_name)
        im_dict = self.Read(dir_name)

        self.model = {}

        for key in im_dict.keys():
            images = im_dict[key]

            for i in range(len(images)):

                if not key in self.model:
                    self.model[key] = []

                os.chdir(images[i][0])
                image = self.Open_image(images[i][1])
                
                card = self.P(image)
                
                self.model[key] += [card.tolist()]
        self.Save_model()
        return True

    def Save_model(self):
        
        os.chdir(self.model_path)
        with open(self.model_file_name, "w") as write_file:
            json.dump(self.model, write_file)

        print('Model is saved here: ' + os.path.join(self.model_path, self.model_file_name))

    def Load_model(self):
        
        os.chdir(self.model_path)
        with open("model.json", "r") as read_file:
            self.model = json.load(read_file)

        return 0

    def Predict(self, dir_name, file_name):
        self.Load_model()

        os.chdir(dir_name)

        a = self.model['0']
        self.N = len(a[0][0])

        image = self.Open_image(file_name)
        image = self.Preproc_image(image, self.N)
        min_key = ''
        min_dist = -1

        # viewImage(image, "a")
        goal_card = self.P(image)

        for key in self.model.keys():        
            cards = self.model[key]
            
            for card in cards:
                dist = self.Dist(card, goal_card)
                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
                    min_key = key

        return min_key

pm = Potential_Method()

# ============================================================================================================

# time_ = time.time()

# # input fit
# fit_dir_name = r"C:\Users\Пользователь\Desktop\preprocessed_pics\train"

# print('Fitting...')
# pm.Fit(fit_dir_name)

# print("--- %s seconds ---" % (time.time() - time_))

# ============================================================================================================

time_ = time.time()

# input predict
dir_name = r"C:\Users\Пользователь\Desktop\pics_dataset\test\84029aea-b710-4455-b14a-e099050bf556\7"
file_name = r"Steel Ball Run2.png"
# N = 128

print('Predicting...')

res = pm.Predict(dir_name, file_name)
print(res)

print("--- %s seconds ---" % (time.time() - time_))
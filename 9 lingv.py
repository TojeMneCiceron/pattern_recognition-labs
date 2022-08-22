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

class Lingv_Method:
    def __init__(self):
        self.N = 0
        self.model = {}
        self.directions = { "1" : (-1, 0), "2" : (-1, 1), "3" : (0, 1), "4" : (1, 1), "5" : (1, 0), "6" : (1, -1), "7" : (0, -1), "8" : (-1, -1) }
        self.model_path = r"C:\Users\Пользователь\Desktop\models"
        self.model_file_name = "lingv_model.json"
        # self.cur_sign = ""
        # self.cur_image = [[]]

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

    def Find_neighbours(self, image, i, j):
        # first = "0"
        nbours = []

        for key in self.directions.keys():
            x = i + self.directions[key][0]
            y = j + self.directions[key][1]

            if x >= 0 and y >= 0 and x < self.N and y < self.N and image[x, y] == 0:
                nbours += [key]
        return nbours

    def First_nbour(self, image, i, j):
        for key, dir in self.directions.items():
            x = i + dir[0]
            y = j + dir[1]

            if x >= 0 and y >= 0 and x < self.N and y < self.N and image[x, y] == 0:
                return key
        return "0"

    def Nbours_count(self, image, i, j):
        count = 0
        for key, dir in self.directions.items():
            x = i + dir[0]
            y = j + dir[1]

            if x >= 0 and y >= 0 and x < self.N and y < self.N and image[x, y] == 0:
                count += 1
        return count

    def Start_point(self, image):

        for i in range(self.N - 1, 0, -1):
            for j in range(self.N):
                if image[i,j] == 0:
                    return i, j
        return i, j


    def DFS(self, image, i, j):
        stack = [] # stack of points
        stack += [(i, j, "0", False)]
        a_stack = []
        image[i,j] = 255

        sign = "("

        while len(stack) > 0:
            cur_point = stack.pop()
            nbours = self.Find_neighbours(image, cur_point[0], cur_point[1])

            if len(nbours) == 0:
                continue

            first = nbours[0]
            sign += cur_point[2]
            if cur_point[3]:
                sign += ")("
            # x = self.directions[first][0] + i
            # y = self.directions[first][1] + j
            # stack.append((x,y, first, ))

            # print(cur_point)
            # print(len(nbours))
            # print(nbours)
            # print(reversed())
            for nbour in reversed(nbours):
                x = self.directions[nbour][0] + cur_point[0]
                y = self.directions[nbour][1] + cur_point[1]
                image[x,y] = 255
                # self.cur_sign += "("
                # self.cur_sign += nbour
                # self.cur_sign += ")"
                stack.append((x,y, nbour, (not nbour == first)))
        if sign[:-1] != '(':
            return sign + ')'
        return sign[:-1]

    def Normalize(self, sign):
        sign = sign.replace('0', '')
        # print(sign)
        temp = sign[0]
        for i in range(1, len(sign)):
            if sign[i] != temp[-1]:
                temp += sign[i]

        old = sign
        res = ""
        while old != res:
            old = temp
            temp = temp.replace('323', '2')
            temp = temp.replace('232', '2')
            temp = temp.replace('121', '2')
            temp = temp.replace('212', '2')
            temp = temp.replace('343', '4')
            temp = temp.replace('434', '4')
            temp = temp.replace('545', '4')
            temp = temp.replace('454', '4')
            temp = temp.replace('565', '6')
            temp = temp.replace('656', '6')
            temp = temp.replace('767', '6')
            temp = temp.replace('676', '6')
            temp = temp.replace('787', '8')
            temp = temp.replace('878', '8')
            temp = temp.replace('181', '8')
            temp = temp.replace('818', '8')

            res = temp[0]
            for i in range(1, len(temp)):
                if temp[i] != res[-1]:
                    res += temp[i]

            if old == res:
                break
            else:
                old = res
        return res

    def LiteralSign(self, sign):
        res = sign
        res = res.replace('()','')
        res = res.replace('(','+(')
        res = res.replace('1','+a')
        res = res.replace('2','+c')
        res = res.replace('3','+b')
        res = res.replace('4','+d')
        res = res.replace('5','-a')
        res = res.replace('6','-c')
        res = res.replace('7','-b')
        res = res.replace('8','-d')
        res = res.replace('(+','(')
        return res[1:]

    def Sign(self, image):
        self.N = image.shape[0]
        a, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        viewImage(image, "j")
        x, y = self.Start_point(image)
        # print((x, y))
        # image[x,y] = 120
        # viewImage(image, "dsf")
        sign = self.DFS(image, x, y)
        sign = self.Normalize(sign)
        # print(sign)
        sign = self.LiteralSign(sign)

        return sign

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
                # print(images[i][1])
                # print(key)
                sign = self.Sign(image)

                self.model[key] += [sign]
        self.Save_model()
        return True

    def Save_model(self):

        os.chdir(self.model_path)
        with open(self.model_file_name, "w") as write_file:
            json.dump(self.model, write_file)

        print('Model is saved here: ' + os.path.join(self.model_path, self.model_file_name))

    def Load_model(self):

        os.chdir(self.model_path)
        with open(self.model_file_name, "r") as read_file:
            self.model = json.load(read_file)

        return 0

    def Predict(self, dir_name, file_name):
        self.Load_model()

        os.chdir(dir_name)

        # a = self.model['0']
        self.N = 128

        image = self.Open_image(file_name)
        viewImage(image, "sd")
        # print(self.N)
        image = self.Preproc_image(image, self.N)
        viewImage(image, "sd")
        goal_sign = self.Sign(image)
        # print(goal_sign)
        for key in self.model.keys():
            signs = self.model[key]

            for sign in signs:
                if sign == goal_sign:
                    return key

        return "Не удалось распознать символ"

lm = Lingv_Method()

# time_ = time.time()
# # input fit
# fit_dir_name = r"C:\Users\Пользователь\Desktop\preprocessed_pics\train"

# print('Fitting...')
# lm.Fit(fit_dir_name)

# print("--- %s seconds ---" % (time.time() - time_))

# # ======================================================================================================================

time_ = time.time()
# input predict
dir_name = r"C:\Users\Пользователь\Desktop\pics_dataset\test\45e13cda-3651-4208-8cf1-6b49a0c5e859\C"
file_name = r"_0070_8.jpg"
# N = 128

print('Predicting...')

res = lm.Predict(dir_name, file_name)
print(res)
# print(lm.model['C'])

print("--- %s seconds ---" % (time.time() - time_))

# ======================================================================================================================


# dir_name = r"C:\Users\Пользователь\Desktop\preprocessed_pics\train\Z"
# file_name = r"13.png"

# os.chdir(dir_name)
# image = lm.Open_image(file_name)
# # image = lm.Preproc_image(image, 128)
# # viewImage(image, "j")
# print("Signing...")
# sign = lm.Sign(image)
# print(sign)
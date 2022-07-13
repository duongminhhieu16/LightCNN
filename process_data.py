import os
import cv2

train_path = "/home/jovyan/vggface2_train/"
test_path = "/home/jovyan/vggface2_test/"

list2 = os.listdir(train_path)
list_ = []
for i in list2:
    list_.append(int(i))
list_ = sorted(list_) # sort dataset 
list1 = list_[:]


train = []
test = []
# val = []
idx = 0
for file in list1:
    idx = 0
    dirr = os.listdir(train_path+str(file))
    for p in dirr:
        if idx < len(dirr)*9/10:
            train.append(train_path + str(file) + "/" + p)
        else:
            test.append(train_path + str(file) + "/" + p)
        idx += 1
        print(idx)

tr = []
t = []
idx = 0

for i in train:
    idx += 1
    print(idx)
    img = cv2.imread(i)
    if img.shape[0] > 128 and img.shape[1] > 128: # exclude the images that have shape < 128x128
        tr.append(i)
print("There are {0} images in the train set".format(str(idx)))

idx = 0
for i in test:
    idx +=1
    img = cv2.imread(i)
    if img.shape[0] > 128 and img.shape[1] > 128:
        t.append(i)
print("There are {0} images in the test set".format(str(idx)))

# write train list and test list to file text
with open('/home/jovyan/LightCNN-master/train_list.txt', 'w') as file:
    for p in tr:
        file.write(p + "\n")

with open('/home/jovyan/LightCNN-master/test_list.txt', 'w') as file:
    for p in t:
        file.write(p + "\n")

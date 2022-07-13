import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path

def default_loader(path):
    img = Image.open(path).convert('L')
    return img

def default_list_reader(fileList):
    imgList = []
    cnt = 0
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath = line.strip().split(' ')[0]
            label = line.strip().split('/')[4] # cnt
            cnt = cnt + 1
            imgList.append((imgPath, int(label))) #-1
            print(cnt)
    return imgList
    # imgList = []
    # idx = 0
    # a = []
    # err = []
    # with open(fileList, 'r') as file:
    #     tmp = ""
    #     label = "b"
    #     for line in file.readlines():
    #         imgPath = line.strip().split(' ')
    #         try:
    #             img = Image.open(imgPath[0]) # open the image file
    #             img.verify() # verify that it is, in fact an image
    #         except (IOError, SyntaxError) as e: # print out the names of corrupt files
    #             err.append(imgPath[0])
    #             continue
    #         arr = line.split("/")[6].split(".")[0].split("_")
    #         for i in range(len(arr)-1):
    #             tmp += arr[i]
    #             if i < len(arr)-2:
    #                 tmp += "_"
    #         if label != tmp:
    #             label = tmp
    #             idx += 1
    #         a.append(idx)
    #         imgList.append((imgPath[0], idx))
    #         tmp = ""
    #     # with open("/home/vkistuser/LightCNN-master/t.txt", 'w') as file:
    #     #     for p in a:
    #     #         file.write(str(p) + "\n")
    #     with open("/home/vkistuser/LightCNN-master/corrupted_images.txt", 'w') as file:
    #         for p in err:
    #             file.write(p + "\n")
    # return imgList
class DataPrefetcher():
    def __init__(self, dataloader, img_shape, device):
        self.dataloader = dataloader
        self._len = len(dataloader)
        self.device = device
        torch.cuda.device(device)
        self.stream = torch.cuda.Stream()
        self.img_shape = img_shape

    def prefetch(self):
        try:
            self.next_image, self.next_label = next(self.dl_iter)
        except StopIteration:
            self.next_image = None
            self.next_label = None
            return
        with torch.cuda.stream(self.stream):
            self.next_label = self.next_label.to(self.device, non_blocking=True)
            self.next_image = self.next_image.to(self.device, non_blocking=True)

            self.next_image = self.next_image.float()
            self.next_image = torch.nn.functional.interpolate(
                input=self.next_image,
                size=self.img_shape,
                mode="trilinear",
                align_corners=False,
                )

    def __iter__(self):
        self.dl_iter = iter(self.dataloader)
        self.prefetch()
        return self

    def __len__(self):
        return self._len

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image = self.next_image
        label = self.next_label

        if image is None or label is None:
            raise StopIteration

        image.record_stream(torch.cuda.current_stream())
        label.record_stream(torch.cuda.current_stream())
        self.prefetch()
        return image, label

class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

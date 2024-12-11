import torch
from torchvision.datasets import VOCDetection
from pprint import pprint
from torchvision.transforms import ToTensor
class VOCDataset(VOCDetection):
    def __init__(self,root,year,image_set,download,transform):
        super().__init__(root,year,image_set,download,transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
    def __getitem__(self, item):
        image,data = super().__getitem__(item)
        all_boxes = []
        all_labels = []
        for obj in data["annotation"]["object"]:
            x_min = int(obj["bndbox"]["xmin"])
            y_min = int(obj["bndbox"]["ymin"])
            x_max = int(obj["bndbox"]["xmax"])
            y_max = int(obj["bndbox"]["ymax"])
            all_boxes.append([x_min,y_min,x_max,y_max])
            all_labels.append(self.categories.index(obj["name"]))
        all_boxes = torch.FloatTensor(all_boxes)
        all_labels = torch.LongTensor(all_labels)
        target = {
            "boxes":all_boxes,
            "labels": all_labels
        }

        return image,target
if __name__ == '__main__':
    transform = ToTensor()
    dataset = VOCDataset(root="./myVOC",year = "2012",image_set = "train",download = False,transform = transform)  #download = True if you haven't download VOC dataset
    image,target = dataset[2000]
    print(image.shape)
    print(target)
    # pprint(target["annotation"]["object"])
    # dataset = VOCDetection(root="./myVOC",year = "2012",image_set = "train",download = False,transform = None)
    # image, label = dataset[2000]
    # pprint(label)
    # # image.show()



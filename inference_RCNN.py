import os
import numpy as np
import cv2
import argparse
from pprint import pprint
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def get_args():
    parser = argparse.ArgumentParser(description="Train faster rcnn model")
    # parser.add_argument("--image_path", "-i", type=str, required=True)
    parser.add_argument("--year", "-y", type=str, default="2012")
    parser.add_argument("--conf_threshold", "-c", type=float, default=0.3)
    parser.add_argument("--saved_checkpoint", "-o", type=str, default="trained_models/best.pt")
    parser.add_argument("--video_path", "-v", type=str, required=True)

    args = parser.parse_args()
    return args
def train(args):
    categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                  'train', 'tvmonitor']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels,num_classes=21)
    checkpoint = torch.load(args.saved_checkpoint,map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.float()
    # IMAGE
    # ori_image = cv2.imread(args.image_path)
    # image = cv2.cvtColor(ori_image,cv2.COLOR_BGR2RGB)
    # image = np.transpose(ori_image,(2,0,1))/255.
    # image = [torch.from_numpy(image).float()]
    # model.eval()
    # with torch.no_grad():
    #     output = model(image)[0]
    #     bboxes = output["boxes"]
    #     labels = output["labels"]
    #     scores = output["scores"]
    #     for bbox,label,score in zip(bboxes,labels,scores):
    #         if score > args.conf_threshold:
    #             x_min, y_min,x_max,y_max = bbox
    #             cv2.rectangle(ori_image,(int(x_min),int(y_min),int(x_max),int(y_max)),(0,0,255),4)
    #             category = categories[label]
    #             cv2.putText(ori_image, category, (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX,
    #                         3, (0, 255, 0), 3, cv2.LINE_AA)
    #     cv2.imwrite("prediction.jpg",ori_image)

    # VIDEO
    cap = cv2.VideoCapture(args.video_path)
    while cap.isOpened():
        _,frame = cap.read()
        if not _:
            break
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = np.transpose(image,(2,0,1))/255.
        image = [torch.from_numpy(image).float()]
        model.eval()
        with torch.no_grad():
            output = model(image)[0]
            bboxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]
            for bbox,label,score in zip(bboxes,labels,scores):
                if score > args.conf_threshold:
                    x_min, y_min,x_max,y_max = bbox
                    cv2.rectangle(frame,(int(x_min),int(y_min),int(x_max),int(y_max)),(0,0,255),4)
                    category = categories[label]
                    cv2.putText(frame, category, (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX,
                                3, (0, 255, 0), 3, cv2.LINE_AA)
            re_frame = cv2.resize(frame,(600 ,550))
            cv2.imshow("Result",re_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = get_args()
    train(args)

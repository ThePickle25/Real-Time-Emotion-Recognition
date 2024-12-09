from __future__ import print_function
import argparse
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from data_loaders import Plain_Dataset, eval_data_dataloader
from detect_face import detect
from models.experimental import attempt_load

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from transformers import ViTForImageClassification, TrainerCallback

classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
id2label = {id: label for id, label in enumerate(classes)}
label2id = {label: id for id, label in id2label.items()}


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

yolov5 = load_model('Yolov5/weights/yolov5n-0.5.pt', device)

best_model_path = "best_emotion_model"
model = ViTForImageClassification.from_pretrained(best_model_path)
model.to(device)
model.eval()


def load_img(path):
    img = Image.open(path)
    img = transformation(img).float()
    img = torch.autograd.Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    return img.to(device)


if True:

    # To capture video from webcam.
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frame
        _, img = cap.read()
        # Detect the faces
        # apply yolov5 model to detect face
        faces = detect(yolov5, img, device)
        if faces != None:
            # Draw the rectangle around each face
            for (x1, y1, x2, y2) in faces:
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                roi = img[y1:y2, x1:x2]
                roi = cv2.resize(roi, (224, 224))
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                # roi = cv2.resize(roi, (48, 48))
                cv2.imwrite(f"Webcam/roi.jpg", roi)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                imgg = load_img(f"Webcam/roi.jpg")
                outputs = model(imgg)
                predicted_class = outputs.logits.argmax(dim=1).item()
                predicted_label = id2label[predicted_class]

                # wrong = torch.where(classs != 3, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
                # classs = torch.argmax(pred, 1)
                # prediction = classes[classs.item()]

                img = cv2.putText(img, predicted_label, (x1 + org[0], y1 + org[1]), font,
                                  fontScale, color, thickness, cv2.LINE_AA)
        else:
            img = cv2.putText(img, "Nobody face is detected", org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('img', img)
        # Stop if (Q) key is pressed
        k = cv2.waitKey(30)
        if k == ord("q"):
            break

    # Release the VideoCapture object
    cap.release()

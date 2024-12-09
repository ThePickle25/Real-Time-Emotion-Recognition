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
from deep_emotion import Deep_Emotion
from data_loaders import Plain_Dataset, eval_data_dataloader
from detect_face import  detect
from models.experimental import attempt_load
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0")
# print(device)


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


parser = argparse.ArgumentParser(description="Configuration of testing process")
parser.add_argument('-d', '--data', type=str, required=True,
                    help='Folder that contains the finaltest.csv and test images')
parser.add_argument('-m', '--model', type=str, required=True, help='Path to pretrained model')
parser.add_argument('-t', '--test_acc', type=bool,default='', help='Only show test accuarcy')
parser.add_argument('-c', '--cam', type=bool,default=True, help='Test the model in real time with webcam connect via usb')
args = parser.parse_args()

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = Plain_Dataset(csv_file=args.data + '/test.csv', img_dir=args.data + '/' + 'test/', datatype='test',
                        transform=transformation)
test_loader = DataLoader(dataset, batch_size=64, num_workers=0)

yolov5 = load_model('Yolov5/weights/yolov5n-0.5.pt', device)

net = Deep_Emotion()
print("Deep Emotion:-", net)
net.load_state_dict(torch.load(args.model, map_location=torch.device('cuda:0')))
net.to(device)
net.eval()
# Model Evaluation on test data
classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
total = []
if args.test_acc:
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            wrong = torch.where(classs != labels, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
            acc = 1 - (torch.sum(wrong) / 64)
            total.append(acc.item())

    print('Accuracy of the network on the test images: %.2f %%' % (100 * np.mean(total)))


# helper_function for real time testing
def load_img(path):
    img = Image.open(path)
    img = transformation(img).float()
    img = torch.autograd.Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    return img.to(device)

if args.cam:

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
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi, (48, 48))
                cv2.imwrite(f"Webcam/roi.jpg", roi)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                imgg = load_img(f"Webcam/roi.jpg")
                out = net(imgg)
                pred = F.softmax(out)
                classs = torch.argmax(pred, 1)
                wrong = torch.where(classs != 3, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
                classs = torch.argmax(pred, 1)
                prediction = classes[classs.item()]

                img = cv2.putText(img, prediction, (x1 + org[0], y1 + org[1]), font,
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm1= nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.7)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels= 512, kernel_size=3, padding=1)
        self.batchnorm7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(3*3*512, 128)
        self.batchnorm9 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 512)
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 7)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.conv4(x))
        x = self.batchnorm4(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.relu(self.conv5(x))
        x = self.batchnorm5(x)
        x = self.relu(self.conv6(x))
        x = self.batchnorm6(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.relu(self.conv7(x))
        x = self.batchnorm7(x)
        x = self.relu(self.conv8(x))
        x = self.batchnorm8(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(self.batchnorm9(x))
        x = self.dropout2(x)

        x = self.fc2(x)
        x = self.relu(self.batchnorm10(x))

        x = self.fc3(x)

        return x


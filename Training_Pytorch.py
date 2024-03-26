import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mrcfile as mrc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def map_array_generator(data_list):
    for file in data_list:
        map = mrc.read(file)
        data = np.expand_dims(map, axis=0)
        data = torch.tensor(data, dtype=torch.float32)
        yield data

def getFilePathsAndLabels(root):
    Classes = os.listdir(root)
    file_paths = []
    labels = []

    for folder in Classes:
        try:
            folder_path = os.path.join(root, folder)
            class_file_list = os.listdir(folder_path)

            try:
                # Append file paths and labels for each file in the class folder
                for file in class_file_list:

                    file_paths.append(os.path.join(folder_path, file))
                    labels.append(folder)

            except:
                print('An error occured when getting the file paths. Current paths is {} and files in that path are {}.'.format(folder_path, class_file_list))
        except:
            print('Error occured when getting class paths. Classes are {} and the current folder is {}. If this is just DS_store you can ignore'.format(Classes, folder))

    return file_paths, labels

class mapDataset(Dataset):

    def __init__(self, file_paths, labels, label_to_index, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        label = self.label_to_index[label]  # Convert the label to index

        # Use generator to load the data from the file
        data = next(map_array_generator([file_path]))

        if self.transform:
            data = self.transform(data)

        return data, label

class Conv3DModel(nn.Module):
    def __init__(self, num_classes):
        super(Conv3DModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.num_classes = num_classes
        # Adaptive fully connected layers
        self.fc1 = nn.Linear(64 * 6 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))

        # Compute the size of the input tensor after convolution and pooling
        batch_size = x.size(0)
        num_features = x.view(batch_size, -1).size(1)

        x = x.view(batch_size, -1)  # Flatten the tensor

        # Update the fully connected layers based on the input size
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



root = 'Classes/Train/'

file_paths, labels = getFilePathsAndLabels(root)
label_to_index = {"Tomogram": 0, "NonTomogram": 1}

Data = mapDataset(file_paths, labels, label_to_index)



model = Conv3DModel(num_classes=len(label_to_index))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split your data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# Create data loaders for training and testing
train_dataset = mapDataset(train_data, train_labels, label_to_index)
test_dataset = mapDataset(test_data, test_labels, label_to_index)

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        data, labels = batch  # Unpack the data and labels
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Calculate accuracy on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%')

print('Training finished')

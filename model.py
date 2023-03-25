import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle


class ColorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ColorClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ColorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def read_color_data(csv_file):
    df = pd.read_csv(csv_file)
    X = df[['Red (8 bit)', 'Green (8 bit)', 'Blue (8 bit)']].values / 255.0
    y = df['Name'].values
    return X, y


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def test_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)


def main():
    print("Compiling model...")
    csv_file = './colors.csv'

    X, y = read_color_data(csv_file)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42)

    input_dim = X.shape[1]
    num_classes = len(np.unique(y_encoded))

    model = ColorClassifier(input_dim, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_dataset = ColorDataset(X_train, y_train)
    test_dataset = ColorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50

    print("Training model...")
    for _ in range(num_epochs):
        train_model(model, train_dataloader, criterion, optimizer, device)
        test_model(model, test_dataloader, criterion, device)

    print("Saving model...")
    torch.save(model.state_dict(), 'color_classifier.pth')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print("Model saved!")


if __name__ == '__main__':
    main()

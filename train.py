import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# DATASET PATH
# -----------------------------------
dataset_path = r"C:\Users\DELL\Desktop\PlantDataset"

# -----------------------------------
# TRANSFORMS
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# -----------------------------------
# LOAD DATASET
# -----------------------------------
dataset = datasets.ImageFolder(
    root=dataset_path,
    transform=transform
)

class_names = dataset.classes
num_classes = len(class_names)

print("\nClasses Found:\n")

for i, c in enumerate(class_names):
    print(i, ":", c)

# -----------------------------------
# SPLIT DATASET
# -----------------------------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)

# -----------------------------------
# DATALOADERS
# -----------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

# -----------------------------------
# DEVICE
# -----------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {device}")

# -----------------------------------
# LOAD SHUFFLENET
# -----------------------------------
model = models.shufflenet_v2_x0_5(weights="DEFAULT")

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

# -----------------------------------
# LOSS FUNCTION
# -----------------------------------
criterion = nn.CrossEntropyLoss()

# -----------------------------------
# OPTIMIZER
# -----------------------------------
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------------
# TRAINING
# -----------------------------------
epochs = 3

train_losses = []
train_accuracies = []

print("\nTraining Started...\n")

for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"Loss: {epoch_loss:.4f}")
    print(f"Accuracy: {epoch_accuracy:.2f}%")

# -----------------------------------
# SAVE MODEL
# -----------------------------------
torch.save(
    model.state_dict(),
    "plant_shuffle_model.pth"
)

print("\nModel Saved!")

# -----------------------------------
# TESTING
# -----------------------------------
model.eval()

all_labels = []
all_predictions = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_accuracy = 100 * correct / total

print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# -----------------------------------
# REPORT
# -----------------------------------
print("\nClassification Report:\n")

print(classification_report(
    all_labels,
    all_predictions,
    target_names=class_names
))

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------
cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(20,20))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ShuffleNet Confusion Matrix")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("shuffle_confusion_matrix.png")

plt.show()
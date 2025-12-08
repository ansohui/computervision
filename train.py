import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from googlenet import GoogLeNet  # 같은 폴더에 있는 googlenet.py에서 import

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #  Dataset / Dataloader 
    train_dir = "POC_Dataset/Training"
    test_dir = "POC_Dataset/Testing"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("Classes:", train_dataset.classes)
    num_classes = len(train_dataset.classes)

    #  Model / Loss / Optimizer 
    model = GoogLeNet(num_classes=num_classes, aux_logits=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5

    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, aux1, aux2 = model(images)

            loss_main = criterion(outputs, labels)
            loss_aux1 = criterion(aux1, labels)
            loss_aux2 = criterion(aux2, labels)
            loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}")

        # Eval 
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)   # eval 모드에선 main만 반환
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = correct / total if total > 0 else 0
        print(f"          Test Accuracy: {acc * 100:.2f}%")

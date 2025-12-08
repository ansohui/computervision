import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from googlenet import GoogLeNet  # 같은 폴더에 있는 googlenet.py에서 import

def plot_confusion_matrix(cm, class_names, epoch):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (epoch {epoch})")
    plt.tight_layout()

    plt.savefig(f"cm_epoch_{epoch:02d}.png")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #  Dataset / Dataloader 
    train_dir = "POC_Dataset/Training"
    test_dir = "POC_Dataset/Testing"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    #Validation/Test: remove augmentation 
    test_transform = transforms.Compose([   
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])


    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("Classes:", train_dataset.classes)
    num_classes = len(train_dataset.classes)

    #  Model / Loss / Optimizer 
    model = GoogLeNet(num_classes=num_classes, aux_logits=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )
    num_epochs = 20

    best_acc = 0.0
    best_state = None
    patience = 5        # 5 epoch 연속으로 개선 없으면 stop
    no_improve = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0



        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            ncols=100,
            leave=False,
        ):
            images = images.to(device)
            labels = labels.to(device)

            outputs, aux1, aux2 = model(images)

            loss_main = criterion(outputs, labels)
            loss_aux1 = criterion(aux1, labels)
            loss_aux2 = criterion(aux2, labels)

            #loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
            loss = loss_main + 0.1 * (loss_aux1 + loss_aux2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}")
        
        scheduler.step()
        # Eval 
        model.eval()
        correct = 0
        total = 0

        all_labels = []  #Confusion matrix용
        all_preds = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)   # eval 모드에선 main만 반환
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        acc = correct / total if total > 0 else 0
        print(f"          Test Accuracy: {acc * 100:.2f}%")

        # Confusion matrix 출력 
        cm = confusion_matrix(all_labels, all_preds)
        print("Class order:", train_dataset.classes)
        print("          Confusion matrix:")
        print(cm)

        plot_confusion_matrix(cm, train_dataset.classes,epoch+1)

        # Early Stopping 
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            no_improve = 0
            print(f"          ✅ New best accuracy: {best_acc * 100:.2f}%")
        else:
            no_improve += 1
            print(f"          No improvement for {no_improve} epoch(s).")

        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best acc = {best_acc * 100:.2f}%")
            break
    #save model
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), "googlenet_poc_best.pt")
        print(f"Best model saved with accuracy {best_acc * 100:.2f}%")

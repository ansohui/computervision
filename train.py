import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

from googlenet import GoogLeNet  # 같은 폴더에 있는 googlenet.py에서 import

def plot_confusion_matrix(cm, class_names, filename):
    """혼동행렬을 result/filename 으로 저장"""
    os.makedirs("result", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join("result", filename))
    plt.close()


if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
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
    eval_transform = transforms.Compose([   
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    full_train_for_train = datasets.ImageFolder(root=train_dir, transform=train_transform)
    full_train_for_val   = datasets.ImageFolder(root=train_dir, transform=eval_transform)
    
    print("Classes:", full_train_for_train.classes)
    num_classes = len(full_train_for_train.classes)


    val_ratio = 0.1
    val_size = int(len(full_train_for_train) * val_ratio)
    train_size = len(full_train_for_train) - val_size

    # 재현성을 위해 seed 고정
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_train_for_train)),
        [train_size, val_size],
        generator=generator
    )

    train_dataset = torch.utils.data.Subset(full_train_for_train, train_indices.indices)
    val_dataset   = torch.utils.data.Subset(full_train_for_val,   val_indices.indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("Classes:", full_train_for_train.classes)
    num_classes = len(full_train_for_train.classes)


    #  Model / Loss / Optimizer 
    model = GoogLeNet(num_classes=num_classes, aux_logits=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1
    )
    num_epochs = 20

    best_acc = 0.0
    best_state = None
    patience = 5        # 5 epoch 연속으로 개선 없으면 stop
    no_improve = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_acc": []
    }

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
        
        # Eval 
        model.eval()
        correct = 0
        total = 0

        all_labels = []  #Confusion matrix용
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)   # eval 모드에선 main만 반환
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_acc = correct / total if total > 0 else 0
        print(f"          Val Accuracy: {val_acc * 100:.2f}%")
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)

        #Val Confusion matrix 출력 
        cm_val = confusion_matrix(all_labels, all_preds)
        print("Class order:", full_train_for_train.classes)
        print("          Val Confusion matrix:")
        print(cm_val)

        plot_confusion_matrix(cm_val, full_train_for_train.classes,
                              filename=f"cm_val_epoch_{epoch+1:02d}.png")

        # Early Stopping 
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            no_improve = 0
            print(f"          ✅ New best VAL accuracy: {best_acc * 100:.2f}%")
        else:
            no_improve += 1
            print(f"          No improvement for {no_improve} epoch(s).")

        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best acc = {best_acc * 100:.2f}%")
            break
        scheduler.step()
    #save model
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), "result/googlenet_poc_best.pt")
        print(f"Best model saved with VAL accuracy {best_acc * 100:.2f}%")
    df = pd.DataFrame(history)
    df.to_csv("result/training_log.csv", index=False)
    print("Saved training log to result/training_log.csv")
    #final test
    model.eval()
    test_correct = 0
    test_total = 0
    all_test_labels = []
    all_test_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()

            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())

    test_acc = test_correct / test_total if test_total > 0 else 0
    print(f"[Final Test] Accuracy: {test_acc * 100:.2f}%")

    cm_test = confusion_matrix(all_test_labels, all_test_preds)
    print("Final Test Confusion matrix:")
    print(cm_test)

    plot_confusion_matrix(cm_test, test_dataset.classes,
                          filename="cm_test_final.png")


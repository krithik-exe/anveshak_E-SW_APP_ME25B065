import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score , auc , roc_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import Architecture and Config from cnn.py
from cnn import RobustCNN, DATASET_DIR, IMG_SIZE, DEVICE

def run_evaluation():
    # 1. Prepare Test Data
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    num_classes = len(full_dataset.classes)

    # Must match the split and seed used in cnn.py
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    torch.manual_seed(42) 
    _, test_dataset = random_split(full_dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. Load Model
    model = RobustCNN(num_classes=num_classes).to(DEVICE)
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load("model.pth"))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Generating metrics...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 3. Metrics
    print("\n--- Final Metrics ---")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))
    # 4. AUC-ROC Curve (Multi-class)
    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2])
    all_probs_np = np.array(all_probs)

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs_np[:, i])
        plt.plot(fpr, tpr, label=f'Class {full_dataset.classes[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('AUC-ROC Curve')
    plt.legend()
    plt.show()

    # 5. Failure Point Analysis
    def visualize_failures(model, loader, classes):
        model.eval()
        failed_images, true_labels, pred_labels = [], [], []
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                for i in range(len(preds)):
                    if preds[i] != labels[i]:
                        failed_images.append(images[i].cpu())
                        true_labels.append(labels[i].item())
                        pred_labels.append(preds[i].item())
                    if len(failed_images) >= 5: break # Just top 5
                if len(failed_images) >= 5: break

        # Plotting code for these 5 images...
        print(f"Analyzing {len(failed_images)} failures...")
    # 6. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=full_dataset.classes, 
                yticklabels=full_dataset.classes)
    plt.xlabel('Predicted by CNN')
    plt.ylabel('True Label (Ground Truth)')
    plt.title('Rover Cone Detection: Confusion Matrix')
    plt.show()
    

if __name__ == "__main__":
    run_evaluation()
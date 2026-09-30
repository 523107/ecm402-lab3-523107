"""
Part B - Train the PyTorch MLP for Automatic Modulation Classification.

Run from the repository root:

    python part_b/train_amc.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from dataset import generate_amc_dataset, MODULATIONS
from features import extract_amc_features
from model import AMCMLP


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 0

np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------

os.makedirs(
    "results/figures",
    exist_ok=True
)

os.makedirs(
    "results/tables",
    exist_ok=True
)


# ------------------------------------------------------------
# 1. Generate dataset
# ------------------------------------------------------------

print("Generating AMC dataset...")

X_iq, y, snr = generate_amc_dataset(
    n_per_class_per_snr=100,
    n_symbols=128,
    seed=SEED,
)

print(
    "I/Q dataset shape:",
    X_iq.shape
)

print(
    "Labels shape:",
    y.shape
)


# ------------------------------------------------------------
# 2. Extract features
# ------------------------------------------------------------

print("\nExtracting features...")

X = extract_amc_features(X_iq)

print(
    "Feature matrix shape:",
    X.shape
)


# ------------------------------------------------------------
# 3. Train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test, snr_train, snr_test = (
    train_test_split(
        X,
        y,
        snr,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
)


print("\nTrain samples:", len(X_train))
print("Test samples :", len(X_test))


# ------------------------------------------------------------
# 4. Standardize features
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# ------------------------------------------------------------
# 5. Convert to PyTorch tensors
# ------------------------------------------------------------

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32,
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long,
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long,
)


# ------------------------------------------------------------
# 6. DataLoader
# ------------------------------------------------------------

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
)


# ------------------------------------------------------------
# 7. Model
# ------------------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

model = AMCMLP(
    input_dim=12,
    hidden_dim=128,
    num_classes=5,
).to(device)


# ------------------------------------------------------------
# 8. Loss and optimizer
# ------------------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)


# ------------------------------------------------------------
# 9. Training
# ------------------------------------------------------------

epochs = 50

train_losses = []
train_accuracies = []

print("\nTraining...")

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(
            logits,
            yb
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * len(xb)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == yb
        ).sum().item()

        total += len(yb)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    train_losses.append(
        epoch_loss
    )

    train_accuracies.append(
        epoch_accuracy
    )

    if (
        epoch == 0
        or (epoch + 1) % 5 == 0
    ):
        print(
            f"Epoch {epoch + 1:3d}/{epochs} "
            f"| Loss = {epoch_loss:.4f} "
            f"| Accuracy = {epoch_accuracy:.4f}"
        )


# ------------------------------------------------------------
# 10. Test evaluation
# ------------------------------------------------------------

model.eval()

with torch.no_grad():

    X_test_device = X_test_tensor.to(
        device
    )

    logits = model(
        X_test_device
    )

    y_pred = logits.argmax(
        dim=1
    ).cpu().numpy()


test_accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nTest accuracy: {test_accuracy:.4f}"
)


# ------------------------------------------------------------
# 11. Classification report
# ------------------------------------------------------------

print("\nClassification report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=MODULATIONS,
        digits=4,
    )
)


# ------------------------------------------------------------
# 12. Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
)


print("Confusion matrix:")
print(cm)


np.savetxt(
    "results/tables/confusion_matrix.csv",
    cm,
    fmt="%d",
    delimiter=",",
)


# ------------------------------------------------------------
# 13. Loss curve
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    train_losses
)

plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")

plt.title(
    "AMC MLP Training Loss"
)

plt.tight_layout()

plt.savefig(
    "results/figures/amc_loss.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# 14. Training accuracy curve
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    train_accuracies
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "AMC MLP Training Accuracy"
)

plt.tight_layout()

plt.savefig(
    "results/figures/amc_accuracy.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# 15. Confusion matrix figure
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest",
)

plt.title(
    "AMC Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(MODULATIONS)
)

plt.xticks(
    tick_marks,
    MODULATIONS,
    rotation=45,
)

plt.yticks(
    tick_marks,
    MODULATIONS,
)

plt.xlabel(
    "Predicted class"
)

plt.ylabel(
    "True class"
)

plt.tight_layout()

plt.savefig(
    "results/figures/amc_confusion_matrix.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# 16. Save test accuracy
# ------------------------------------------------------------

with open(
    "results/tables/amc_results.txt",
    "w",
) as f:

    f.write(
        f"Test accuracy: "
        f"{test_accuracy:.6f}\n"
    )

    f.write(
        "\nConfusion matrix:\n"
    )

    f.write(
        np.array2string(cm)
    )


print(
    "\nAMC training completed."
)

print(
    "Figures saved in results/figures/"
)

print(
    "Tables saved in results/tables/"
)
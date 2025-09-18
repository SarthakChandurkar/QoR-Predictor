from models.fusion_model import DelayPredictor
from utils import create_dataset_with_features, AverageMeter
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import sys
import matplotlib.pyplot as plt
import numpy as np

graph_folder = "data/pt_files"
recipe_folder = "data/recipes"
label_folder = "data/out_label"

X, Y = create_dataset_with_features(graph_folder, recipe_folder, label_folder)
train_data, test_data, train_labels, test_labels = train_test_split(X, Y, train_size=0.85)
val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5)


model = DelayPredictor()
bestModel = DelayPredictor()

from torch.nn.utils import clip_grad_norm_
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

best_loss = sys.maxsize

loss_fn = nn.MSELoss()

def batchify(data, labels, batch_size=32):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        g_batch = [item[0] for item in batch]
        r_batch = torch.stack([item[1] for item in batch])
        y_batch = torch.stack(labels[i:i+batch_size])
        yield g_batch, r_batch, y_batch

# Training Loop
best_loss = sys.maxsize
for epoch in range(10):
    model.train()
    epochLoss = AverageMeter()
    for g_batch, r_batch, y_batch in batchify(train_data, train_labels):
        batch = Batch.from_data_list(g_batch)
        optimizer.zero_grad()
        pred = model(batch, r_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward(retain_graph=True)
        optimizer.step()
        numInputs = pred.view(-1,1).size(0)
        epochLoss.update(loss.detach().item(),numInputs)

    # Newly added
    avg_loss = epochLoss.avg
    scheduler.step(avg_loss)   
    if best_loss >= epochLoss.avg:
        best_loss = epochLoss.avg
        bestModel = model
        print(f"Best model at epoch: {epoch}")
        torch.save(model.state_dict(), 'exp-epoch-{}-loss-{}-2.pt'.format(epoch, best_loss))

    print(f"Epoch {epoch}: Loss {epochLoss.avg:.4f}")
    

# Save Model
torch.save(bestModel.state_dict(), "Model-exp2.pt")


def evaluate(model, data, labels):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for g_batch, r_batch, y in batchify(data, labels):
            batch = Batch.from_data_list(g_batch)
            outputs = model(batch, r_batch)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return mse, r2, mape, y_true, y_pred


train_mse, train_r2, train_mape, y_train_true, y_train_pred = evaluate(model, train_data, train_labels)
val_mse, val_r2, val_mape, y_val_true, y_val_pred = evaluate(model, val_data, val_labels)
test_mse, test_r2, test_mape, y_test_true, y_test_pred = evaluate(model, test_data, test_labels)


print(f"Train MSE: {train_mse:.4f}, R2: {train_r2:.4f}, MAPE : {train_mape * 100:.2f}, Train Accuracy : {100 - train_mape * 100:.2f}%")
print(f"Val   MSE: {val_mse:.4f}, R2: {val_r2:.4f}, MAPE : {val_mape * 100:.2f}, Validation Accuracy : {100 - val_mape * 100:.2f}%")
print(f"Test  MSE: {test_mse:.4f}, R2: {test_r2:.4f}, MAPE : {test_mape * 100:.2f}, Test Accuracy : {100 - test_mape * 100:.2f}%")

import pandas as pd

def save_predictions_to_csv(y_true, y_pred, set_name="test"):
    df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })
    df.to_csv(f"{set_name}_predictions.csv", index=False)
    print(f"Saved {set_name} predictions to {set_name}_predictions.csv")

# Save all three sets
save_predictions_to_csv(y_train_true, y_train_pred, "train")
save_predictions_to_csv(y_val_true, y_val_pred, "val")
save_predictions_to_csv(y_test_true, y_test_pred, "test")

def plot_multiple_predictions(y_true_list, y_pred_list, titles):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, ax in enumerate(axes):
        ax.scatter(y_true_list[i], y_pred_list[i], alpha=0.6, color='dodgerblue')
        ax.plot([min(y_true_list[i]), max(y_true_list[i])],
                [min(y_true_list[i]), max(y_true_list[i])], 'r--', label="Ideal")
        ax.set_xlabel("Actual Delay")
        ax.set_ylabel("Predicted Delay")
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# Compare all three
plot_multiple_predictions(
    [y_train_true, y_val_true, y_test_true],
    [y_train_pred, y_val_pred, y_test_pred],
    ["Train Set", "Validation Set", "Test Set"]
)

def plot_predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual Delay", set_name="Set"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual", linewidth=2, color='forestgreen')
    plt.plot(y_pred, label="Predicted", linewidth=2, color='dodgerblue', linestyle='--')
    plt.title(f"{title} - {set_name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Delay")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Assuming y_*_true and y_*_pred are already available as lists or arrays
plot_predicted_vs_actual(y_train_true, y_train_pred, set_name="Train")
plot_predicted_vs_actual(y_val_true, y_val_pred, set_name="Validation")
plot_predicted_vs_actual(y_test_true, y_test_pred, set_name="Test")

def plot_residuals(y_true_list, y_pred_list, titles):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, ax in enumerate(axes):
        y_true = np.array(y_true_list[i])
        y_pred = np.array(y_pred_list[i])
        residuals = y_pred - y_true

        ax.scatter(y_true, residuals, alpha=0.6, color='coral', label='Residuals')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.5)
        ax.set_xlabel("Actual Delay")
        ax.set_ylabel("Residual (Predicted - Actual)")
        ax.set_title(f"Residual Plot: {titles[i]}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

plot_residuals(
    [y_train_true, y_val_true, y_test_true],
    [y_train_pred, y_val_pred, y_test_pred],
    ["Train Set", "Validation Set", "Test Set"]
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file (update the path if needed)
df = pd.read_csv('test_predictions.csv')

# Extract actual and predicted values
y_test_true = df['Actual'].values
y_test_pred = df['Predicted'].values

# Load the CSV file (update the path if needed)
df = pd.read_csv('val_predictions.csv')

# Extract actual and predicted values
y_val_true = df['Actual'].values
y_val_pred = df['Predicted'].values

# Load the CSV file (update the path if needed)
df = pd.read_csv('train_predictions.csv')

# Extract actual and predicted values
y_train_true = df['Actual'].values
y_train_pred = df['Predicted'].values

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
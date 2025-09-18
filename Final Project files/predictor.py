import torch
from models.fusion_model import DelayPredictor
from utils import create_dataset_with_features
from torch_geometric.data import Batch
import pandas as pd
import os

# Paths
graph_folder = "pred/pt_files"
recipe_folder = "pred/recipes"
label_folder = "pred/out_label"
saved_model_path = "two.pt"

# Load data
X, Y = create_dataset_with_features(graph_folder, recipe_folder, label_folder)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DelayPredictor().to(device)
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.eval()

# Batchify utility
def batchify(data, labels, batch_size=32):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        g_batch = [item[0] for item in batch]
        r_batch = torch.stack([item[1] for item in batch])
        y_batch = torch.stack(labels[i:i+batch_size])
        yield g_batch, r_batch, y_batch

# Make predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for g_batch, r_batch, y_batch in batchify(X, Y):
        batch = Batch.from_data_list(g_batch).to(device)  

        preds = model(batch, r_batch)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Save to CSV
df = pd.DataFrame({
    "GroundTruth": all_labels,
    "Prediction": all_preds
})

output_csv = "predictions.csv"
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")

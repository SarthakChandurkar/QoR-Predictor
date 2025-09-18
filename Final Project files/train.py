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

graph_folder = "data/pt_files"
recipe_folder = "data/recipes"
label_folder = "data/out_label"

X, Y = create_dataset_with_features(graph_folder, recipe_folder, label_folder)
train_data, test_data, train_labels, test_labels = train_test_split(X, Y, train_size=0.85)
val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5)


model = DelayPredictor()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Newly added
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
        torch.save(model.state_dict(), 'exp-epoch-{}-loss-{}-0.85.pt'.format(epoch, best_loss))

    print(f"Epoch {epoch}: Loss {epochLoss.avg:.4f}")
    

# Save Model
torch.save(model.state_dict(), "ss_pcm-0.85.pt")


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
  
    return mse, r2, mape

train_mse, train_r2, train_mape = evaluate(model, train_data, train_labels)
val_mse, val_r2, val_mape = evaluate(model, val_data, val_labels)
test_mse, test_r2, test_mape = evaluate(model, test_data, test_labels)

print(f"Train MSE: {train_mse:.4f}, R2: {train_r2:.4f}, MAPE : {train_mape * 100:.2f}, Train Accuracy : {100 - train_mape * 100:.2f}%")
print(f"Val   MSE: {val_mse:.4f}, R2: {val_r2:.4f}, MAPE : {val_mape * 100:.2f}, Validation Accuracy : {100 - val_mape * 100:.2f}%")
print(f"Test  MSE: {test_mse:.4f}, R2: {test_r2:.4f}, MAPE : {test_mape * 100:.2f}, Test Accuracy : {100 - test_mape * 100:.2f}%")


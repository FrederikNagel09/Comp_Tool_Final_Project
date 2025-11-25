import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
from torch import nn
from torch.utils.data import DataLoader



def generate_and_split(
    n_vectors=10,
    seed=42,
    test_size=0.2
):
    """
    Generates synthetic vectors and splits into train/validation sets.

    - First 256 dims: simulated BERT embeddings
    - First 8 dims: simulated metadata features
    - Output vector shape: (264,)
    """

    rng = np.random.default_rng(seed)

    # BERT part
    bert_part = rng.uniform(-1, 1, size=(n_vectors, 256))

    # Metadata part
    meta_part = rng.uniform(-1, 1, size=(n_vectors, 8))

    # Combined features
    X = np.hstack([meta_part, bert_part])

    # Example synthetic labels (binary)
    y = rng.integers(0, 2, size=n_vectors)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    return X_train, X_val, y_train, y_val

X_train, X_val, y_train, y_val = generate_and_split(n_vectors=10)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256+8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork()
print(model)

model = NeuralNetwork()


# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).float().unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).float().unsqueeze(1)


# Training setup
loss_fn = nn.BCELoss() # the loss function to caculate error, used # binary_entropy_loss 
optimizer = torch.optim.Adam(model.parameters()) # adam is a special optimzier

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    
    # use the model, and count errors
    logits = model(X_train_tensor)
    loss = loss_fn(logits, y_train_tensor)
    
    # Backward pass
    optimizer.zero_grad() # clear previous training loop's gradient 
    loss.backward()
    optimizer.step()
    
    # Check if parameters are still updating
    if epoch > 0 and abs(loss.item() - prev_loss) < 1e-10:
        print(f"Early stopping at epoch {epoch}: parameters not updating")
        break
    prev_loss = loss.item()




# Final predictions and evaluation
model.eval()

# Training accuracy
train_logits = model(X_train_tensor)
train_preds = (train_logits > 0.5).float()
train_acc = (train_preds == y_train_tensor).float().mean()

print(f"Training Accuracy: {train_acc:.4f}")

# Validation accuracy
val_logits = model(X_val_tensor)    
val_preds = (val_logits > 0.5).float()
val_acc = (val_preds == y_val_tensor).float().mean()


print(f"Validation Accuracy: {val_acc:.4f}")

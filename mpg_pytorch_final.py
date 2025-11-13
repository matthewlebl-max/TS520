import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import onnx
import onnxruntime as rt

# =====================================
# 1. LOAD CLEAN CSV
# =====================================
path_data = "C:/Users/matth/Downloads/auto_mpg_clean.csv"
df = pd.read_csv(path_data)

X = df.iloc[:, 1:].values.astype(np.float32)   # 7 features
y = df.iloc[:, 0].values.astype(np.float32)    # mpg

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).view(-1, 1)

X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test).view(-1, 1)

# =====================================
# 2. NORMALIZE
# =====================================
means = X_train_t.mean(0, keepdim=True)
stds = X_train_t.std(0, keepdim=True) + 1e-6

def normalize(x):
    return (x - means) / stds

# =====================================
# 3. MODEL
# =====================================
class MPGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = normalize(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MPGNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =====================================
# 4. TRAIN
# =====================================
for epoch in range(300):
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = F.mse_loss(pred, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(epoch, loss.item())

# =====================================
# 5. Evaluate
# =====================================
pred_test = model(X_test_t).detach().numpy().flatten()
print("R2:", r2_score(y_test, pred_test))

# =====================================
# 6. EXPORT TO ONNX
# =====================================
dummy = torch.randn(1, 7)

torch.onnx.export(
    model,
    dummy,
    "mpg_pytorch.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17
)

print("ONNX FILE SAVED AS mpg_pytorch.onnx")

# =====================================
# 7. TEST ONNX RUNTIME
# =====================================
sess = rt.InferenceSession("mpg_pytorch.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

test_sample = X_test[:1].astype(np.float32)  # first car

res = sess.run([label_name], {input_name: test_sample})
print("ONNX prediction:", res)
print("Real MPG:", y_test[0])

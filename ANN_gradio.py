import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gradio as gr


df = pd.read_csv(r"C:\Users\Dell\Work_tasks\Iris.csv")

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = pd.get_dummies(df['Species']).values


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
y_test = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

class ImprovedIrisANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ImprovedIrisANN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x


input_size = X.shape[1]
hidden_size = 10
output_size = y.shape[1]
learning_rate = 0.001
epochs = 1000

model = ImprovedIrisANN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    train_outputs = model(X_train)
    train_loss = criterion(train_outputs, y_train)

    train_loss.backward()
    optimizer.step()


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data = scaler.transform(input_data)  
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    species = ['Setosa', 'Versicolor', 'Virginica']
    return species[predicted.item()]


iface = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Slider(minimum=4, maximum=8, value=5.0, label="Sepal Length (cm)"),  
        gr.Slider(minimum=2, maximum=4.5, value=3.0, label="Sepal Width (cm)"),  
        gr.Slider(minimum=1, maximum=7, value=4.5, label="Petal Length (cm)"), 
        gr.Slider(minimum=0.1, maximum=2.5, value=1.5, label="Petal Width (cm)")  
    ],
    outputs="text",
    live=True,
    title="Iris Flower Prediction",
    description="Predict the species of an Iris flower based on its measurements."
)

print("Launching Gradio interface...")

iface.launch(server_port=7861, share = True)
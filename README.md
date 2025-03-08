# EX 1 : Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

![Screenshot 2025-02-27 085038](https://github.com/user-attachments/assets/8a23e8ff-cc3a-4c27-bd03-286d2c0c9d9e)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SWATHI D
### Register Number: 212222230154
```python
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1, 4)
    self.fc2 = nn.Linear(4, 6)
    self.fc3 = nn.Linear(6, 1)
    self.relu = nn.ReLU()
    self.history = {'loss':[]}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
## Dataset Information
![Screenshot 2025-02-27 084550](https://github.com/user-attachments/assets/57977aa3-ad27-4bf4-afc5-fc89d3f1ad1e)

## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot 2025-02-27 084633](https://github.com/user-attachments/assets/dbb0e281-d351-492d-b5d4-0e517a4d57cb)


### New Sample Data Prediction
![Screenshot 2025-02-27 084707](https://github.com/user-attachments/assets/da08c827-084b-4b64-85be-ea4fb60af556)

![Screenshot 2025-02-27 084725](https://github.com/user-attachments/assets/a75e6136-4c3b-43df-bd8c-ca92840789c0)

## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.

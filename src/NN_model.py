# Here I create my Neural network model and define training and validation function
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset 
########################################################
# Neural Network Definition
########################################################
class BinaryClassifierNN(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.BCNlayers = nn.Sequential(
            nn.Linear(input_size, 18), # This is input layer
            nn.ReLU(),      # non-linear Activation function
            nn.Dropout(0.2),
            nn.Linear(18,9), # This is a hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(9,1) # This is output layer which gives 1 outcome since we are doing binary classification
        )

    def forward(self,x):
        return self.BCNlayers(x) #Specify the exact order in which input tensor x passes through the created layers
########################################################
# Definition of function that train and validates
########################################################
def train_and_validate(training_dataset, validation_dataset, Mymodel, loss_fn, optimizer, epochs, device): 
    model = Mymodel.to(device) # Send Mymodel to device

    epoch_arr = []
    accuracy_arr = []
    train_loss_arr = []
    valid_loss_arr = []

    for epoch in range(epochs):

        # This is for training the model
        model.train()
        total_training_loss = 0.0
    
        for batch_X, batch_y in training_dataset:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device) # Send the input and output vectors to device
        
            optimizer.zero_grad()          # Clear stored gradients
            outputs = model(batch_X)       # Forward pass input vector to model and get outputs as predictions
            training_loss = loss_fn(outputs, batch_y) # Compute loss
            training_loss.backward()                # Backpropagation => Extremizing loss function
            optimizer.step()               # Update weights and biases

            total_training_loss += training_loss.item()      # We just add up loss in each epoch

        # This is for validating the model
        model.eval()
        total_validaton_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_valid_X, batch_valid_y in validation_dataset:
                batch_valid_X, batch_valid_y = batch_valid_X.to(device), batch_valid_y.to(device) # Send the input and output vectors to device
                valid_outputs = model(batch_valid_X) # Forward pass input vector to model and get outputs as predictions
                valid_loss = loss_fn(valid_outputs, batch_valid_y) # Compute loss
                # No backpropagation and Updating of weights and biases for model validation

                total_validaton_loss += valid_loss.item()      # We just add up loss in each epoch
                # We also need to evaluate accuracy of prediction by our model
                predicted_valid = (valid_outputs > 0.0).float().cpu().numpy() # Find "valid_outputs > 0.0" => The class which wins has prob > 1/2
                targets_valid = batch_valid_y.cpu().numpy()

                all_predictions.extend(predicted_valid)
                all_targets.extend(targets_valid)
            
        avg_training_loss = total_training_loss/ len(training_dataset)
        avg_validation_loss = total_validaton_loss/ len(validation_dataset)
        accuracy = accuracy_score(all_targets, all_predictions) * 100.0

        epoch_arr.append(epoch+1)
        accuracy_arr.append(accuracy)
        train_loss_arr.append(avg_training_loss)
        valid_loss_arr.append(avg_validation_loss)

    # Return a dictionary containing all tracked metrics
    return {"epochs": epoch_arr,"train_loss": train_loss_arr,"valid_loss": valid_loss_arr,"accuracy": accuracy_arr,}

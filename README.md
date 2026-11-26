# Titanic Survival Prediction Using Neural Network

This is a binary classification project that predicts passenger survival probability on the Titanic using a neural network built in PyTorch.

## Project Overview

This project is a prepared as a complete ML pipeline starting from the raw data to trained neural network model. In order to prepare the raw data for training,  feature engineering and data preprocessing have been used. Additionally, model construction, model training and model evaluation have also been performed.

**Final validation accuracy: ~81%**

## Repository Structure

```
Titanic_Survival/
- data/
    - Titanic_train.csv        # Raw training data (891 passengers)
    - Titanic_test.csv         # Raw test data (418 passengers)
- src/
    - data_preprocessing.py    # Feature engineering and preprocessing functions
    - NN_model.py              # Neural network creation and training-validation loop
- Training_EvaluationNN.ipynb  # Main training and evaluation notebook
- Data_Visualization.ipynb     # Initial exploratory data analysis and visualizations
- outputs/
    - model_weights.pth        # Saved model weights
    - Titanic_survival_predictions.csv  # Test set predictions
- requirements.txt # Required libraries
- README.md # Project overview
```

## Model Architecture

A 3-layer fully connected network:

```
Input (42 features) - Linear(42, 18) - ReLU - Dropout(0.2)
                    - Linear(18, 9)  - ReLU - Dropout(0.2)
                    - Linear(9, 1)   - BCEWithLogitsLoss
```

**Optimizer:** AdamW (with lr=0.001, weight_decay=0.01)
**Loss function:** BCEWithLogitsLoss 
**Epochs:** 20

## Feature Engineering

Key features engineered from the raw data:

| Feature | Description |
|   |   |
| `Family_member` | SibSp + Parch (total family size aboard) |
| `Title` | Extracted from passenger Name (Mr, Miss, Mrs, Master etc.) |
| `Sex_Pclass` | Interaction feature combining sex and passenger class |
| `CabinCategory` | First letter of Cabin (deck indicator) |
| `TicketSum` | Digit sum of ticket number |
| `Ticket_Is_Odd` | Parity of ticket number |

Numerical features are `log1p`-transformed then z-score normalised.  
Categorical features are one-hot encoded with `pd.get_dummies`.

## Preprocessing File

The `preprocess_titanic_data()` function in `src/data_preprocessing.py` handles the full data preprocessing pipeline maintaining feature alignment between train, validation, and test sets.

## How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Open `Data_Visualization.ipynb` to explore the data visualization related plots.

3. Open `Training_EvaluationNN.ipynb` and run all cells to train and evaluate the model.

## Results

| Metric | Value |
|   |   |
| Validation accuracy | ~81% |
| Training loss (epoch 20) | ~0.38 |
| Validation loss (epoch 20) | ~0.46 |

**The train/validation loss gap is small, indicating the model generalises well without significant overfitting.**

## Requirements

See `requirements.txt`. Main dependencies: `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

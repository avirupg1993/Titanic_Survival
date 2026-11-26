# This python script is for all functions I need to preprocess the data before feeding it into my ML model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
################################################################################
# Define the function to sum the ticket numbers
def sum_ticket(ticket):
    last_part = str(ticket).split()[-1]
    if last_part.isdigit():
        return sum(int(digit) for digit in last_part) # Sum the digits appearing in "last_part"
    return 0

def even_odd_ticket(ticket):
    last_part = str(ticket).split()[-1]
    
    if last_part.isdigit():
        return int(last_part) % 2  # Returns 0 for Even, 1 for Odd
    else:
        return -1  # Handles non-numeric ticket entries

################################################################################
def preprocess_titanic_data(df, scaler_A=None, scaler_ticket=None, reference_columns=None):
    """
    Preprocesses raw Titanic-survivor DataFrame for a Neural Network.
    """
    data = df.copy()
    
    # Define column groups
    colsA = ["Age", "Fare", "SibSp", "Parch", "Family_member"]
    colsB = ["Sex", "Embarked","Sex_Pclass","Title"]
    cols_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    
    # Feature Engineering: Family Member = SibSp + Parch
    data['Family_member'] = data['SibSp'].fillna(0) + data['Parch'].fillna(0)
    
    
    # For Numerical Features: Fill NA -> Log1p -> Standardize
    data[colsA] = data[colsA].fillna(0)
    data[colsA] = np.log1p(data[colsA]).astype('float32')
    
    if scaler_A is None:
        scaler_A = StandardScaler()
        data[colsA] = scaler_A.fit_transform(data[colsA])
    else:
        data[colsA] = scaler_A.transform(data[colsA])
    
    # Extract Mr Mrs Miss from the name column 
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.')
        
    # For Categorical Features: Fill NA -> One-Hot Encoding
    data['Pclass'] = data['Pclass'].fillna(0)
    data['Sex'] = data['Sex'].fillna("Unknown")
    data['Embarked'] = data['Embarked'].fillna("U")
    data['CabinCategory'] = data['Cabin'].fillna('U1').str[0]
    data['Title'] = data['Title'].fillna("Unknown")
    
    # Feature Engineering: Sex_Pclass = Sex + Pclass
    data['Sex_Pclass'] = data['Sex'] + '_' + data['Pclass'].astype(str)
    
    # one-hot encoding of all categorical features
    ohe_cols = ['Pclass'] + colsB + ['CabinCategory']
    data = pd.get_dummies(data, columns=ohe_cols, dtype=np.float32)
    
    # Creating Ticket Features using sum_ticket and even_odd_ticket functions
    data['TicketSum'] = data['Ticket'].apply(sum_ticket)
    data['TicketSum'] = np.log1p(data['TicketSum']).astype('float32')
    
    if scaler_ticket is None:
        scaler_ticket = StandardScaler()
        data[['TicketSum']] = scaler_ticket.fit_transform(data[['TicketSum']])
    else:
        data[['TicketSum']] = scaler_ticket.transform(data[['TicketSum']])
        
    data['Ticket_Is_Odd'] = data['Ticket'].apply(even_odd_ticket).astype('float32')
    
    # Dropping unnecessary columns 
    data = data.drop(columns=cols_drop, errors='ignore')
    
    # Aligning columns with training data (otherwise training and validation may fail)
    if reference_columns is not None:
        data = data.reindex(columns=reference_columns, fill_value=0.0)
        
    return data, scaler_A, scaler_ticket


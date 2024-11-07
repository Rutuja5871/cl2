# import pandas as pd
# import numpy as np
# from pgmpy.models import BayesianNetwork
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.estimators import MaximumLikelihoodEstimator
# from pgmpy.inference import VariableElimination
# from sklearn.model_selection import train_test_split

# def load_and_preprocess_data():
#     df = pd.read_csv("datasets\heartdisease-IR.csv")
#     return df

# def create_bayesian_network(data):
#     """Create and train a Bayesian Network model."""
#     # Define the network structure based on logical relationships
#     model = BayesianNetwork([
#         ('age', 'cholestrol'),
#         ('Gender', 'cholestrol'),
#         ('Family', 'heartdisease'),
#         ('diet', 'cholestrol'),
#         ('Lifestyle', 'cholestrol'),
#         ('cholestrol', 'heartdisease'),
#     ])
    
#     # Estimate CPDs using Maximum Likelihood Estimation
#     mle = MaximumLikelihoodEstimator(model=model, data=data)
    
#     # Add CPDs for each node
#     for node in model.nodes():
#         cpd = mle.estimate_cpd(node)
#         model.add_cpds(cpd)
    
#     # Check if the model is valid
#     assert model.check_model()
    
#     return model

# def predict_heart_disease(model, patient_data):
#     """Make predictions using the trained Bayesian Network."""
#     inference = VariableElimination(model)
    
#     # Predict probability of heart disease
#     prediction = inference.query(variables=['heartdisease'], evidence=patient_data)
#     return prediction

# def evaluate_model(model, test_data):
#     """Evaluate the model's performance."""
#     correct_predictions = 0
#     total_predictions = len(test_data)
    
#     for idx, row in test_data.iterrows():
#         # Prepare evidence
#         evidence = {
#             'age': row['age'],
#             'Gender': row['Gender'],
#             'Family': row['Family'],
#             'diet': row['diet'],
#             'Lifestyle': row['Lifestyle'],
#             'cholestrol': row['cholestrol']
#         }
        
#         # Make prediction
#         prediction = predict_heart_disease(model, evidence)
#         predicted_class = np.argmax(prediction.values)
        
#         # Compare with actual class
#         if predicted_class == row['heartdisease']:
#             correct_predictions += 1
    
#     accuracy = correct_predictions / total_predictions
#     return accuracy

# def main():
#     # Load and preprocess data
#     data = load_and_preprocess_data()
    
#     # Split data into train and test sets
#     train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
#     # Create and train the Bayesian Network
#     model = create_bayesian_network(train_data)
    
#     # Evaluate the model
#     accuracy = evaluate_model(model, test_data)
#     print(f"Model Accuracy: {accuracy:.2f}")
    
#     # Example prediction for a new patient
#     example_patient = {
#         'age': 0,
#         'Gender': 0,
#         'Family': 0,
#         'diet': 0,
#         'Lifestyle': 0,
#         'cholestrol': 0
#     }
    
#     prediction = predict_heart_disease(model, example_patient)
#     print("\nPrediction for example patient:")
#     print(f"Probability distribution of Heart Disease:")
#     for i, prob in enumerate(prediction.values):
#         print(f"Class {i}: {prob:.2f}")

# if __name__ == "__main__":
#     main()



import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    """Load and preprocess the Heart Disease dataset."""
    # Load the dataset
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    data = pd.read_csv("datasets\heart.csv")
    
    # Discretize continuous variables
    continuous_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    for var in continuous_vars:
        kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        data[var] = kbd.fit_transform(data[[var]])
    
    return data

def create_bayesian_network(data):
    """Create and train a Bayesian Network model."""
    # Define the network structure
    model = BayesianNetwork([
        ('age', 'trestbps'),
        ('age', 'chol'),
        ('sex', 'thalach'),
        ('sex', 'target'),
        ('cp', 'target'),
        ('trestbps', 'target'),
        ('chol', 'target'),
        ('fbs', 'target'),
        ('restecg', 'target'),
        ('thalach', 'target'),
        ('exang', 'target'),
        ('oldpeak', 'target'),
        ('slope', 'target'),
        ('ca', 'target'),
        ('thal', 'target')
    ])
    
    # Estimate CPDs using Maximum Likelihood Estimation
    mle = MaximumLikelihoodEstimator(model=model, data=data)
    for node in model.nodes():
        cpd = mle.estimate_cpd(node)
        model.add_cpds(cpd)
    
    # Check if the model is valid
    assert model.check_model()
    
    return model

def predict_heart_disease(model, patient_data):
    """Make predictions using the trained Bayesian Network."""
    inference = VariableElimination(model)
    
    # Predict probability of heart disease
    prediction = inference.query(variables=['target'], evidence=patient_data)
    return prediction

def evaluate_model(model, test_data):
    """Evaluate the model's performance."""
    correct_predictions = 0
    total_predictions = len(test_data)
    
    for idx, row in test_data.iterrows():
        # Prepare evidence
        evidence = row.drop('target').to_dict()
        
        # Make prediction
        prediction = predict_heart_disease(model, evidence)
        predicted_class = np.argmax(prediction.values)
        
        # Compare with actual class
        if predicted_class == row['target']:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

def main():
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create and train the Bayesian Network
    model = create_bayesian_network(train_data)
    
    # Evaluate the model
    accuracy = evaluate_model(model, test_data)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Example prediction for a new patient
    example_patient = {
        'age': 1,  # middle-aged (discretized)
        'sex': 1,  # male
        'cp': 1,   # chest pain type
        'trestbps': 1,  # normal blood pressure (discretized)
        'chol': 1,      # normal cholesterol (discretized)
        'fbs': 1,       # fasting blood sugar < 120 mg/dl
        'restecg': 1,   # normal ECG
        'thalach': 1,   # normal max heart rate (discretized)
        'exang': 1,     # no exercise induced angina
        'oldpeak': 1,   # no ST depression (discretized)
        'slope': 1,     # flat ST slope
        'ca': 1,        # no major vessels colored by fluoroscopy
        'thal': 1       # normal blood flow
    }
    
    prediction = predict_heart_disease(model, example_patient)
    print("\nPrediction for example patient:")
    print(f"Probability of Heart Disease: {prediction.values[1]:.2f}")

if __name__ == "__main__":
    main()
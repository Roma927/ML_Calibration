import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('E:\\MLModelsUncertaintyEstimationandCalibration_Mariam_Ashraf\\Knowledge_base_train.csv')

# Encode categorical features if present
categorical_cols = data.select_dtypes(include=['object']).columns
if not categorical_cols.empty:
    print(f"Encoding categorical columns: {categorical_cols.tolist()}")
    for col in categorical_cols:
        data[col] = LabelEncoder().fit_transform(data[col])

# Define features (X) and target (y)
X = data.drop(['class'], axis=1) 
y = data['class']

# Save feature names
feature_names = X.columns.tolist()
with open('features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classification models
models = {
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=500),
    "MLPClassifier": MLPClassifier(random_state=42, max_iter=500),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
    "GaussianNB": GaussianNB(),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(),
}

trained_models = {}
calibration_results = {}

calibration_methods = ['sigmoid', 'isotonic']  # Adding multiple calibration methods

# Train, calibrate, and evaluate models
for name, model in models.items():
    print(f"Training model: {name}")
    model.fit(X_train, y_train)
    trained_models[name] = model

    # Evaluate model accuracy before calibration
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model: {name} | Accuracy: {accuracy:.4f}")

    # Apply multiple calibration methods
    for method in calibration_methods:
        print(f"Calibrating model {name} using {method} method...")
        calibrated_model = CalibratedClassifierCV(model, method=method, cv="prefit")
        calibrated_model.fit(X_test, y_test)
        predictions_calibrated = calibrated_model.predict(X_test)
        accuracy_calibrated = accuracy_score(y_test, predictions_calibrated)
        print(f"Model: {name} (Calibrated with {method}) | Accuracy: {accuracy_calibrated:.4f}")

        # Store results
        calibration_results[f"{name}_{method}"] = {
            "Calibration Method": method,
            "Accuracy (Before Calibration)": accuracy,
            "Accuracy (After Calibration)": accuracy_calibrated,
            "Confusion Matrix": confusion_matrix(y_test, predictions_calibrated).tolist(),
            "Classification Report": classification_report(y_test, predictions_calibrated, output_dict=True),
        }

        # Reliability plot
        probabilities = calibrated_model.predict_proba(X_test)
        plt.figure(figsize=(8, 6))
        plt.hist(probabilities.max(axis=1), bins=10, label='Predicted Probabilities', alpha=0.6, color='blue')
        plt.title(f"Reliability Plot for {name} ({method})")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        output_dir = "reliability_plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{name}_{method}_reliability_plot.png"))
        plt.show()
    from sklearn.preprocessing import label_binarize

    # Binarize labels for multiclass calibration
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

    for i, class_label in enumerate(np.unique(y_test)):
        prob_true, prob_pred = calibration_curve(y_test_binarized[:, i], probabilities[:, i], n_bins=10)

        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 's-', label=f'Class {class_label} Calibrated Model')
        plt.plot([0, 1], [0, 1], '--', color='red', label='Perfect Calibration')
        plt.title(f"Calibration Curve for Class {class_label} - {name}")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"{name}_calibration_curve_class_{class_label}.png"))
        plt.close()

# Save trained models
with open('trained_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)

# Save calibration results
with open('calibration_results.pkl', 'wb') as f:
    pickle.dump(calibration_results, f)

print("Models training, calibration with multiple methods, and reliability visualization completed successfully.")

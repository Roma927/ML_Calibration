### ML_Calibration
---

#### **Project Overview**
This project focuses on training and evaluating multiple classification models using a given dataset. The key objectives include:  
1. Training models from scikit-learn.  
2. Evaluating model performance on accuracy.  
3. Applying **calibration techniques** (`sigmoid` and `isotonic`) to improve reliability.  
4. Generating reliability plots to visualize probability predictions.  
5. Saving trained models and results for further analysis.

---

#### **Dataset**
- **File:** `Knowledge_base_train.csv`
- **Target Variable:** `class`
- **Features:** Categorical and numeric columns. Categorical columns are encoded using `LabelEncoder`.

---

#### **Models Used**
The following scikit-learn models are trained and evaluated:

1. **Random Forest Classifier**
2. **Support Vector Classifier (SVC)**
3. **Logistic Regression**
4. **Multi-layer Perceptron (MLP) Classifier**
5. **Gradient Boosting Classifier**
6. **AdaBoost Classifier**
7. **Gaussian Naive Bayes**
8. **Decision Tree Classifier**
9. **K-Nearest Neighbors (KNN)**

---

#### **Steps Performed**
1. **Data Preprocessing:**
   - Categorical columns were identified and encoded using `LabelEncoder`.
   - Features and target were split into train-test sets (80%-20%).

2. **Training and Evaluation:**
   - Each model was trained on the train set.
   - Test set accuracy was calculated pre-calibration.

3. **Calibration:**
   - Models were calibrated using **sigmoid** and **isotonic** methods.
   - Post-calibration accuracy and reliability plots were generated for each method.

4. **Saving Results:**
   - Trained models and calibration results were saved as `.pkl` files.
   - Reliability plots were stored as `.png` files.

---

#### **Performance Results**
- **Random Forest Classifier:** 
  - Pre-Calibration Accuracy: **57.50%**
  - Post-Calibration (Sigmoid): **63.75%**
  - Post-Calibration (Isotonic): **71.25%**
- **SVC:** 
  - Pre-Calibration Accuracy: **40.00%**
  - Post-Calibration (Sigmoid): **43.75%**
  - Post-Calibration (Isotonic): **47.50%**
- **Logistic Regression:** 
  - Pre-Calibration Accuracy: **43.75%**
  - Post-Calibration (Sigmoid): **48.75%**
  - Post-Calibration (Isotonic): **51.25%**
- **MLP Classifier:** 
  - Pre-Calibration Accuracy: **40.00%**
  - Post-Calibration (Sigmoid): **41.25%**
  - Post-Calibration (Isotonic): **47.50%**
- **Gradient Boosting Classifier:** 
  - Pre-Calibration Accuracy: **53.75%**
  - Post-Calibration (Sigmoid): **55.00%**
  - Post-Calibration (Isotonic): **62.50%**
- **AdaBoost Classifier:** 
  - Pre-Calibration Accuracy: **45.00%**
  - Post-Calibration (Sigmoid): **40.00%**
  - Post-Calibration (Isotonic): **40.00%**
- **Gaussian Naive Bayes:** 
  - Pre-Calibration Accuracy: **37.50%**
  - Post-Calibration (Sigmoid): **48.75%**
  - Post-Calibration (Isotonic): **53.75%**
- **Decision Tree Classifier:** 
  - Pre-Calibration Accuracy: **58.75%**
  - Post-Calibration (Sigmoid): **58.75%**
  - Post-Calibration (Isotonic): **58.75%**
- **KNN:** 
  - Pre-Calibration Accuracy: **40.00%**
  - Post-Calibration (Sigmoid): **41.25%**
  - Post-Calibration (Isotonic): **40.00%**

---

#### **Installation and Setup**
1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd project
   ```

2. **Install required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training and evaluation script:**
   ```bash
   python Main.py
   ```

4. **Optional Flask API:**
   - Modify and run the `Server.py` script to serve the models via a REST API.

---

#### **Output Files**
1. **Trained Models:** Stored in `trained_models.pkl`.
2. **Calibration Results:** Stored in `calibration_results.pkl` with detailed metrics for each model and calibration method.
3. **Reliability Plots:** Stored in `reliability_plots/` for each model-calibration combination.

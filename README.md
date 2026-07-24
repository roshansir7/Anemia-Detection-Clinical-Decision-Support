# Anemia Detection and Clinical Decision Support

A machine-learning and Streamlit project for analysing clinical blood data and predicting whether a patient may be anemic.

The project combines exploratory data analysis, predictive modelling, hyperparameter tuning and an interactive clinical decision-support application.

## Project Objective

The main objective is to develop a classification model that can support the early identification of anemia using demographic information and common blood-test measurements.

The system classifies patients into two groups:

* `0` – Non-Anemic
* `1` – Anemic

## Clinical Features

The prediction system uses the following patient information:

| Feature | Description                               |
| ------- | ----------------------------------------- |
| Gender  | Patient gender                            |
| Age     | Patient age                               |
| Hb      | Hemoglobin level                          |
| RBC     | Red blood cell count                      |
| PCV     | Packed cell volume                        |
| MCV     | Mean corpuscular volume                   |
| MCH     | Mean corpuscular hemoglobin               |
| MCHC    | Mean corpuscular hemoglobin concentration |

## Data Science Workflow

The Jupyter Notebook covers the complete machine-learning process:

1. Loading and inspecting the anemia dataset
2. Cleaning and preprocessing clinical data
3. Encoding categorical variables
4. Performing exploratory data analysis
5. Examining feature distributions and correlations
6. Preparing training and testing datasets
7. Handling class imbalance
8. Training multiple classification models
9. Evaluating model performance
10. Performing hyperparameter tuning with GridSearchCV
11. Saving the selected model for use in Streamlit

## Model Evaluation

The models are compared using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Cross-validation results

The repository contains separate CSV files for:

* Original model performance
* Hyperparameter-tuning results
* Original versus tuned model comparison

The selected Decision Tree model is saved as:

```text
best_decision_tree_model.pkl
```

## Streamlit Application

The Streamlit application is organised into four main pages.

### Welcome

Provides an introduction to the project, dataset and machine-learning objectives.

### Data Overview

Displays:

* Number of records
* Number of columns
* Missing values
* Anemic and non-anemic case counts
* Dataset preview
* Statistical summary
* Data types and missing-value information

### Visualisations

Provides interactive analysis of:

* Anemic versus non-anemic patients
* Gender and anemia status
* Clinical feature distributions
* Hemoglobin differences
* Correlation matrix
* Feature importance
* Pairwise feature relationships

Users can choose different chart types and explore the clinical dataset interactively.

### Model Evaluation and Prediction

Allows users to:

* Compare original model performance
* Review tuned model performance
* Compare original and tuned models
* Select different evaluation metrics
* Enter individual patient measurements
* Generate an anemia prediction
* View the predicted probability

## Dataset Upload

The application supports custom datasets in:

```text
.xlsx
.csv
```

Uploaded datasets should follow the same column structure used during model training.

## Repository Structure

```text
Predictive-Modeling-and-Clinical-Decision-Support-for-Anemia-Detection-Using-Machine-Learning/
│
├── Anemia Dataset.xlsx
├── Data_Science_Final.ipynb
├── anemia.py
├── best_decision_tree_model.pkl
├── blood-drop.png
├── performance_measures.csv
├── grid_search_results.csv
├── model_performance_comparison.csv
├── requirements.txt
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn
* Imbalanced-learn
* Joblib
* Streamlit
* Jupyter Notebook
* OpenPyXL

## Run the Data Science Notebook

Clone the repository:

```bash
git clone https://github.com/roshansir7/Predictive-Modeling-and-Clinical-Decision-Support-for-Anemia-Detection-Using-Machine-Learning.git
```

Move into the project folder:

```bash
cd Predictive-Modeling-and-Clinical-Decision-Support-for-Anemia-Detection-Using-Machine-Learning
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Data_Science_Final.ipynb
```

Run the notebook cells from top to bottom.

## Run the Streamlit Application

After installing the requirements, run:

```bash
streamlit run anemia.py
```

The application should open automatically in your browser.

When it does not open automatically, visit:

```text
http://localhost:8501
```

## Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud using your GitHub account.
2. Select **Create app**.
3. Choose this GitHub repository.
4. Select the `main` branch.
5. Set the main file path to:

```text
anemia.py
```

6. Click **Deploy**.

All required application files must remain in the repository root, especially:

```text
Anemia Dataset.xlsx
best_decision_tree_model.pkl
performance_measures.csv
grid_search_results.csv
model_performance_comparison.csv
blood-drop.png
requirements.txt
```

## Limitations

* The prediction depends on the quality and representativeness of the training dataset.
* The model should be externally validated before clinical use.
* Clinical thresholds may vary between populations and laboratories.
* The application is intended for education and decision support rather than independent diagnosis.
* A qualified healthcare professional should interpret real patient results.

## Future Development

Possible improvements include:

* Adding ROC-AUC and sensitivity-specificity visualisations
* Using SHAP for patient-level model explanations
* Adding input-range validation
* Testing the model on external clinical datasets
* Adding secure patient-record management
* Improving mobile responsiveness
* Adding downloadable prediction reports
* Comparing Decision Tree performance with additional ensemble models

## Disclaimer

This application was developed for educational and research purposes. It is not a certified medical device and must not replace professional medical assessment, laboratory testing or clinical diagnosis.

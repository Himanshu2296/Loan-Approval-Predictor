# Loan Approval Predictor

A simple machine learning project that predicts whether a loan application is likely to be approved based on applicant information.

## Objective

Build a machine learning model that predicts loan approval using:

* Monthly income
* Credit score
* Debt-to-income ratio
* Employment years
* Loan amount
* Prior defaults

## Machine Learning Model

The project uses:

**Logistic Regression**

## Project Architecture

```text
Loan Applicant
      ↓
Applicant Information
      ↓
Logistic Regression Model
      ↓
Prediction
      ↓
Loan Approved / Loan Not Approved
```

## Project Structure

```text
Loan-Approval-Predictor/
│
├── data/
│   └── loan.csv
│
├── inspect_data.py
├── train.py
├── predict.py
├── loan_model.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/himanshu2296/Loan-Approval-Predictor.git
```

Move into the project:

```bash
cd Loan-Approval-Predictor
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train.py
```

The trained model will be saved as:

```text
loan_model.joblib
```

## Make a Prediction

Run:

```bash
python predict.py
```

Enter the applicant's information when prompted.

The program will return:

```text
Loan Approved
```

or:

```text
Loan Not Approved
```

## Model Performance

The initial Logistic Regression model achieved:

**Accuracy: 70.5%**

This is a baseline model. The advanced version will improve the preprocessing, model evaluation, user interface, and overall application.

## Technologies

* Python
* Pandas
* Scikit-learn
* Logistic Regression
* Joblib
* Git
* GitHub

## Future Improvements

* Feature scaling
* Model comparison
* Cross-validation
* Precision, recall, and F1-score
* Confusion matrix
* Prediction probability
* Streamlit web application
* Model improvement
* Online deployment

## Disclaimer

This project is created for educational purposes. Its predictions should not be used as the sole basis for real financial or lending decisions.

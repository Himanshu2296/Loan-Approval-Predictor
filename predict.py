import pandas as pd
import joblib

model = joblib.load("loan_model.joblib")

print("======= Loan Approval Predictor =======")

income_monthly = float(input("Monthly Income: "))
credit_score = int(input("Credit Score: "))
debt_to_income = float(input("Debt to Income: "))
employment_years = float(input("Employment Years: "))
loan_amount = float(input("Loan Amount: "))
prior_defaults = int(input("Prior Defaults: "))

applicant = pd.DataFrame([{
    "income_monthly": income_monthly,
    "credit_score": credit_score,
    "debt_to_income": debt_to_income,
    "employment_years": employment_years,
    "loan_amount": loan_amount,
    "prior_defaults": prior_defaults
}])

prediction = model.predict(applicant)[0]

print("\n===== Result =====")

if prediction == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")
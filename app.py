import streamlit as st
import pandas as pd
import joblib

from utils import evaluate_model

model = joblib.load("loan_model.joblib")

accuracy, precision, recall, f1, cm = evaluate_model(model)

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("Loan Approval Predictor")

st.write("Enter the applicant information below to predict loan approval.")

st.divider()

st.header("Applicant Information")

income_monthly = st.number_input("Monthly Income", min_value=0.0, value=100.0, step=1.0)

credit_score = st.number_input("Credit Score",min_value=0,max_value=900, value=700, step=1)

employment_years = st.number_input("Employment Years",min_value=0.0,value=5.0, step=0.5)

st.header("Loan Information")

debt_to_income = st.number_input("Debt to Income",min_value=0.0,value=50.0, step=1.0)

loan_amount = st.number_input("Loan Amount",min_value=0.0,value=100.0, step=1.0)

prior_defaults = st.number_input("Prior Defaults",min_value=0,value=0,step=1)

st.divider()

if st.button("Predict Loan Approval", use_container_width=True):

    if income_monthly <= 0:
        st.error("Monthly income must be greater than 0.")

    elif loan_amount <= 0:
        st.error("Loan amount must be greater than 0.")

    else:

        applicant = pd.DataFrame([{
            "income_monthly": income_monthly,
            "credit_score": credit_score,
            "debt_to_income": debt_to_income,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "prior_defaults": prior_defaults
        }])

        prediction = model.predict(applicant)[0]

        probabilities = model.predict_proba(applicant)[0]

        approval_probability = probabilities[1] * 100
        rejection_probability = probabilities[0] * 100

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Not Approved")

        st.write(f"Approval Probability: {approval_probability:.2f}%")

        st.progress(int(approval_probability))

        st.write(f"Rejection Probability: {rejection_probability:.2f}%")

        st.divider()

        st.header("Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{accuracy * 100:.1f}%")
            st.metric("Precision", f"{precision * 100:.1f}%")

        with col2:
            st.metric("Recall", f"{recall * 100:.1f}%")
            st.metric("F1 Score", f"{f1 * 100:.1f}%")

        st.divider()

        st.header("Confusion Matrix")

        st.write("The confusion matrix shows how many loan applications were classified correctly and incorrectly")

        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual Not Approved", "Actual Approved"],
                columns=["Predicted Not approved", "Predicted Approved"]
            )
        )
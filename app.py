import joblib
import gradio as gr
import numpy as np

# Load the trained model
model = joblib.load("loan_approval_model.pkl")

# Prediction Function
def predict_loan(
    no_of_dependents,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
):

    input_data = np.array([[
        no_of_dependents,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    prediction = model.predict(input_data)[0]

    # If model supports probability
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(input_data)) * 100
    else:
        confidence = None

    # Output
    if prediction == 1:
        result = "✅ Loan Approved"
    else:
        result = "❌ Loan Rejected"

    if confidence is not None:
        return f"{result}\nConfidence: {confidence:.2f}%"
    else:
        return result


# Gradio Interface
demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Number of Dependents"),
        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term (Years)"),
        gr.Number(label="CIBIL Score"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Assets Value"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Loan Approval Prediction",
    description="Enter applicant details to predict loan approval."
)

import os
port = int(os.environ.get("PORT", 7860))
demo.launch(
    server_name="0.0.0.0",
    server_port=port
)

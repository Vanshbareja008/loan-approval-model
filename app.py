import os
import joblib
import numpy as np
import gradio as gr

# ==========================================================
# Load the trained model
# ==========================================================
model = joblib.load("loan_app_model.pkl")

# ==========================================================
# Prediction Function
# ==========================================================
def predict_loan(
    no_of_dependents,
    income_annum,
    education,
    self_employed,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
):

    # Encode categorical values
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    input_data = np.array([[
        no_of_dependents,
        income_annum,
        education,
        self_employed,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(input_data)) * 100
    else:
        confidence = 100

    if prediction == 0:
        result = "✅ Loan Approved"
    else:
        result = "❌ Loan Rejected"

    return f"{result}\n\nConfidence: {confidence:.2f}%"

# ==========================================================
# Gradio Interface
# ==========================================================

demo = gr.Interface(
    fn=predict_loan,

    inputs=[
        gr.Number(label="Number of Dependents"),
        gr.Number(label="Annual Income (₹)"),

        gr.Dropdown(
            ["Graduate", "Not Graduate"],
            label="Education"
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Self Employed"
        ),

        gr.Number(label="Loan Amount (₹)"),
        gr.Number(label="Loan Term (Years)"),
        gr.Number(label="CIBIL Score"),
        gr.Number(label="Residential Assets Value (₹)"),
        gr.Number(label="Commercial Assets Value (₹)"),
        gr.Number(label="Luxury Assets Value (₹)"),
        gr.Number(label="Bank Assets Value (₹)")
    ],

    outputs=gr.Textbox(
        label="Prediction Result",
        lines=4
    ),

    title="🏦 Loan Approval Prediction System",

    description="""
# 🏦 Loan Approval Prediction System

### 👨‍💻 Developed by **Vansh**
### 🎓 Roll No.: **241047**

---

## 📌 Instructions

✔ Enter all details carefully.

✔ Select the correct **Education** and **Self Employed** status.

✔ Click **Submit** to predict loan approval.

---

## 📝 Example Input

| Field | Example |
|-------|---------|
| Number of Dependents | 2 |
| Annual Income | 6500000 |
| Education | Graduate |
| Self Employed | No |
| Loan Amount | 2500000 |
| Loan Term | 12 |
| CIBIL Score | 780 |
| Residential Assets | 5000000 |
| Commercial Assets | 1500000 |
| Luxury Assets | 1000000 |
| Bank Assets | 900000 |

---

### 📊 Output

✅ Loan Approved

or

❌ Loan Rejected

along with the prediction confidence.
"""
)

# ==========================================================
# Launch App
# ==========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )

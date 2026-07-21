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
    outputs=gr.Textbox(label="Prediction"),
    title="🏦 Loan Approval Prediction System",

    description="""
# 🏦 Loan Approval Prediction System

### 👨‍💻 Developed by
**Vansh**

**Roll No.: 241047**

---

## 📌 Instructions
- Enter all values correctly.
- Select the appropriate Education and Self Employed status.
- Click **Submit** to predict whether the loan is likely to be approved.

---

## 📝 Example Input

| Parameter | Example Value |
|-----------|--------------:|
| Number of Dependents | **2** |
| Annual Income (₹) | **6500000** |
| Education | **Graduate** |
| Self Employed | **No** |
| Loan Amount (₹) | **2500000** |
| Loan Term (Years) | **12** |
| CIBIL Score | **780** |
| Residential Assets (₹) | **5000000** |
| Commercial Assets (₹) | **1500000** |
| Luxury Assets (₹) | **1000000** |
| Bank Assets (₹) | **900000** |

---

### 📊 Prediction Output

✅ **Loan Approved**

or

❌ **Loan Rejected**

The prediction also displays the model confidence.
"""
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )

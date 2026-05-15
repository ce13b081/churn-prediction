"""Streamlit Live Predictor page: real-time churn prediction via the FastAPI."""
import pathlib
import sys

import requests
import streamlit as st

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

API_URL = "http://localhost:8000/predict"


def _risk_color(risk: str) -> str:
    return {"Low": "green", "Medium": "orange", "High": "red"}.get(risk, "gray")


def main():
    st.title("Live Churn Predictor")
    st.markdown(
        "Fill in the customer details below and click **Predict** to get an "
        "instant churn probability from the trained model."
    )
    st.info(
        "Ensure the FastAPI server is running: "
        "`uvicorn api.main:app --reload --port 8000`",
        icon="ℹ️",
    )

    with st.form("prediction_form"):
        st.subheader("Customer Profile")
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.radio("Partner", ["Yes", "No"], horizontal=True)
            dependents = st.radio("Dependents", ["Yes", "No"], horizontal=True)

        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 0.0, 120.0, 65.0, step=0.5)
            total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges), step=10.0)

        with c3:
            phone_service = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            paperless = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

        st.subheader("Internet & Streaming")
        i1, i2, i3 = st.columns(3)

        with i1:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            no_internet = internet == "No"
            internet_options = ["No internet service"] if no_internet else ["Yes", "No", "No internet service"]

            online_security = st.selectbox("Online Security", internet_options)
            online_backup = st.selectbox("Online Backup", internet_options)

        with i2:
            device_protection = st.selectbox("Device Protection", internet_options)
            tech_support = st.selectbox("Tech Support", internet_options)

        with i3:
            streaming_tv = st.selectbox("Streaming TV", internet_options)
            streaming_movies = st.selectbox("Streaming Movies", internet_options)

        st.subheader("Contract & Payment")
        p1, p2 = st.columns(2)
        with p1:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with p2:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            prob = result["churn_probability"]
            risk = result["risk_level"]
            churn = result["churn_predicted"]
            model_name = result.get("model_name", "unknown")

            st.divider()
            st.subheader("Prediction Result")
            r1, r2, r3 = st.columns(3)
            r1.metric("Churn Predicted", "YES" if churn else "NO")
            r2.metric("Churn Probability", f"{prob:.1%}")
            r3.metric("Risk Level", risk)

            st.progress(prob, text=f"Churn probability: {prob:.1%}")

            if churn:
                st.error(
                    f"This customer is **likely to churn** (risk: {risk}). "
                    "Consider proactive retention measures.",
                    icon="🚨",
                )
            else:
                st.success(
                    f"This customer is **unlikely to churn** (risk: {risk}).",
                    icon="✅",
                )

            st.caption(f"Prediction by model: {model_name.replace('_', ' ').title()}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the API. Make sure the FastAPI server is running:\n\n"
                "```\nuvicorn api.main:app --reload --port 8000\n```",
                icon="🔌",
            )
        except requests.exceptions.HTTPError as exc:
            st.error(f"API error: {exc}", icon="❌")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}", icon="❌")


main()

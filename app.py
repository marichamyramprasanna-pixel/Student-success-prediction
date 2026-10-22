import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Success Platform", layout="centered")
st.title("🎓 Student Success Prediction Platform")
st.write("Enter student metrics below to assess academic success probability and identify early intervention needs.")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load('student_success_model.pkl')

model = load_model()

# User Inputs
col1, col2 = st.columns(2)
with col1:
    attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 85.0)
    study_hours = st.slider("Study Hours / Week", 0.0, 40.0, 15.0)
    prev_gpa = st.slider("Previous GPA", 0.0, 4.0, 3.0, step=0.1)
    assignments = st.slider("Assignments Completed (%)", 0.0, 100.0, 80.0)

with col2:
    lms_logins = st.number_input("LMS Logins / Month", min_value=0, max_value=300, value=45)
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    internet = st.selectbox("Reliable Internet Access", ["Yes", "No"])

# Prediction
if st.button("Predict Student Outcome"):
    input_df = pd.DataFrame([{
        'attendance_rate': attendance,
        'study_hours_per_week': study_hours,
        'previous_gpa': prev_gpa,
        'assignments_completed': assignments,
        'lms_logins_per_month': lms_logins,
        'extracurricular': extracurricular,
        'internet_access': internet
    }])

    prob = model.predict_proba(input_df)[0][1]
    prediction = "On Track (Success)" if prob >= 0.5 else "At-Risk (Intervention Needed)"

    st.subheader("Prediction Result")
    if prob >= 0.5:
        st.success(f"**Status:** {prediction} (Confidence: {prob:.1%})")
    else:
        st.error(f"**Status:** {prediction} (Success Probability: {prob:.1%})")
        st.info("💡 **Recommended Action:** Schedule an academic advising session and review attendance/assignment bottlenecks.")
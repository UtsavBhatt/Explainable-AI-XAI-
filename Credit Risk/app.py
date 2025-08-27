import streamlit as st
import pandas as pd
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="XAI for Credit Risk Analysis",
    page_icon="📊",
    layout="wide"
)

# --- Data and Model Loading ---
@st.cache_resource
def load_artifacts():
    """
    Loads all the saved models, encoder, and training data from disk.
    Using st.cache_resource to prevent reloading on every interaction.
    """
    try:
        models = {
        'XGBoost': pickle.load(open('Credit Risk/xgboost_model.pkl', 'rb')),
        'LightGBM': pickle.load(open('Credit Risk/lightgbm_model.pkl', 'rb'))
        }
        encoder_cols = pickle.load(open('Credit Risk/encoder.pkl', 'rb'))
        X_train = pickle.load(open('Credit Risk/X_train.pkl', 'rb'))
        return models, encoder_cols, X_train
    except FileNotFoundError:
        st.error("One or more model files not found. Please run the `Models.py` script first to train and save the artifacts.")
        return None, None, None

# --- App Title and Introduction ---
st.title("XAI for Credit Risk Analysis")
st.markdown("""
This interactive application allows you to explore the predictions of different credit risk models.
By using Explainable AI (XAI) techniques like SHAP and LIME, we can make the model's decision-making process more transparent.

This approach aligns with the principles of robust, transparent, and trustworthy AI as emphasized in standards like **ISO/IEC 42001**.

**Select a model and provide applicant details in the sidebar to get a risk prediction and its explanation.**
""")

st.markdown("---")
st.header("ISO/IEC 42001: A Framework for Responsible AI")
st.markdown("""
This tool is designed to align with the core principles of the ISO/IEC 42001 standard for AI management systems, promoting responsible and trustworthy AI. Here's how:

*   **Transparency & Explainability:** The app provides clear, decision-level explanations for each prediction. By using both SHAP and LIME, it offers multiple perspectives on *why* a model made a specific decision, demystifying the "black box."

*   **Robustness & Validation:** By allowing users to compare explanations from different high-performing models (XGBoost and LightGBM) for the same input, the app demonstrates a robust approach to understanding model behavior. This comparison is key for a thorough AI risk management process and helps avoid over-reliance on a single model's logic.

*   **Accountability:** The clear, model-specific visualizations create a tangible audit trail. For any given prediction, you can see precisely which factors influenced the model's decision, establishing a clear line of accountability for the outcomes.
""")

# --- Sidebar for User Controls ---
st.sidebar.title("Controls")
st.sidebar.markdown("Select a model and input applicant data to see the prediction.")

models, encoder_cols, X_train = load_artifacts()

# Add a model selection dropdown to the sidebar if models were loaded successfully
if models:
    st.sidebar.header("User Input Features")
    selected_model_name = st.sidebar.selectbox(
        "Choose a Model",
        list(models.keys())
    )

    with st.sidebar.expander("Enter Applicant Details", expanded=True):
        # --- User Input Fields ---
        person_age = st.slider('Age', 20, 80, 25)
        person_income = st.number_input('Annual Income', min_value=4000, max_value=250000, value=50000, step=1000)
        person_emp_length = st.slider('Years of Employment', 0, 45, 5)
        loan_amnt = st.number_input('Loan Amount', min_value=500, max_value=35000, value=10000, step=500)
        loan_int_rate = st.slider('Interest Rate (%)', 5.0, 23.0, 10.0, 0.1)

        # Categorical Inputs
        person_home_ownership = st.selectbox('Home Ownership', ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
        loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
        loan_grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        cb_person_default_on_file = st.selectbox('Has Defaulted Before?', ['N', 'Y'])
        cb_person_cred_hist_length = st.slider('Credit History Length (years)', 2, 30, 5)

    # Create a dictionary from the user inputs
    user_input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        # Calculated feature
        'loan_percent_income': loan_amnt / person_income if person_income > 0 else 0,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length
    }

    # --- Prediction Logic ---
    if st.sidebar.button('Get Prediction'):
        # 1. Get the selected model object
        selected_model = models[selected_model_name]

        # 2. Preprocess the user input to match the model's training data
        input_df = pd.DataFrame([user_input_data])
        # Perform one-hot encoding
        input_encoded = pd.get_dummies(input_df)
        # Align columns with the training data, filling missing columns with 0
        final_input = input_encoded.reindex(columns=encoder_cols, fill_value=0)

        # 3. Make a prediction
        prediction = selected_model.predict(final_input)[0]
        prediction_proba = selected_model.predict_proba(final_input)[0][1] # Probability of default

        # 4. Display the prediction result
        st.subheader(f"Prediction by: {selected_model_name}")

        if prediction == 1:
            st.error("Result: Loan Rejected (High Risk of Default)")
        else:
            st.success("Result: Loan Approved (Low Risk of Default)")

        # Display the probability using a metric card
        st.metric(
            label="Predicted Probability of Default",
            value=f"{prediction_proba:.2%}"
        )
        st.info("This probability indicates the model's confidence that the applicant will default on the loan.", icon="💡")

        # Add a collapsible section for model explanations
        with st.expander("See Model Explanations"):
            # --- SHAP Explanation ---
            st.subheader(f"SHAP Explanation for {selected_model_name}")
            st.markdown("""
            **How to interpret this plot:** The SHAP force plot shows which features pushed the model's prediction higher (in red) and which pushed it lower (in blue).
            - **Base value:** The average prediction for all applicants.
            - **Features:** The most influential features for this specific applicant are shown. The size of the bar represents the magnitude of the feature's impact.
            - **Final prediction (f(x)):** The output value after considering all feature impacts.
            """)

            # --- Model-Specific SHAP Logic ---
            if selected_model_name in ['XGBoost', 'LightGBM']:
                explainer = shap.TreeExplainer(selected_model)
                shap_values = explainer.shap_values(final_input)
                # For these models, shap_values is a single array for the positive class.
                shap.force_plot(explainer.expected_value, shap_values[0], final_input.iloc[0], matplotlib=True, show=False)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf()

            # --- LIME Explanation (common for all) ---
            st.subheader(f"LIME Explanation for {selected_model_name}")
            st.markdown("""
            **How to interpret this plot:** LIME (Local Interpretable Model-agnostic Explanations) shows which features were most important for this single prediction.
            - **Green bars:** Features that support the predicted outcome.
            - **Red bars:** Features that contradict the predicted outcome.
            - **Feature values:** The conditions (e.g., `loan_amnt <= 10000`) show the specific feature values for this applicant that influenced the decision.
            """)
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=['Approved (0)', 'Rejected (1)'],
                mode='classification'
            )
            lime_exp = lime_explainer.explain_instance(final_input.iloc[0].values, selected_model.predict_proba, num_features=10)
            fig = lime_exp.as_pyplot_figure()
            st.pyplot(fig)

        # Add another expander to explain probability differences
        with st.expander("Why do the models' probabilities differ?"):
            st.markdown("""
            It's common for different models like XGBoost and LightGBM to produce different prediction probabilities, even if they agree on the final outcome (Approve/Reject). Here’s why:

            1.  **Different Tree Building Strategy:**
                -   **XGBoost** builds trees *level-wise*. It creates one full layer of the tree at a time, resulting in balanced, symmetric trees.
                -   **LightGBM** builds trees *leaf-wise*. It focuses on the single leaf that will reduce the error the most, leading to deeper, more complex, and asymmetric trees. This fundamental difference in how they learn from the data creates different decision boundaries.

            2.  **Internal Algorithms & Optimizations:**
                -   The models use different internal algorithms for finding the best splits and handling data. For example, LightGBM uses a highly optimized histogram-based method and techniques like Gradient-based One-Side Sampling (GOSS) to speed up training. These internal mechanics are not identical to XGBoost's.

            3.  **Different "Opinions":**
                -   Think of the models as two different experts. Given the same information, they might both conclude that a loan is low-risk, but one expert might be 95% confident while the other is 88% confident. These differences in confidence (probability) reflect their unique ways of interpreting the data.

            Ultimately, they are distinct models that learn different representations of the data, which naturally leads to variations in their prediction probabilities.
            """)

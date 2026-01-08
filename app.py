
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from utils.BITS_Classification_Assignment import get_models, train_models, evaluate_models

st.set_page_config(page_title='Breast Cancer Classification', layout='wide')
st.title('Breast Cancer Classification App')

st.markdown('This app trains six classification models (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, '
            'Random Forest, XGBoost) on the built-in **Breast Cancer** dataset and reports key metrics: Accuracy, AUC, '
            'Precision, Recall, F1, and MCC. It also plots ROC curves.')

# Sidebar controls
st.sidebar.header('Configuration')
random_state = st.sidebar.number_input('Random state', min_value=0, value=42, step=1)
test_size = st.sidebar.slider('Test size', min_value=0.1, max_value=0.5, value=0.2, step=0.05)

col_run, col_dl = st.columns([1, 1])
with col_run:
    run = st.button('Train & Evaluate')

# Session state to hold last results
if 'results_df' not in st.session_state:
    st.session_state['results_df'] = None
    st.session_state['roc_data'] = None

if run:
    # Load data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Prepare models
    models = get_models(random_state=random_state)
    trained = train_models(models, X_train, y_train)
    results_df, roc_data = evaluate_models(trained, X_test, y_test)

    st.session_state['results_df'] = results_df
    st.session_state['roc_data'] = roc_data

# Show metrics table
if st.session_state['results_df'] is not None:
    st.subheader('Classification Metrics (Test Set)')
    st.dataframe(st.session_state['results_df'], use_container_width=True)

    # Download CSV
    csv_bytes = st.session_state['results_df'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download metrics CSV',
        data=csv_bytes,
        file_name='classification_metrics_breast_cancer.csv',
        mime='text/csv',
    )

    # Plot ROC curves where available
    st.subheader('ROC Curves')
    roc_dict = st.session_state['roc_data']
    if roc_dict:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, d in roc_dict.items():
            ax.plot(d['fpr'], d['tpr'], label=f"{name} (AUC={d['auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        st.pyplot(fig)
    else:
        st.info('No ROC curve data available.')
else:
    st.info('Click **Train & Evaluate** to generate metrics and ROC curves.')

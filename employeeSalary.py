import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

import pickle
import io
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Employee Salary Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2E86AB;
    margin-bottom: 1rem;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.info-box {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #667eea;
    margin: 1rem 0;
    color: #2c3e50;
}

.top-model-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    color: white;
    text-align: center;
}

.upload-info {
    background: #e8f4fd;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #667eea;
    margin: 1rem 0;
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# Initialize state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []

# Helper functions
def preprocess_data(data):
    """Preprocess the data"""
    df = data.copy()
    
    # Standardize column names
    column_mapping = {
        'sex': 'gender',
        'educational-num': 'education-num',
        'educational_num': 'education-num'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Handle missing values
    for col in df.select_dtypes(include=['object']).columns:
        if '?' in df[col].values:
            df[col] = df[col].replace('?', 'Others')
        df[col] = df[col].fillna('Others')
    
    # Fill numerical NaN values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != 'income':
            df[col] = df[col].fillna(df[col].median())
    
    # Remove outliers
    if 'age' in df.columns:
        df = df[(df['age'] <= 75) & (df['age'] >= 17)]
    
    # Remove specific workclass
    if 'workclass' in df.columns:
        df = df[~df['workclass'].isin(['Without-pay', 'Never-worked'])]
    
    # Remove specific education categories
    if 'education' in df.columns:
        df = df[~df['education'].isin(['1st-4th', '5th-6th', 'Preschool'])]
        # Drop education column if education-num exists
        if 'education-num' in df.columns:
            df = df.drop(columns=['education'])
    
    # Drop fnlwgt
    if 'fnlwgt' in df.columns:
        df = df.drop(columns=['fnlwgt'])
    
    return df

def encode_features(df):
    df_encoded = df.copy()
    label_encoders = {}
    
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != 'income']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    return df_encoded, label_encoders

def train_models(X_train, X_test, y_train, y_test):
    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            trained_models[name] = model
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
            continue
    
    return results, trained_models

# Main app
def main():
    st.markdown('<h1 class="main-header">💰 Employee Salary Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## Navigation")
    tab = st.sidebar.selectbox("Choose Section", 
                              ["📊 Data Analysis", "🤖 Model Training", "🔮 Predictions", "📈 Model Comparison"])
    
    if tab == "📊 Data Analysis":
        data_analysis_tab()
    elif tab == "🤖 Model Training":
        model_training_tab()
    elif tab == "🔮 Predictions":
        prediction_tab()
    elif tab == "📈 Model Comparison":
        model_comparison_tab()

def data_analysis_tab():
    st.markdown('<div class="sub-header">Data Analysis & Preprocessing</div>', unsafe_allow_html=True)
    
    # Data upload section
    st.markdown("### Upload Your Dataset")
    
    # Info box about dataset requirements
    st.markdown("""
    <div class="upload-info">
        <h4>📋 Dataset Requirements:</h4>
        <ul>
            <li>CSV format with headers</li>
            <li>Must contain an 'income' column for the target variable</li>
            <li>Income values should be categorical (e.g., '>50K', '<=50K')</li>
            <li>Can contain both numerical and categorical features</li>
            <li>Missing values will be handled automatically</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("✅ Data uploaded successfully!")
            
            # Check if income column exists
            if 'income' not in st.session_state.data.columns:
                st.error("❌ Dataset must contain an 'income' column for salary prediction.")
                return
                
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            return
    
    if 'data' in st.session_state:
        data = st.session_state.data
        
        # Data overview
        st.markdown("### Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card"><h3>Rows</h3><h2>{}</h2></div>'.format(data.shape[0]), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>Columns</h3><h2>{}</h2></div>'.format(data.shape[1]), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>Missing Values</h3><h2>{}</h2></div>'.format(data.isnull().sum().sum()), unsafe_allow_html=True)
        with col4:
            income_dist = data['income'].value_counts() if 'income' in data.columns else {"N/A": 1}
            st.markdown('<div class="metric-card"><h3>Income Classes</h3><h2>{}</h2></div>'.format(len(income_dist)), unsafe_allow_html=True)
        
        # Display data
        st.markdown("### Raw Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Show column information
        st.markdown("### Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Columns:**")
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            st.write(numerical_cols if numerical_cols else "None")
            
        with col2:
            st.write("**Categorical Columns:**")
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            st.write(categorical_cols if categorical_cols else "None")
        
        # Data preprocessing
        st.markdown("### Data Preprocessing")
        if st.button("🔄 Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                try:
                    processed_data = preprocess_data(data)
                    st.session_state.processed_data = processed_data
                    st.success(f"✅ Data preprocessed successfully! Shape: {processed_data.shape}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Shape:**", data.shape)
                    with col2:
                        st.write("**Processed Shape:**", processed_data.shape)
                        
                except Exception as e:
                    st.error(f"❌ Error during preprocessing: {str(e)}")
        
        # Visualizations
        if 'processed_data' in st.session_state:
            st.markdown("### Data Visualizations")
            processed_data = st.session_state.processed_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'income' in processed_data.columns:
                    fig_income = px.pie(processed_data, names='income', title='Income Distribution')
                    st.plotly_chart(fig_income, use_container_width=True)
            
            with col2:
                if 'age' in processed_data.columns:
                    fig_age = px.histogram(processed_data, x='age', title='Age Distribution', nbins=20)
                    st.plotly_chart(fig_age, use_container_width=True)
            
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.markdown("### Correlation Matrix")
                corr_matrix = processed_data[numeric_cols].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                                   title="Feature Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("👆 Please upload a CSV file to get started!")

def model_training_tab():
    st.markdown('<div class="sub-header">Model Training & Evaluation</div>', unsafe_allow_html=True)
    
    if 'processed_data' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first in the Data Analysis tab.")
        return
    
    data = st.session_state.processed_data
    
    if 'income' not in data.columns:
        st.error("❌ Dataset must contain 'income' column for training.")
        return
    
    st.markdown("### Training Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0)
    
    with col2:
        use_stratify = st.checkbox("Use Stratified Split", value=True)
        scale_features = st.checkbox("Scale Features", value=True)
    
    if st.button("🚀 Train Models"):
        with st.spinner("Training models... This may take a moment..."):
            try:
                # Encode categorical features
                encoded_data, label_encoders = encode_features(data)
                st.session_state.label_encoders = label_encoders
                
                # Prepare features and target
                X = encoded_data.drop(columns=['income'])
                y = encoded_data['income']
                st.session_state.feature_names = X.columns.tolist()
                
                # Split data
                stratify_param = y if use_stratify else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
                )
                
                # Scale features
                if scale_features:
                    scaler = MinMaxScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    st.session_state.scaler = scaler
                
                # Train models
                results, trained_models = train_models(X_train, X_test, y_train, y_test)
                
                st.session_state.models = trained_models
                st.session_state.model_results = results
                st.session_state.model_trained = True
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                st.success("✅ Models trained successfully!")
                
                # Display results
                st.markdown("### Model Performance")
                results_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Accuracy': [results[model]['accuracy'] for model in results.keys()]
                }).sort_values('Accuracy', ascending=False)
                
                # Best model highlight
                best_model = results_df.iloc[0]['Model']
                best_accuracy = results_df.iloc[0]['Accuracy']
                
                st.markdown(f"""
                <div class="top-model-box">
                    <h3>🏆 Best Model: {best_model}</h3>
                    <p><strong>Accuracy: {best_accuracy:.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Results table
                st.dataframe(results_df, use_container_width=True)
                
                fig = px.bar(results_df, x='Accuracy', y='Model', orientation='h',
                           title='Model Accuracy Comparison',
                           color='Accuracy', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")

def prediction_tab():
    st.markdown('<div class="sub-header">Make Salary Predictions</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the Model Training tab.")
        return
    
    # Model selection
    model_names = list(st.session_state.models.keys())
    selected_model = st.selectbox("Select Model for Prediction", model_names)
    
    st.markdown("### Enter Employee Information")
    
    available_features = st.session_state.feature_names
    
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    # Get feature options from processed data
    if 'processed_data' in st.session_state:
        data = st.session_state.processed_data
        
        with col1:
            st.markdown("#### **Numerical Features**")
            
            if 'age' in available_features:
                input_data['age'] = st.slider("Age", 17, 75, 35)
            
            if 'education-num' in available_features:
                input_data['education-num'] = st.slider("Education Years (1=Elementary, 16=PhD)", 1, 16, 10, 
                                                        help="1-6: Elementary, 7-9: Middle School, 10-12: High School, 13-16: College/University")
            
            if 'hours-per-week' in available_features:
                input_data['hours-per-week'] = st.slider("Hours per Week", 1, 100, 40)
            
            if 'capital-gain' in available_features:
                input_data['capital-gain'] = st.number_input("Capital Gain ($)", 0, 100000, 0)
            
            if 'capital-loss' in available_features:
                input_data['capital-loss'] = st.number_input("Capital Loss ($)", 0, 5000, 0)
        
        with col2:
            st.markdown("#### **Categorical Features**")
            
            categorical_features = [f for f in available_features if f in st.session_state.label_encoders]
            
            for feature in categorical_features:
                if feature in data.columns:
                    options = sorted(data[feature].unique())
                    input_data[feature] = st.selectbox(
                        feature.replace('-', ' ').replace('_', ' ').title(), 
                        options
                    )
    
    # Show available features info
    with st.expander("ℹ️ Available Features in Model"):
        st.write("The model was trained with these features:")
        st.write(available_features)
    
    if st.button("💰 Predict Salary"):
        try:
            input_df = pd.DataFrame([input_data])
            
            # Add missing features with default values
            for feature in available_features:
                if feature not in input_df.columns:
                    if feature in st.session_state.label_encoders:
                        input_df[feature] = 0 
                    else:
                        input_df[feature] = 0 
            
            # Ensure correct column order
            input_df = input_df.reindex(columns=available_features, fill_value=0)
            
            for col in input_df.columns:
                if col in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[col]
                    try:
                        if input_df[col].dtype == 'object':
                            input_df[col] = le.transform(input_df[col])
                    except ValueError as e:
                        st.warning(f"Unseen category in {col}. Using default value.")
                        input_df[col] = 0
            
            # Scale if scaler was used
            if st.session_state.scaler is not None:
                input_df = st.session_state.scaler.transform(input_df)
            
            # Make prediction
            model = st.session_state.models[selected_model]
            prediction = model.predict(input_df)[0]
            
            # Get prediction probability
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                confidence = max(proba) * 100
            else:
                confidence = "N/A"
            
            # Display result
            st.markdown("### 🎯 Prediction Result")
            
            is_high_income = (
                prediction == 1 or 
                prediction == '>50K' or 
                str(prediction).strip() == '>50K'
            )
            
            if is_high_income:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); 
                           padding: 2rem; border-radius: 10px; text-align: center; color: white;">
                    <h2>💰 Predicted Salary: >$50K</h2>
                    <p style="font-size: 1.2rem;">Model: {selected_model}</p>
                    <p>Confidence: {confidence if confidence != 'N/A' else 'N/A'}{'' if confidence == 'N/A' else '%'}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%); 
                           padding: 2rem; border-radius: 10px; text-align: center; color: #333;">
                    <h2>💼 Predicted Salary: ≤$50K</h2>
                    <p style="font-size: 1.2rem;">Model: {selected_model}</p>
                    <p>Confidence: {confidence if confidence != 'N/A' else 'N/A'}{'' if confidence == 'N/A' else '%'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show input summary
            st.markdown("### 📋 Input Summary")
            input_summary = pd.DataFrame([input_data]).T
            input_summary.columns = ['Value']
            st.dataframe(input_summary, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.write("**Debug Info:**")
            st.write(f"Available features: {st.session_state.feature_names}")
            st.write(f"Input data keys: {list(input_data.keys()) if 'input_data' in locals() else 'N/A'}")

def model_comparison_tab():
    st.markdown('<div class="sub-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the Model Training tab.")
        return
    
    results = st.session_state.model_results
    
    # Performance metrics
    st.markdown("### 📊 Model Performance Metrics")
    
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()]
    }).sort_values('Accuracy', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(results_df, x='Model', y='Accuracy', 
                    title='Model Accuracy Comparison',
                    color='Accuracy', color_continuous_scale='viridis')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏆 Top Models")
        for i, (_, row) in enumerate(results_df.head(3).iterrows()):
            medal = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"""
            <div class="top-model-box">
                {medal} <strong>{row['Model']}</strong><br>
                Accuracy: {row['Accuracy']:.4f}
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed classification reports
    st.markdown("### 📋 Detailed Model Analysis")
    
    selected_models = st.multiselect(
        "Select models for detailed analysis",
        list(results.keys()),
        default=list(results.keys())[:3]
    )
    
    for model_name in selected_models:
        if model_name in results:
            st.markdown(f"#### {model_name}")
            
            y_pred = results[model_name]['predictions']
            y_test = st.session_state.y_test
            
            report = classification_report(y_test, y_pred, output_dict=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Classification Report:**")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4))
            
            with col2:
                st.write("**Confusion Matrix:**")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                 title=f"Confusion Matrix - {model_name}")
                st.plotly_chart(fig_cm, use_container_width=True)
    
    # Model export
    st.markdown("### 💾 Export Best Model")
    
    best_model_name = results_df.iloc[0]['Model']
    
    if st.button("📤 Download Best Model"):
        best_model = st.session_state.models[best_model_name]
        
        model_data = {
            'model': best_model,
            'scaler': st.session_state.scaler,
            'label_encoders': st.session_state.label_encoders,
            'feature_names': st.session_state.feature_names,
            'model_name': best_model_name,
            'accuracy': results_df.iloc[0]['Accuracy']
        }
        
        buffer = io.BytesIO()
        pickle.dump(model_data, buffer)
        buffer.seek(0)
        
        st.download_button(
            label=f"Download {best_model_name} Model",
            data=buffer.getvalue(),
            file_name=f"best_salary_model_{best_model_name.replace(' ', '_').lower()}.pkl",
            mime="application/octet-stream"
        )
        
        st.success(f"✅ {best_model_name} model ready for download!")

if __name__ == "__main__":
    main()
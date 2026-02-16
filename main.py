# ==========================================
# DEPENDENCIES & INSTALLATION
# ==========================================
# pip install flask flask-sqlalchemy flask-login flask-jwt-extended flask-limiter bcrypt pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly fpdf apscheduler werkzeug

# ==========================================
# 1. IMPORTS & CONFIGURATION
# ==========================================
import os
import io
import base64
import random
import json
import logging
import datetime
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Critical for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.utils
import joblib
import bcrypt
from fpdf import FPDF
from functools import wraps
from collections import Counter
from apscheduler.schedulers.background import BackgroundScheduler

# Flask & Security
from flask import Flask, render_template_string, request, jsonify, send_file, redirect, url_for, flash, make_response, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Scikit-Learn & ML
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Removed SVC (SVM) and MLP for SPEED
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                             confusion_matrix, roc_curve, precision_recall_curve, 
                             classification_report)

# XGBoost & SHAP
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. GradientBoosting will be used instead.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Explainability features disabled.")

# ==========================================
# 2. APP CONFIGURATION
# ==========================================
app = Flask(__name__)

# Secret Keys (Change in production)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super_secret_enterprise_key_2024')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt_secret_key_change_me')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///enterprise_loan_risk.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=1)

# Initialize Extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
jwt = JWTManager(app)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

# Logger Configuration
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
logger = logging.getLogger(__name__)

# ==========================================
# 3. DATABASE MODELS
# ==========================================

class User(UserMixin, db.Model):
    """User model for Authentication and Role Management."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='User')  # Admin, Analyst, User
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    uploads = db.relationship('UploadedFile', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class ModelVersion(db.Model):
    """Stores metadata for trained ML models."""
    __tablename__ = 'model_versions'
    
    id = db.Column(db.Integer, primary_key=True)
    version_name = db.Column(db.String(50), unique=True, nullable=False)
    algorithm = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float)
    roc_auc = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    train_time = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    model_blob = db.Column(db.LargeBinary)  # Serialized model
    is_active = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<ModelVersion {self.version_name} - {self.algorithm}>'

class Prediction(db.Model):
    """Log of individual predictions made by users."""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    input_data = db.Column(db.Text)  # JSON string of inputs
    prediction_result = db.Column(db.Integer)  # 0 or 1
    probability = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    model_version = db.Column(db.String(50))
    ip_address = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class UploadedFile(db.Model):
    """Track bulk upload processing."""
    __tablename__ = 'uploaded_files'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    status = db.Column(db.String(20), default='Processing') # Processing, Completed, Failed
    rows_processed = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class AuditLog(db.Model):
    """Security and Activity Audit Log."""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# ==========================================
# 4. AUTHENTICATION & SECURITY UTILS
# ==========================================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'Admin':
            flash('Access denied. Administrator privileges required.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def analyst_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role not in ['Admin', 'Analyst']:
            flash('Access denied. Analyst privileges required.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def log_audit(action, details=None):
    """Helper to log actions to database."""
    ip = get_remote_address()
    user_id = current_user.id if current_user.is_authenticated else None
    log_entry = AuditLog(user_id=user_id, action=action, details=details, ip_address=ip)
    db.session.add(log_entry)
    db.session.commit()

# ==========================================
# 5. MACHINE LEARNING ENGINE
# ==========================================

class MLEngine:
    """
    Enterprise-grade ML Engine handling data generation, training, 
    versioning, and inference.
    """
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.active_version_name = None
        
        # Define Synthetic Data Features
        self.numeric_features = ['Age', 'Income', 'EmploymentLength', 'CreditScore', 'LoanAmount', 'LoanTerm', 'ExistingLoans', 'DebtToIncomeRatio']
        self.categorical_features = ['PropertyOwnership', 'MaritalStatus', 'LoanPurpose', 'Education']
        self.target = 'Default'

    def generate_synthetic_data(self, n_samples=3000): # REDUCED FROM 10000 TO 3000 FOR SPEED
        """Generates high-quality synthetic loan data."""
        np.random.seed(42)
        
        data = {
            'Age': np.random.randint(18, 70, n_samples),
            'Income': np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples), # Log-normal for income
            'EmploymentLength': np.random.randint(0, 35, n_samples),
            'CreditScore': np.random.normal(650, 100, n_samples),
            'LoanAmount': np.random.normal(20000, 10000, n_samples),
            'LoanTerm': np.random.choice([12, 24, 36, 48, 60, 84], n_samples),
            'ExistingLoans': np.random.randint(0, 6, n_samples),
            'DebtToIncomeRatio': np.random.beta(2, 5, n_samples) * 0.6, # Mostly < 0.5
            'PropertyOwnership': np.random.choice(['Owned', 'Mortgaged', 'Rented'], n_samples, p=[0.2, 0.5, 0.3]),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.5, 0.1]),
            'LoanPurpose': np.random.choice(['Home', 'Auto', 'Personal', 'Business', 'Education'], n_samples),
            'Education': np.random.choice(['HighSchool', 'Bachelors', 'Masters', 'PhD'], n_samples, p=[0.4, 0.4, 0.15, 0.05])
        }
        
        df = pd.DataFrame(data)
        
        # Ensure data types are correct
        df['CreditScore'] = df['CreditScore'].clip(300, 850)
        df['Income'] = df['Income'].clip(10000, 500000)
        df['LoanAmount'] = df['LoanAmount'].clip(1000, 100000)
        
        # Generate Target (Default) based on a weighted logic (simulating risk factors)
        risk_score = (
            (300 - df['CreditScore']) / 10 +       # Lower credit = higher risk
            (df['DebtToIncomeRatio'] * 50) +       # High DTI = higher risk
            (df['LoanAmount'] / df['Income']) * 20 - # High Loan/Income ratio = higher risk
            (df['ExistingLoans'] * 5) -            # More loans = higher risk
            (df['EmploymentLength'] * 0.5)         # Longer employment = lower risk
        )
        
        # Normalize and apply sigmoid
        noise = np.random.normal(0, 1.5, n_samples)
        prob = 1 / (1 + np.exp(-(risk_score + noise)))
        df['Default'] = (prob > 0.5).astype(int)
        
        # Imbalance the dataset (Defaults are usually rarer)
        default_indices = df[df['Default'] == 1].index
        keep_defaults = np.random.choice(default_indices, size=int(len(default_indices)*0.6), replace=False)
        drop_defaults = default_indices.difference(keep_defaults)
        df = df.drop(drop_defaults)
        
        return df.reset_index(drop=True)

    def create_pipeline(self, algorithm):
        """Creates a scikit-learn pipeline with preprocessing and the classifier."""
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        classifier = None
        if algorithm == 'LogisticRegression':
            classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        elif algorithm == 'RandomForest':
            classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        elif algorithm == 'GradientBoosting':
            classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif algorithm == 'XGBoost':
            if HAS_XGBOOST:
                classifier = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
            else:
                # Fallback
                classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        # REMOVED SVM AND MLP FOR SPEED
            
        if classifier:
            return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
        return None

    def train_and_evaluate(self):
        """Trains multiple models, evaluates them, and selects the best one."""
        logger.info("Starting Training Pipeline...")
        df = self.generate_synthetic_data() # Using reduced samples
        self.feature_names = self.numeric_features + self.categorical_features
        
        X = df.drop(self.target, axis=1)
        y = df[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        algorithms = ['LogisticRegression', 'RandomForest', 'GradientBoosting']
        if HAS_XGBOOST: algorithms.append('XGBoost')
        # REMOVED SVM AND MLP FROM LIST
        
        results = []
        best_model = None
        best_pipeline = None
        best_score = -1
        best_algo = ""
        
        for algo in algorithms:
            try:
                logger.info(f"Training {algo}...")
                pipeline = self.create_pipeline(algo)
                
                # Cross Validation - REDUCED FOLDS FROM 5 TO 2 FOR SPEED
                cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
                
                # Fit full training data
                pipeline.fit(X_train, y_train)
                
                # Metrics
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                
                acc = accuracy_score(y_test, y_pred)
                roc = roc_auc_score(y_test, y_proba)
                f1 = f1_score(y_test, y_pred)
                
                result = {
                    'algorithm': algo,
                    'accuracy': acc,
                    'roc_auc': roc,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                results.append(result)
                
                # Selection criteria: ROC AUC
                if roc > best_score:
                    best_score = roc
                    best_model = pipeline
                    best_algo = algo
                    best_pipeline = pipeline
                    
            except Exception as e:
                logger.error(f"Error training {algo}: {str(e)}")

        # Save Best Model
        if best_model:
            self.model = best_model
            self.preprocessor = best_model.named_steps['preprocessor']
            
            version_name = f"v{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{best_algo}"
            self.active_version_name = version_name
            
            # Save to DB
            buffer = io.BytesIO()
            joblib.dump(best_model, buffer)
            buffer.seek(0)
            
            # Deactivate old models
            ModelVersion.query.filter_by(is_active=True).update({'is_active': False})
            
            new_version = ModelVersion(
                version_name=version_name,
                algorithm=best_algo,
                accuracy=best_score, # Using ROC as primary, but mapped to acc field for simplicity
                roc_auc=best_score,
                f1_score=results[0]['f1_score'] if results else 0,
                train_time=0, 
                model_blob=buffer.read(),
                is_active=True
            )
            db.session.add(new_version)
            db.session.commit()
            
            logger.info(f"New Model Deployed: {version_name} with ROC-AUC: {best_score:.4f}")
            return results
        return None

    def load_active_model(self):
        """Loads the active model from database on startup."""
        active_version = ModelVersion.query.filter_by(is_active=True).first()
        if active_version:
            try:
                self.model = joblib.load(io.BytesIO(active_version.model_blob))
                self.preprocessor = self.model.named_steps['preprocessor']
                self.active_version_name = active_version.version_name
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        return False

    def predict_single(self, data_dict):
        """Predict for a single dictionary input."""
        if not self.model:
            raise Exception("Model not loaded")
            
        df_input = pd.DataFrame([data_dict])
        # Ensure all columns exist
        for col in self.numeric_features + self.categorical_features:
            if col not in df_input.columns:
                df_input[col] = 0 # Default or raise error
                
        prob = self.model.predict_proba(df_input)[0, 1]
        pred = int(prob > 0.5)
        return pred, prob

    def get_shap_explanation(self, data_dict):
        """Generate SHAP values for explainability."""
        if not self.model or not HAS_SHAP:
            return None
            
        try:
            df_input = pd.DataFrame([data_dict])
            # Preprocess
            X_processed = self.preprocessor.transform(df_input)
            
            # Use TreeExplainer for tree models
            algo = self.active_version_name.split('_')[-1]
            
            if algo in ['RandomForest', 'GradientBoosting', 'XGBoost']:
                explainer = shap.TreeExplainer(self.model.named_steps['classifier'])
            else:
                return None 
                
            shap_values = explainer.shap_values(X_processed)
            
            # Base64 Plotting
            plt.figure()
            shap.force_plot(explainer.expected_value[1], shap_values[1], feature_names=self.model.named_steps['preprocessor'].get_feature_names_out(), matplotlib=True, show=False)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"SHAP Error: {e}")
            return None

# Initialize Global Engine
ml_engine = MLEngine()

# ==========================================
# 6. UTILITY FUNCTIONS (PDF, PLOTS)
# ==========================================

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def generate_pdf_report(data, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Loan Risk Prediction Report", ln=1, align='C')
    
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.ln(5)
    
    # Input Data Table
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Applicant Details:", ln=1)
    pdf.set_font("Arial", '', 12)
    
    for key, value in data['input'].items():
        pdf.cell(200, 8, txt=f"{key}: {value}", ln=1)
        
    pdf.ln(10)
    
    # Result
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Prediction Outcome:", ln=1)
    pdf.set_font("Arial", '', 12)
    
    color_map = {'Low': (0, 150, 0), 'Medium': (200, 150, 0), 'High': (200, 0, 0)}
    r, g, b = color_map.get(data['risk_level'], (0, 0, 0))
    pdf.set_text_color(r, g, b)
    pdf.cell(200, 10, txt=f"Risk Level: {data['risk_level']}", ln=1)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt=f"Default Probability: {data['probability']:.2%}", ln=1)
    pdf.cell(200, 10, txt=f"Model Version: {data['model_version']}", ln=1)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 7. HTML TEMPLATES (INLINE)
# ==========================================

BASE_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise AI Risk Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-light: #f4f6f9;
            --text-dark: #343a40;
            --primary: #4e73df;
            --secondary: #858796;
            --success: #1cc88a;
            --info: #36b9cc;
            --warning: #f6c23e;
            --danger: #e74a3b;
        }
        [data-theme="dark"] {
            --bg-light: #222831;
            --text-dark: #e0e0e0;
            --primary: #4e73df;
        }
        body {
            background-color: var(--bg-light);
            color: var(--text-dark);
            font-family: 'Nunito', sans-serif;
            transition: background-color 0.3s, color 0.3s;
        }
        .sidebar {
            min-height: 100vh;
            background: linear-gradient(180deg, #4e73df 10%, #224abe 100%);
            color: white;
        }
        .sidebar .nav-link {
            color: rgba(255,255,255,0.8);
            margin-bottom: 5px;
        }
        .sidebar .nav-link:hover, .sidebar .nav-link.active {
            color: white;
            background: rgba(255,255,255,0.1);
            font-weight: bold;
        }
        .card {
            border: none;
            box-shadow: 0 .15rem 1.75rem 0 rgba(58,59,69,.15);
            margin-bottom: 20px;
        }
        .stat-card {
            border-left: 0.25rem solid;
        }
        .text-xs { font-size: .7rem; }
        .border-left-primary { border-left-color: var(--primary) !important; }
        .border-left-success { border-left-color: var(--success) !important; }
        .border-left-info { border-left-color: var(--info) !important; }
        .border-left-warning { border-left-color: var(--warning) !important; }
        /* Loading Spinner */
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar d-none d-md-block p-3">
                <h4 class="text-center mb-4">RiskAI <i class="fas fa-shield-alt"></i></h4>
                <ul class="nav flex-column">
                    <li class="nav-item"><a href="/" class="nav-link {{ 'active' if active_page == 'dashboard' else '' }}"><i class="fas fa-tachometer-alt me-2"></i> Dashboard</a></li>
                    <li class="nav-item"><a href="/predict" class="nav-link {{ 'active' if active_page == 'predict' else '' }}"><i class="fas fa-calculator me-2"></i> Prediction</a></li>
                    <li class="nav-item"><a href="/bulk" class="nav-link {{ 'active' if active_page == 'bulk' else '' }}"><i class="fas fa-file-upload me-2"></i> Bulk Upload</a></li>
                    {% if current_user.is_authenticated and current_user.role == 'Admin' %}
                    <li class="nav-item"><a href="/admin/models" class="nav-link {{ 'active' if active_page == 'models' else '' }}"><i class="fas fa-brain me-2"></i> Model Mgmt</a></li>
                    <li class="nav-item"><a href="/admin/logs" class="nav-link {{ 'active' if active_page == 'logs' else '' }}"><i class="fas fa-list me-2"></i> Audit Logs</a></li>
                    {% endif %}
                </ul>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 p-4">
                <!-- Topbar -->
                <nav class="navbar navbar-expand navbar-light bg-white mb-4 shadow-sm rounded px-3">
                    <button class="btn btn-link d-md-none rounded-circle mr-3" id="sidebarToggleTop">
                        <i class="fa fa-bars"></i>
                    </button>
                    <h5 class="mb-0 d-none d-sm-block text-capitalize">{{ active_page|replace('_', ' ') }}</h5>
                    
                    <ul class="navbar-nav ms-auto align-items-center">
                        <li class="nav-item me-3">
                            <button class="btn btn-sm btn-outline-secondary" onclick="toggleTheme()">
                                <i class="fas fa-moon"></i>
                            </button>
                        </li>
                        {% if current_user.is_authenticated %}
                        <li class="nav-item dropdown">
                            <a class="nav-link dropdown-toggle" href="#" id="userDropdown" role="button" data-bs-toggle="dropdown">
                                <span class="me-2 d-none d-lg-inline small">{{ current_user.username }}</span>
                                <i class="fas fa-user-circle fa-lg"></i>
                            </a>
                            <ul class="dropdown-menu dropdown-menu-end shadow">
                                <li><a class="dropdown-item" href="/logout"><i class="fas fa-sign-out-alt fa-sm fa-fw me-2 text-gray-400"></i> Logout</a></li>
                            </ul>
                        </li>
                        {% else %}
                        <li class="nav-item"><a href="/login" class="btn btn-primary btn-sm">Login</a></li>
                        {% endif %}
                    </ul>
                </nav>

                <!-- Flash Messages -->
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        </div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}

                <!-- Content Block -->
                {% block content %}{% endblock %}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme');
            const target = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', target);
        }
    </script>
</body>
</html>
"""

DASHBOARD_TEMPLATE = BASE_LAYOUT.replace('{% block content %}{% endblock %}', """
<div class="row">
    <!-- Stats -->
    <div class="col-xl-3 col-md-6 mb-4">
        <div class="card stat-card border-left-primary h-100 py-2">
            <div class="card-body">
                <div class="row no-gutters align-items-center">
                    <div class="col mr-2">
                        <div class="text-xs font-weight-bold text-primary text-uppercase mb-1">Total Predictions</div>
                        <div class="h5 mb-0 font-weight-bold text-gray-800">{{ stats.total_predictions }}</div>
                    </div>
                    <div class="col-auto"><i class="fas fa-calendar fa-2x text-gray-300"></i></div>
                </div>
            </div>
        </div>
    </div>

    <div class="col-xl-3 col-md-6 mb-4">
        <div class="card stat-card border-left-success h-100 py-2">
            <div class="card-body">
                <div class="row no-gutters align-items-center">
                    <div class="col mr-2">
                        <div class="text-xs font-weight-bold text-success text-uppercase mb-1">Model Accuracy</div>
                        <div class="h5 mb-0 font-weight-bold text-gray-800">{{ stats.accuracy|round(4) }}</div>
                    </div>
                    <div class="col-auto"><i class="fas fa-bullseye fa-2x text-gray-300"></i></div>
                </div>
            </div>
        </div>
    </div>

    <div class="col-xl-3 col-md-6 mb-4">
        <div class="card stat-card border-left-info h-100 py-2">
            <div class="card-body">
                <div class="row no-gutters align-items-center">
                    <div class="col mr-2">
                        <div class="text-xs font-weight-bold text-info text-uppercase mb-1">Active Model</div>
                        <div class="h5 mb-0 font-weight-bold text-gray-800 small">{{ stats.model_name }}</div>
                    </div>
                    <div class="col-auto"><i class="fas fa-robot fa-2x text-gray-300"></i></div>
                </div>
            </div>
        </div>
    </div>

    <div class="col-xl-3 col-md-6 mb-4">
        <div class="card stat-card border-left-warning h-100 py-2">
            <div class="card-body">
                <div class="row no-gutters align-items-center">
                    <div class="col mr-2">
                        <div class="text-xs font-weight-bold text-warning text-uppercase mb-1">High Risk %</div>
                        <div class="h5 mb-0 font-weight-bold text-gray-800">{{ stats.high_risk_pct }}%</div>
                    </div>
                    <div class="col-auto"><i class="fas fa-exclamation-triangle fa-2x text-gray-300"></i></div>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-lg-8 mb-4">
        <div class="card shadow mb-4">
            <div class="card-header py-3">
                <h6 class="m-0 font-weight-bold text-primary">Prediction Volume (Last 7 Days)</h6>
            </div>
            <div class="card-body">
                <div id="volumeChart" style="height: 300px;"></div>
            </div>
        </div>
    </div>
    <div class="col-lg-4 mb-4">
        <div class="card shadow mb-4">
            <div class="card-header py-3">
                <h6 class="m-0 font-weight-bold text-primary">Risk Distribution</h6>
            </div>
            <div class="card-body">
                <div id="riskChart" style="height: 300px;"></div>
            </div>
        </div>
    </div>
</div>

<script>
    // Simple Plotly Charts
    const volumeData = [{
        x: {{ stats.dates | safe }},
        y: {{ stats.counts | safe }},
        type: 'scatter',
        mode: 'lines+markers',
        line: {color: '#4e73df'}
    }];
    const volumeLayout = {margin: {t: 0, r: 0, l: 0, b: 0}};
    Plotly.newPlot('volumeChart', volumeData, volumeLayout, {responsive: true, displayModeBar: false});

    const riskData = [{
        values: {{ stats.risk_dist | safe }},
        labels: ['Low', 'Medium', 'High'],
        type: 'pie',
        marker: {colors: ['#1cc88a', '#f6c23e', '#e74a3b']}
    }];
    Plotly.newPlot('riskChart', riskData, {margin: {t: 0, r: 0, l: 0, b: 0}}, {responsive: true, displayModeBar: false});
</script>
<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
""")

PREDICT_TEMPLATE = BASE_LAYOUT.replace('{% block content %}{% endblock %}', """
<div class="row justify-content-center">
    <div class="col-lg-8">
        <div class="card shadow mb-4">
            <div class="card-header py-3">
                <h6 class="m-0 font-weight-bold text-primary">Single Loan Risk Assessment</h6>
            </div>
            <div class="card-body">
                <form id="predictForm" onsubmit="handlePrediction(event)">
                    <div class="row">
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Age</label>
                            <input type="number" class="form-control" name="Age" required min="18" max="100">
                        </div>
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Annual Income ($)</label>
                            <input type="number" class="form-control" name="Income" required min="0">
                        </div>
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Employment Length (Years)</label>
                            <input type="number" class="form-control" name="EmploymentLength" required min="0">
                        </div>
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Credit Score</label>
                            <input type="number" class="form-control" name="CreditScore" required min="300" max="850">
                        </div>
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Loan Amount ($)</label>
                            <input type="number" class="form-control" name="LoanAmount" required min="0">
                        </div>
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Loan Term (Months)</label>
                            <select class="form-select" name="LoanTerm">
                                <option value="12">12</option>
                                <option value="24">24</option>
                                <option value="36">36</option>
                                <option value="60">60</option>
                                <option value="84">84</option>
                            </select>
                        </div>
                         <div class="col-md-6 mb-3">
                            <label class="form-label">Debt to Income Ratio (0-1)</label>
                            <input type="number" step="0.01" class="form-control" name="DebtToIncomeRatio" required min="0" max="1">
                        </div>
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Existing Loans</label>
                            <input type="number" class="form-control" name="ExistingLoans" required min="0">
                        </div>
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Property Ownership</label>
                            <select class="form-select" name="PropertyOwnership">
                                <option value="Owned">Owned</option>
                                <option value="Mortgaged">Mortgaged</option>
                                <option value="Rented">Rented</option>
                            </select>
                        </div>
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Marital Status</label>
                            <select class="form-select" name="MaritalStatus">
                                <option value="Single">Single</option>
                                <option value="Married">Married</option>
                                <option value="Divorced">Divorced</option>
                            </select>
                        </div>
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Loan Purpose</label>
                            <select class="form-select" name="LoanPurpose">
                                <option value="Home">Home</option>
                                <option value="Auto">Auto</option>
                                <option value="Personal">Personal</option>
                                <option value="Business">Business</option>
                            </select>
                        </div>
                        <div class="col-md-6 mb-3">
                            <label class="form-label">Education</label>
                            <select class="form-select" name="Education">
                                <option value="HighSchool">HighSchool</option>
                                <option value="Bachelors">Bachelors</option>
                                <option value="Masters">Masters</option>
                                <option value="PhD">PhD</option>
                            </select>
                        </div>
                    </div>
                    <div class="text-center mt-4">
                        <button type="submit" class="btn btn-primary btn-lg w-50" id="predictBtn">Predict Risk</button>
                    </div>
                </form>
            </div>
        </div>
        
        <!-- Result Card -->
        <div class="card shadow mb-4 d-none" id="resultCard">
            <div class="card-header py-3 d-flex justify-content-between align-items-center">
                <h6 class="m-0 font-weight-bold text-primary">Analysis Result</h6>
                <a href="#" onclick="downloadPDF()" class="btn btn-sm btn-outline-dark"><i class="fas fa-file-pdf"></i> PDF</a>
            </div>
            <div class="card-body">
                <div class="row text-center">
                    <div class="col-4">
                        <div class="text-uppercase text-xs mb-1">Risk Level</div>
                        <h3 class="font-weight-bold" id="resRisk">--</h3>
                    </div>
                    <div class="col-4">
                        <div class="text-uppercase text-xs mb-1">Probability</div>
                        <h3 class="font-weight-bold" id="resProb">--</h3>
                    </div>
                    <div class="col-4">
                        <div class="text-uppercase text-xs mb-1">Decision</div>
                        <h3 class="font-weight-bold" id="resDecision">--</h3>
                    </div>
                </div>
                <hr>
                <div id="shapArea" class="text-center mt-3"></div>
            </div>
        </div>
    </div>
</div>

<script>
    let currentPrediction = null;

    async function handlePrediction(e) {
        e.preventDefault();
        const btn = document.getElementById('predictBtn');
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Processing...';
        btn.disabled = true;

        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const json = await res.json();
            
            if (json.success) {
                displayResult(json.data, data);
            } else {
                alert(json.message);
            }
        } catch (err) {
            console.error(err);
            alert('Server Error');
        }
        
        btn.innerHTML = 'Predict Risk';
        btn.disabled = false;
    }

    function displayResult(data, inputData) {
        currentPrediction = { ...data, input: inputData };
        document.getElementById('resultCard').classList.remove('d-none');
        document.getElementById('resRisk').innerText = data.risk_level;
        document.getElementById('resProb').innerText = (data.probability * 100).toFixed(2) + '%';
        
        const decision = data.prediction === 1 ? 'REJECT' : 'APPROVE';
        const decEl = document.getElementById('resDecision');
        decEl.innerText = decision;
        decEl.className = 'font-weight-bold ' + (decision === 'APPROVE' ? 'text-success' : 'text-danger');

        if (data.shap_image) {
            document.getElementById('shapArea').innerHTML = '<h6 class="text-start mb-2">Feature Impact (SHAP)</h6><img src="data:image/png;base64,' + data.shap_image + '" class="img-fluid rounded border">';
        }
    }

    async function downloadPDF() {
        if (!currentPrediction) return;
        const res = await fetch('/api/pdf', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(currentPrediction)
        });
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'loan_report.pdf';
        a.click();
    }
</script>
""")

BULK_TEMPLATE = BASE_LAYOUT.replace('{% block content %}{% endblock %}', """
<div class="row">
    <div class="col-lg-8">
        <div class="card shadow mb-4">
            <div class="card-header py-3">
                <h6 class="m-0 font-weight-bold text-primary">Bulk CSV Upload</h6>
            </div>
            <div class="card-body">
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> Upload a CSV file with headers: 
                    <code>Age, Income, EmploymentLength, CreditScore, LoanAmount, LoanTerm, ExistingLoans, DebtToIncomeRatio, PropertyOwnership, MaritalStatus, LoanPurpose, Education</code>
                </div>
                <form action="/bulk/upload" method="POST" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label class="form-label">Select CSV File</label>
                        <input type="file" class="form-control" name="file" accept=".csv" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Process File</button>
                </form>
            </div>
        </div>
    </div>
    
    <div class="col-lg-4">
        <div class="card shadow mb-4">
            <div class="card-header py-3">
                <h6 class="m-0 font-weight-bold text-primary">Recent Uploads</h6>
            </div>
            <div class="card-body">
                <ul class="list-group list-group-flush">
                    {% for upload in uploads %}
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        {{ upload.filename }}
                        <span class="badge bg-{{ 'success' if upload.status == 'Completed' else 'secondary' }}">{{ upload.status }}</span>
                    </li>
                    {% else %}
                    <li class="list-group-item text-muted">No uploads yet.</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
    </div>
</div>
""")

ADMIN_MODELS_TEMPLATE = BASE_LAYOUT.replace('{% block content %}{% endblock %}', """
<div class="card shadow mb-4">
    <div class="card-header py-3 d-flex justify-content-between align-items-center">
        <h6 class="m-0 font-weight-bold text-primary">Model Version Control</h6>
        <button onclick="triggerRetrain()" class="btn btn-warning btn-sm"><i class="fas fa-sync"></i> Retrain Now</button>
    </div>
    <div class="card-body">
        <div class="table-responsive">
            <table class="table table-bordered table-hover">
                <thead class="table-light">
                    <tr>
                        <th>Version</th>
                        <th>Algorithm</th>
                        <th>ROC-AUC</th>
                        <th>Created</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for v in versions %}
                    <tr class="{{ 'table-primary' if v.is_active else '' }}">
                        <td>{{ v.version_name }}</td>
                        <td>{{ v.algorithm }}</td>
                        <td>{{ v.roc_auc|round(4) }}</td>
                        <td>{{ v.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
                        <td>
                            {% if v.is_active %}
                            <span class="badge bg-success">Active</span>
                            {% else %}
                            <span class="badge bg-secondary">Inactive</span>
                            {% endif %}
                        </td>
                        <td>
                            {% if not v.is_active %}
                            <a href="/admin/activate/{{ v.id }}" class="btn btn-sm btn-outline-primary">Rollback</a>
                            {% endif %}
                            <a href="/admin/delete/{{ v.id }}" class="btn btn-sm btn-outline-danger" onclick="return confirm('Delete this model?')">Delete</a>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
    async function triggerRetrain() {
        if(!confirm("Start full model retraining? This may take several minutes.")) return;
        const btn = event.target;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Training...';
        
        try {
            const res = await fetch('/api/admin/retrain', { method: 'POST' });
            const json = await res.json();
            alert(json.message);
            location.reload();
        } catch(e) {
            alert("Error starting training");
            btn.disabled = false;
        }
    }
</script>
""")

LOGIN_TEMPLATE = """
<div class="container d-flex align-items-center justify-content-center" style="height: 100vh; background: linear-gradient(135deg, #4e73df 0%, #224abe 100%);">
    <div class="card shadow p-4" style="width: 400px;">
        <div class="text-center mb-4">
            <h3 class="text-primary"><i class="fas fa-shield-alt"></i> RiskAI</h3>
            <p class="text-muted">Enterprise Login</p>
        </div>
        <form method="POST">
            <div class="mb-3">
                <label class="form-label">Username</label>
                <input type="text" name="username" class="form-control" required>
            </div>
            <div class="mb-3">
                <label class="form-label">Password</label>
                <input type="password" name="password" class="form-control" required>
            </div>
            <button type="submit" class="btn btn-primary w-100">Login</button>
        </form>
        <div class="mt-3 text-center">
            <a href="/register" class="text-decoration-none small">Create an account</a>
        </div>
    </div>
</div>
"""

# ==========================================
# 8. API ROUTES (REST)
# ==========================================

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    if User.query.filter_by(username=data.get('username')).first():
        return jsonify({'success': False, 'message': 'User already exists'}), 400
    
    user = User(username=data['username'], email=data.get('email', ''))
    user.set_password(data['password'])
    user.role = data.get('role', 'User')
    db.session.add(user)
    db.session.commit()
    
    log_audit('API_REGISTER', f"User {user.username} created via API")
    return jsonify({'success': True, 'message': 'User created'}), 201

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    
    if user and user.check_password(data['password']):
        access_token = create_access_token(identity=user.id)
        log_audit('API_LOGIN', f"User {user.username} logged in")
        return jsonify({'success': True, 'token': access_token, 'role': user.role}), 200
    
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def api_predict():
    try:
        data = request.json
        
        # Basic Validation
        required_fields = ['Age', 'Income', 'CreditScore', 'LoanAmount']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'message': f'Missing field: {field}'}), 400
        
        # ML Prediction
        prediction, prob = ml_engine.predict_single(data)
        
        # Determine Risk
        if prob < 0.35: risk = "Low"
        elif prob < 0.65: risk = "Medium"
        else: risk = "High"
        
        # SHAP
        shap_img = None
        # Only compute SHAP if user is authenticated and authorized, or for demo purposes occasionally
        # to save CPU
        if random.random() > 0.7: 
            shap_img = ml_engine.get_shap_explanation(data)

        # Log to DB (if user session exists)
        user_id = None
        if current_user.is_authenticated:
            user_id = current_user.id
        
        # Log Prediction
        pred_log = Prediction(
            user_id=user_id,
            input_data=json.dumps(data),
            prediction_result=prediction,
            probability=float(prob),
            risk_level=risk,
            model_version=ml_engine.active_version_name,
            ip_address=get_remote_address()
        )
        db.session.add(pred_log)
        db.session.commit()
        
        response_data = {
            'prediction': int(prediction),
            'probability': float(prob),
            'risk_level': risk,
            'model_version': ml_engine.active_version_name,
            'shap_image': shap_img
        }
        
        return jsonify({'success': True, 'data': response_data})
        
    except Exception as e:
        logger.error(f"Prediction API Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/pdf', methods=['POST'])
def api_pdf():
    try:
        data = request.json
        pdf_bytes = generate_pdf_report(data)
        return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf', as_attachment=True, download_name='loan_report.pdf')
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/retrain', methods=['POST'])
@jwt_required()
def api_retrain():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user or user.role != 'Admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
    # Run training in background thread would be better, but for simplicity:
    results = ml_engine.train_and_evaluate()
    return jsonify({'success': True, 'message': 'Retraining completed', 'results': results})

# ==========================================
# 9. WEB ROUTES (FRONTEND)
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            log_audit('WEB_LOGIN', f"User {username} logged in")
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'danger')
            
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'warning')
        else:
            user = User(username=username, email=f'{username}@example.com')
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
            
    return render_template_string(LOGIN_TEMPLATE.replace('Login', 'Register').replace('Enterprise Login', 'Create Account'))

@app.route('/logout')
@login_required
def logout():
    log_audit('WEB_LOGOUT', f"User {current_user.username} logged out")
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    # Calculate Stats
    total_preds = Prediction.query.count()
    active_model = ModelVersion.query.filter_by(is_active=True).first()
    accuracy = active_model.accuracy if active_model else 0.0
    
    # High Risk Count
    high_risk_count = Prediction.query.filter(Prediction.risk_level == 'High').count()
    high_risk_pct = (high_risk_count / total_preds * 100) if total_preds > 0 else 0
    
    # Time Series Data (Last 7 days)
    dates = []
    counts = []
    for i in range(6, -1, -1):
        d = datetime.datetime.utcnow() - datetime.timedelta(days=i)
        dates.append(d.strftime('%Y-%m-%d'))
        c = Prediction.query.filter(Prediction.timestamp >= d, Prediction.timestamp < d + datetime.timedelta(days=1)).count()
        counts.append(c)
        
    # Risk Dist
    low = Prediction.query.filter(Prediction.risk_level == 'Low').count()
    med = Prediction.query.filter(Prediction.risk_level == 'Medium').count()
    high = high_risk_count
    
    stats = {
        'total_predictions': total_preds,
        'accuracy': accuracy,
        'model_name': active_model.algorithm if active_model else 'None',
        'high_risk_pct': round(high_risk_pct, 1),
        'dates': json.dumps(dates),
        'counts': json.dumps(counts),
        'risk_dist': json.dumps([low, med, high])
    }
    
    return render_template_string(DASHBOARD_TEMPLATE, active_page='dashboard', stats=stats)

@app.route('/predict')
@login_required
def predict_page():
    return render_template_string(PREDICT_TEMPLATE, active_page='predict')

@app.route('/bulk')
@login_required
def bulk_page():
    uploads = UploadedFile.query.order_by(UploadedFile.created_at.desc()).limit(5).all()
    return render_template_string(BULK_TEMPLATE, active_page='bulk', uploads=uploads)

@app.route('/bulk/upload', methods=['POST'])
@login_required
def bulk_upload():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('bulk_page'))
        
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('bulk_page'))
        
    if file and file.filename.endswith('.csv'):
        upload_rec = UploadedFile(user_id=current_user.id, filename=file.filename, status='Processing')
        db.session.add(upload_rec)
        db.session.commit()
        
        try:
            df = pd.read_csv(file)
            # Process
            results = []
            for _, row in df.iterrows():
                data = row.to_dict()
                try:
                    pred, prob = ml_engine.predict_single(data)
                    risk = 'Low' if prob < 0.35 else ('Medium' if prob < 0.65 else 'High')
                    results.append({**data, 'Prediction': pred, 'Probability': prob, 'Risk_Level': risk})
                except:
                    results.append({**data, 'Error': 'Processing Failed'})
            
            # Save to Output CSV
            output = io.StringIO()
            pd.DataFrame(results).to_csv(output, index=False)
            output.seek(0)
            
            upload_rec.status = 'Completed'
            upload_rec.rows_processed = len(results)
            db.session.commit()
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                attachment_filename=f'results_{file.filename}'
            )
        except Exception as e:
            upload_rec.status = 'Failed'
            db.session.commit()
            logger.error(f"Bulk Upload Error: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(url_for('bulk_page'))
            
    flash('Invalid file type', 'danger')
    return redirect(url_for('bulk_page'))

# Admin Routes
@app.route('/admin/models')
@login_required
@admin_required
def admin_models():
    versions = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
    return render_template_string(ADMIN_MODELS_TEMPLATE, active_page='models', versions=versions)

@app.route('/admin/activate/<int:id>')
@login_required
@admin_required
def activate_model(id):
    version = ModelVersion.query.get_or_404(id)
    # Deactivate all
    ModelVersion.query.update({'is_active': False})
    # Activate selected
    version.is_active = True
    db.session.commit()
    
    # Reload engine
    ml_engine.load_active_model()
    log_audit('MODEL_ROLLBACK', f"Model {version.version_name} activated by {current_user.username}")
    flash(f'Model {version.version_name} activated.', 'success')
    return redirect(url_for('admin_models'))

@app.route('/admin/delete/<int:id>')
@login_required
@admin_required
def delete_model(id):
    version = ModelVersion.query.get_or_404(id)
    if version.is_active:
        flash('Cannot delete active model.', 'danger')
    else:
        db.session.delete(version)
        db.session.commit()
        flash('Model deleted.', 'success')
    return redirect(url_for('admin_models'))

@app.route('/admin/logs')
@login_required
@admin_required
def audit_logs():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(100).all()
    return render_template_string(ADMIN_MODELS_TEMPLATE.replace('Model Version Control', 'Audit Logs').replace('{% block content %}{% endblock %}', """
    <div class="card shadow mb-4">
        <div class="card-header py-3"><h6 class="m-0 font-weight-bold text-primary">System Logs</h6></div>
        <div class="card-body">
            <table class="table table-sm">
                <thead><tr><th>Time</th><th>User</th><th>Action</th><th>IP</th></tr></thead>
                <tbody>
                    {% for log in logs %}
                    <tr>
                        <td>{{ log.timestamp }}</td>
                        <td>{{ log.user_id or 'System' }}</td>
                        <td>{{ log.action }}</td>
                        <td>{{ log.ip_address }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    """), active_page='logs', logs=logs)

# ==========================================
# 10. BACKGROUND TASKS & MAIN
# ==========================================

def scheduled_retrain():
    with app.app_context():
        logger.info("Starting scheduled retrain job...")
        try:
            ml_engine.train_and_evaluate()
            logger.info("Scheduled retrain completed successfully.")
        except Exception as e:
            logger.error(f"Scheduled retrain failed: {e}")

if __name__ == '__main__':
    # Initialize DB
    with app.app_context():
        db.create_all()
        
        # Create Default Admin if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', email='admin@enterprise.com', role='Admin')
            admin.set_password('admin123') # CHANGE THIS IN PRODUCTION
            db.session.add(admin)
            db.session.commit()
            logger.info("Default admin user created (admin/admin123)")
            
        # Load or Train Model
        if not ml_engine.load_active_model():
            logger.info("No active model found. Training initial model...")
            ml_engine.train_and_evaluate()
            
    # Setup Scheduler for Auto-retrain (every 24 hours)
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=scheduled_retrain, trigger="interval", hours=24)
    scheduler.start()
    
    try:
        # Run App
        # In production, use Gunicorn: gunicorn -w 4 -b 0.0.0.0:8000 main:app
        app.run(host='0.0.0.0', port=8000, debug=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
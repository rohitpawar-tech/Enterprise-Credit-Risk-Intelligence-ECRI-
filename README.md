# Enterprise-Credit-Risk-Intelligence-ECRI-
Enterprise Credit Risk Intelligence (ECRI) is a scalable web platform for automated loan risk prediction using XGBoost and Random Forest. It delivers real-time insights, audit tracking, and bulk processing for financial institutions.

Enterprise Credit Risk Intelligence (ECRI)
Overview
Enterprise Credit Risk Intelligence (ECRI) is a scalable, production-grade web application for automated loan risk assessment. It leverages advanced machine learning models, including XGBoost and Random Forest, to predict loan default probabilities. The platform provides financial institutions with real-time decision support, comprehensive audit trails, and bulk processing capabilities.


#Key Features
Advanced Machine Learning: Implements and compares XGBoost, Random Forest, and Logistic Regression to select the best performing model.
Model Version Control: Tracks training history, accuracy metrics, and supports automatic model rollback.
Explainability (SHAP): Provides visual interpretations of individual predictions to explain why a loan was flagged as high risk.
Role-Based Access Control: Secure authentication system with distinct roles for Admin, Analyst, and User.
Real-Time Dashboard: Interactive analytics displaying prediction volumes, accuracy metrics, and risk distributions.
Bulk Processing: Supports high-volume CSV upload for batch risk scoring.
Audit Logging: Comprehensive tracking of user actions, IP addresses, and system changes for compliance.


Technology Stack:
Backend: Python, Flask, SQLAlchemy, Flask-Login, Flask-JWT-Extended
Machine Learning: Scikit-Learn, XGBoost, SHAP, Pandas, NumPy
Frontend: HTML5, Bootstrap 5, JavaScript (Fetch API)
Database: SQLite (configurable for PostgreSQL/MySQL)
Utilities: APScheduler (background tasks), FPDF (report generation)

Installation
1)Clone the repository:
git clone [repository-url]cd [project-directory]

2)Create a virtual environment (Recommended):
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate


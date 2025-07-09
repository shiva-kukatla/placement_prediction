# placement_prediction
PLACEMENT PREDICTION APP

A Flask-based web application that predicts student placement based on academic performance and skills using Machine Learning.

FEATURES:
- Predicts placement chances based on input data (CGPA, skills, etc.)
- Built with Flask, HTML/CSS, and trained ML models
- User-friendly interface
- Supports model retraining and accuracy tracking

TECH STACK:
- Python
- Flask
- Scikit-learn / XGBoost / MLPClassifier
- HTML + CSS
- Pandas, NumPy

FOLDER STRUCTURE:
placement_flask_app/
├── app.py
├── model.pkl
├── static/
├── templates/
├── placement.csv
├── README.txt
├── requirements.txt
└── .gitignore

HOW TO RUN LOCALLY:

1. Clone the repository:
   git clone https://github.com/shiva-kukatla/placement_prediction.git

2. Navigate to the project folder:
   cd placement_prediction

3. Install dependencies:
   pip install -r requirements.txt

4. Run the Flask app:
   python app.py

5. Open your browser and visit:
   http://127.0.0.1:5000

ML MODEL:
The application uses a trained ML model (MLP + XGBoost) to predict placement outcomes with up to 90% accuracy.

CONTACT:
Created by: Shiva Kukatla
GitHub: https://github.com/shiva-kukatla

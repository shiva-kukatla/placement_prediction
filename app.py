from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("placement_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Feedback threshold definitions (excluding SSC and HSC)
THRESHOLDS = {
    "cgpa"        : (7.0,  "📘 Improve your CGPA to at least 7.0 for better academic impact."),
    "internships" : (1 ,   "💼 Complete at least one internship to boost practical experience."),
    "projects"    : (2 ,   "🛠️ Add more projects (minimum 2) to showcase hands-on skills."),
    "workshops"   : (1 ,   "📄 Attend more workshops/certifications to stand out in your resume."),
    "aptitude"    : (70,   "🧠 Raise your aptitude score to 70+ for better test performance."),
    "softskills"  : (3.5,  "🗣️ Improve your soft skills (communication/teamwork) to at least 3.5."),
    "extra"       : (1 ,   "🎯 Engage in extracurricular activities to show leadership or initiative."),
    "training"    : (1 ,   "🏫 Participate in placement training to better prepare for interviews.")
}

@app.route("/", methods=["GET", "POST"])
def index():
    result, feedback = None, []
    if request.method == "POST":
        # Collect inputs
        cgpa = float(request.form["cgpa"])
        internships = int(request.form["internships"])
        projects = int(request.form["projects"])
        workshops = int(request.form["workshops"])
        aptitude = int(request.form["aptitude"])
        softskills = float(request.form["softskills"])
        extra = 1 if request.form["extra"] == "Yes" else 0
        training = 1 if request.form["training"] == "Yes" else 0
        ssc = int(request.form["ssc"])
        hsc = int(request.form["hsc"])

        # Feature engineering
        cgpa_apt = cgpa * aptitude
        proj_work = projects + workshops
        soft_apt = softskills * aptitude
        academic_avg = (ssc + hsc) / 2
        final_score = cgpa * softskills + proj_work

        # Model prediction
        features = np.array([[cgpa, internships, projects, workshops, aptitude,
                              softskills, extra, training, ssc, hsc,
                              cgpa_apt, proj_work, soft_apt, academic_avg, final_score]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        # Feedback only if Not Placed
        if prediction == 0:
            input_vals = {
                "cgpa": cgpa,
                "internships": internships,
                "projects": projects,
                "workshops": workshops,
                "aptitude": aptitude,
                "softskills": softskills,
                "extra": extra,
                "training": training
            }
            for key, (min_val, message) in THRESHOLDS.items():
                if input_vals[key] < min_val:
                    feedback.append(message)

        result = "🎉 Placed" if prediction == 1 else "❌ Not Placed"

    return render_template("index.html", result=result, feedback=feedback)

if __name__ == "__main__":
    app.run(debug=True)

# 🎓 PlacePredict — Student Placement Prediction System

A machine learning web app that predicts college student placement probability using a **Random Forest classifier** trained on 2,000 student records with **96.25% accuracy**.

Built for **College Project Expo** using Python, Flask, and scikit-learn.

---

## 🚀 Deploy Free on Render.com (Get a Public URL)

### Step 1 — Upload to GitHub
1. Create a free account at [github.com](https://github.com)
2. Click **"New Repository"** → name it `placement-predictor` → Public
3. Upload all these files to the repo (drag & drop on GitHub)

### Step 2 — Deploy on Render
1. Go to [render.com](https://render.com) and sign up free (use GitHub login)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account → select your `placement-predictor` repo
4. Fill in these settings:
   - **Name:** placement-predictor
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `gunicorn app:app`
5. Click **"Create Web Service"**
6. Wait ~3 minutes → you'll get a URL like: `https://placement-predictor.onrender.com`

**That's your public URL! Share it with anyone.**

---

## 💻 Run Locally (For Testing)

```bash
# Install dependencies
pip install flask scikit-learn numpy pandas

# Train the model
python train_model.py

# Run the app
python app.py
```

Open: http://localhost:5000

---

## 📁 Project Structure

```
placement-predictor/
├── app.py              # Flask web server + prediction API
├── train_model.py      # ML model training script
├── model.pkl           # Trained Random Forest model
├── branch_encoder.pkl  # Label encoder for branch
├── comm_encoder.pkl    # Label encoder for communication
├── requirements.txt    # Python dependencies
├── Procfile            # Render deployment config
├── build.sh            # Build script
└── templates/
    └── index.html      # Frontend UI
```

---

## 🤖 ML Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| Training samples | 2,000 students |
| Test accuracy | 96.25% |
| Features | 13 (CGPA, %, backlogs, internships, skills...) |
| Top feature | CGPA (16.5% importance) |

### Features Used
1. Branch / Stream
2. CGPA
3. 12th Percentage
4. 10th Percentage
5. Number of Backlogs
6. Internships Count
7. Projects Count
8. Aptitude Score
9. Communication Level
10. Technical Skills Count
11. Extra Activities Count
12. Certifications Count
13. Hackathon Participation

---

## 🎯 For Expo Presentation

**Talking points for judges:**
- "We trained a Random Forest on 2,000 synthetic student records modeled after real placement patterns"
- "The model achieves 96.25% accuracy on unseen test data"
- "CGPA is the most important feature (16.5%), followed by aptitude score (13%) and tech skills (10.3%)"
- "The system gives personalized insights — strengths, improvement areas, target companies, and a 30-day action plan"
- "Deployed as a live public web app accessible from any device"

---

Made with Python · Flask · scikit-learn · Deployed on Render

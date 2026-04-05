from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'branch_encoder.pkl'), 'rb') as f:
    le_branch = pickle.load(f)
with open(os.path.join(BASE_DIR, 'comm_encoder.pkl'), 'rb') as f:
    le_comm = pickle.load(f)

BRANCH_MAP = {
    'Computer Science & Engineering': 'CSE',
    'Information Technology': 'IT',
    'Electronics & Communication': 'ECE',
    'Mechanical Engineering': 'ME',
    'Civil Engineering': 'CE',
    'Electrical Engineering': 'EE',
    'Data Science & AI': 'DS_AI',
    'MBA / Management': 'MBA',
    'BCA / MCA': 'BCA_MCA'
}

COMPANY_MAP = {
    'CSE':     ['Google', 'Microsoft', 'Amazon', 'Infosys', 'TCS', 'Wipro', 'Accenture', 'Cognizant'],
    'IT':      ['TCS', 'Infosys', 'Wipro', 'Capgemini', 'HCL', 'Tech Mahindra', 'Accenture'],
    'DS_AI':   ['Amazon', 'Flipkart', 'Fractal Analytics', 'Mu Sigma', 'IBM', 'KPMG', 'Deloitte'],
    'ECE':     ['Qualcomm', 'Texas Instruments', 'Samsung', 'Bosch', 'L&T', 'Infosys BPM'],
    'BCA_MCA': ['TCS', 'Infosys', 'Wipro', 'Mphasis', 'Hexaware', 'Persistent Systems'],
    'MBA':     ['Deloitte', 'KPMG', 'EY', 'Accenture', 'Amazon', 'Flipkart', 'HDFC Bank'],
    'ME':      ['L&T', 'Tata Motors', 'Mahindra', 'Bosch', 'Siemens', 'John Deere'],
    'EE':      ['Siemens', 'ABB', 'L&T', 'BHEL', 'Schneider Electric', 'Honeywell'],
    'CE':      ['L&T Construction', 'DLF', 'Shapoorji Pallonji', 'Tata Projects', 'AECOM']
}

def get_insights(data, probability, branch_code):
    strengths, improvements, actions = [], [], []

    cgpa = data['cgpa']
    hsc = data['hsc_percent']
    backlogs = data['backlogs']
    internships = data['internships']
    projects = data['projects']
    aptitude = data['aptitude_score']
    comm = data['comm']
    tech = data['tech_skills_count']
    extra = data['extra_activities']

    if cgpa >= 8.0:
        strengths.append(f"Strong CGPA of {cgpa} — above average academic performance")
    if internships >= 2:
        strengths.append(f"{internships} internships demonstrate excellent industry exposure")
    if tech >= 5:
        strengths.append(f"Broad technical skill set ({tech} skills) makes you versatile")
    if projects >= 3:
        strengths.append(f"{projects} projects showcase hands-on problem-solving ability")
    if comm in ['good', 'excellent']:
        strengths.append(f"{comm.capitalize()} communication skills are a strong differentiator")
    if aptitude >= 70:
        strengths.append(f"High aptitude score ({aptitude}%) — strong for written tests")
    if extra >= 3:
        strengths.append("Active extracurricular participation shows well-rounded personality")
    if backlogs == 0:
        strengths.append("Clean academic record with no backlogs")

    if cgpa < 7.0:
        improvements.append(f"CGPA of {cgpa} is below 7.0 — focus on improving grades this semester")
    if backlogs > 0:
        improvements.append(f"Clear your {backlogs} active backlog(s) — many companies have strict criteria")
    if internships == 0:
        improvements.append("No internship experience — apply immediately on Internshala or LinkedIn")
    if tech < 3:
        improvements.append("Learn at least 2–3 in-demand tech skills relevant to your branch")
    if comm in ['poor', 'average']:
        improvements.append("Improve communication skills — join a public speaking club or GD practice group")
    if aptitude < 60:
        improvements.append(f"Aptitude score of {aptitude}% needs improvement — practice on IndiaBIX, RS Aggarwal")
    if projects < 2:
        improvements.append("Build at least 2 end-to-end projects and host them on GitHub")

    if probability < 50:
        actions = [
            "Do 30 mins of aptitude practice daily (IndiaBIX / RS Aggarwal)",
            "Start one project this week using a real dataset or problem",
            "Apply to at least 3 internship listings on Internshala this week",
            "Attend mock GD/PI sessions at your placement cell",
            "Get at least one online certification (NPTEL, Coursera, or Google)"
        ]
    elif probability < 75:
        actions = [
            "Polish your resume — add all projects, internships, and certifications",
            "Practice 5 LeetCode easy/medium problems per week",
            "Research top 10 companies visiting your campus and their selection process",
            "Prepare 3–4 strong answers for HR interview questions",
            "Network on LinkedIn with alumni from your college for referrals"
        ]
    else:
        actions = [
            "Target top-tier companies — your profile is competitive",
            "Prepare system design and advanced DSA for tech interviews",
            "Build a strong LinkedIn profile and GitHub portfolio",
            "Participate in a hackathon or coding contest to add to your profile",
            "Connect with seniors already placed in your target companies"
        ]

    companies = COMPANY_MAP.get(branch_code, COMPANY_MAP['CSE'])
    if probability < 50:
        companies = companies[3:]
    elif probability < 75:
        companies = companies[2:]
    else:
        companies = companies[:5]

    return strengths[:4], improvements[:4], actions[:4], companies

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json()

        branch_label = d.get('branch', 'CSE')
        branch_code = BRANCH_MAP.get(branch_label, 'CSE')

        try:
            branch_enc = le_branch.transform([branch_code])[0]
        except:
            branch_enc = 0

        comm = d.get('comm', 'average')
        try:
            comm_enc = le_comm.transform([comm])[0]
        except:
            comm_enc = 1

        features = np.array([[
            branch_enc,
            float(d.get('cgpa', 7.0)),
            float(d.get('hsc_percent', 75)),
            float(d.get('ssc_percent', 80)),
            int(d.get('backlogs', 0)),
            int(d.get('internships', 0)),
            int(d.get('projects', 0)),
            float(d.get('aptitude_score', 65)),
            comm_enc,
            int(d.get('tech_skills_count', 3)),
            int(d.get('extra_activities', 1)),
            int(d.get('certifications', 1)),
            int(d.get('hackathons', 0))
        ]])

        probability = float(model.predict_proba(features)[0][1]) * 100
        prediction = int(model.predict(features)[0])

        data_for_insights = {
            'cgpa': float(d.get('cgpa', 7.0)),
            'hsc_percent': float(d.get('hsc_percent', 75)),
            'backlogs': int(d.get('backlogs', 0)),
            'internships': int(d.get('internships', 0)),
            'projects': int(d.get('projects', 0)),
            'aptitude_score': float(d.get('aptitude_score', 65)),
            'comm': comm,
            'tech_skills_count': int(d.get('tech_skills_count', 3)),
            'extra_activities': int(d.get('extra_activities', 1)),
        }

        strengths, improvements, actions, companies = get_insights(
            data_for_insights, probability, branch_code
        )

        if probability >= 75:
            verdict = 'Highly Likely'
        elif probability >= 55:
            verdict = 'Likely'
        elif probability >= 40:
            verdict = 'Moderate Chance'
        else:
            verdict = 'Unlikely'

        return jsonify({
            'success': True,
            'probability': round(probability, 1),
            'prediction': prediction,
            'verdict': verdict,
            'strengths': strengths,
            'improvements': improvements,
            'actions': actions,
            'companies': companies
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

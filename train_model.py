import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

np.random.seed(42)
N = 2000

branches = ['CSE', 'IT', 'ECE', 'ME', 'CE', 'EE', 'DS_AI', 'MBA', 'BCA_MCA']
branch_placement_bias = {
    'CSE': 0.82, 'IT': 0.78, 'DS_AI': 0.85, 'ECE': 0.68,
    'BCA_MCA': 0.72, 'MBA': 0.70, 'ME': 0.58, 'EE': 0.60, 'CE': 0.45
}

comm_map = {'poor': 0, 'average': 1, 'good': 2, 'excellent': 3}

def generate_student():
    branch = np.random.choice(branches)
    bias = branch_placement_bias[branch]

    cgpa = np.clip(np.random.normal(7.2, 1.1), 4.0, 10.0)
    hsc = np.clip(np.random.normal(76, 12), 40, 100)
    ssc = np.clip(np.random.normal(80, 10), 45, 100)
    backlogs = np.random.choice([0, 0, 0, 1, 1, 2, 3], p=[0.45, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02])
    internships = np.random.choice([0, 1, 2, 3], p=[0.35, 0.4, 0.18, 0.07])
    projects = np.random.choice([0, 1, 3, 5], p=[0.2, 0.35, 0.3, 0.15])
    aptitude = np.clip(np.random.normal(65, 18), 10, 100)
    comm = np.random.choice(list(comm_map.keys()), p=[0.1, 0.35, 0.4, 0.15])
    tech_skills = np.random.randint(0, 9)
    extra_skills = np.random.randint(0, 5)
    certifications = np.random.randint(0, 4)
    hackathons = np.random.choice([0, 1], p=[0.6, 0.4])

    score = (
        (cgpa / 10) * 25 +
        (hsc / 100) * 10 +
        (ssc / 100) * 5 +
        max(0, (1 - backlogs * 0.3)) * 10 +
        (internships / 3) * 15 +
        (projects / 5) * 10 +
        (aptitude / 100) * 12 +
        (comm_map[comm] / 3) * 10 +
        (tech_skills / 8) * 8 +
        (extra_skills / 4) * 3 +
        hackathons * 2
    )

    noise = np.random.normal(0, 8)
    final_score = score + noise + (bias - 0.7) * 15

    placed = 1 if final_score > 48 else 0

    return {
        'branch': branch,
        'cgpa': round(cgpa, 2),
        'hsc_percent': round(hsc, 1),
        'ssc_percent': round(ssc, 1),
        'backlogs': int(backlogs),
        'internships': int(internships),
        'projects': int(projects),
        'aptitude_score': round(aptitude, 1),
        'communication': comm,
        'tech_skills_count': int(tech_skills),
        'extra_activities': int(extra_skills),
        'certifications': int(certifications),
        'hackathons': int(hackathons),
        'placed': placed
    }

data = [generate_student() for _ in range(N)]
df = pd.DataFrame(data)

print(f"Dataset shape: {df.shape}")
print(f"Placement rate: {df['placed'].mean():.2%}")
print(f"\nBranch-wise placement:\n{df.groupby('branch')['placed'].mean().sort_values(ascending=False)}")

le_branch = LabelEncoder()
le_comm = LabelEncoder()

df['branch_enc'] = le_branch.fit_transform(df['branch'])
df['comm_enc'] = le_comm.fit_transform(df['communication'])

feature_cols = [
    'branch_enc', 'cgpa', 'hsc_percent', 'ssc_percent', 'backlogs',
    'internships', 'projects', 'aptitude_score', 'comm_enc',
    'tech_skills_count', 'extra_activities', 'certifications', 'hackathons'
]

X = df[feature_cols]
y = df['placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

importances = model.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
print(f"\nFeature Importances:\n{feat_imp}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('branch_encoder.pkl', 'wb') as f:
    pickle.dump(le_branch, f)
with open('comm_encoder.pkl', 'wb') as f:
    pickle.dump(le_comm, f)

print("\nModel and encoders saved successfully!")
print(f"Final model accuracy: {acc:.2%}")

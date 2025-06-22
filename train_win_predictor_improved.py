import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load the data
print("Loading data...")
deliveries_df = pd.read_csv('deliveries_2008-2024.csv')
matches_df = pd.read_csv('matches_2008-2024.csv')

# Clean column names
deliveries_df.columns = deliveries_df.columns.str.strip()
matches_df.columns = matches_df.columns.str.strip()

# Team name mapping
TEAM_NAME_MAP = {
    'Chennai Super Kings': 'Chennai Super Kings',
    'Delhi Capitals': 'Delhi Capitals',
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Kings XI Punjab',
    'Punjab Kings': 'Kings XI Punjab',
    'Kolkata Knight Riders': 'Kolkata Knight Riders',
    'Mumbai Indians': 'Mumbai Indians',
    'Rajasthan Royals': 'Rajasthan Royals',
    'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
    'Sunrisers Hyderabad': 'Sunrisers Hyderabad',
}

def clean_team_name(name):
    if pd.isna(name) or not isinstance(name, str):
        return None
    return TEAM_NAME_MAP.get(name.strip(), None)

# Clean team names
deliveries_df['batting_team'] = deliveries_df['batting_team'].apply(clean_team_name)
deliveries_df['bowling_team'] = deliveries_df['bowling_team'].apply(clean_team_name)
matches_df['team1'] = matches_df['team1'].apply(clean_team_name)
matches_df['team2'] = matches_df['team2'].apply(clean_team_name)
matches_df['winner'] = matches_df['winner'].apply(clean_team_name)

# Filter data
deliveries_df = deliveries_df[deliveries_df['batting_team'].notnull() & deliveries_df['bowling_team'].notnull()]
matches_df = matches_df[matches_df['team1'].notnull() & matches_df['team2'].notnull() & matches_df['winner'].notnull()]

# Create match winners mapping
match_winners = matches_df[['id', 'winner']].set_index('id')['winner'].to_dict()

# Team names for encoding
TEAMS = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

# Enhanced feature engineering
rows = []
for (match_id, batting_team, bowling_team), group in deliveries_df.groupby(['match_id', 'batting_team', 'bowling_team']):
    if match_id not in match_winners:
        continue
    
    winner = match_winners[match_id]
    target = 1 if winner == batting_team else 0
    
    group = group.sort_values(['over', 'ball'])
    
    # Create features for each over (starting from 3rd over for more data)
    for over in range(3, min(group['over'].max() + 1, 21)):
        over_data = group[group['over'] <= over]
        current_runs = over_data['total_runs'].sum()
        wickets_out = over_data['player_dismissed'].notnull().sum()
        
        # Enhanced features
        run_lst_5 = group[(group['over'] > over - 5) & (group['over'] <= over)]['total_runs'].sum()
        wicket_lst_5 = group[(group['over'] > over - 5) & (group['over'] <= over)]['player_dismissed'].notnull().sum()
        
        # New features
        run_lst_3 = group[(group['over'] > over - 3) & (group['over'] <= over)]['total_runs'].sum()
        wicket_lst_3 = group[(group['over'] > over - 3) & (group['over'] <= over)]['player_dismissed'].notnull().sum()
        
        # Calculate run rate and required run rate
        balls_left = (20 - over) * 6
        wickets_left = 10 - wickets_out
        current_run_rate = current_runs / over if over > 0 else 0
        
        # Momentum features
        recent_run_rate = run_lst_5 / 5 if over >= 5 else current_run_rate
        recent_wicket_rate = wicket_lst_5 / 5 if over >= 5 else 0
        
        # Pressure features
        overs_remaining = 20 - over
        balls_per_wicket = balls_left / wickets_left if wickets_left > 0 else balls_left
        
        # Boundary analysis (if available)
        try:
            boundaries = over_data[over_data['batsman_runs'].isin([4, 6])].shape[0]
            boundary_rate = boundaries / over if over > 0 else 0
        except:
            boundary_rate = 0
        
        # One-hot encoding for teams
        encoded_batting = [1 if batting_team == t else 0 for t in TEAMS]
        encoded_bowling = [1 if bowling_team == t else 0 for t in TEAMS]
        
        # Create feature vector
        features = [
            current_runs, wickets_out, over, run_lst_5, wicket_lst_5,
            run_lst_3, wicket_lst_3, balls_left, wickets_left, current_run_rate,
            recent_run_rate, recent_wicket_rate, overs_remaining, balls_per_wicket,
            boundary_rate
        ]
        features.extend(encoded_batting)
        features.extend(encoded_bowling)
        features.append(target)
        
        rows.append(features)

# Create DataFrame
columns = [
    'current_runs', 'wickets_out', 'over', 'run_lst_5', 'wicket_lst_5',
    'run_lst_3', 'wicket_lst_3', 'balls_left', 'wickets_left', 'current_run_rate',
    'recent_run_rate', 'recent_wicket_rate', 'overs_remaining', 'balls_per_wicket',
    'boundary_rate'
] + [f'bat_{t}' for t in TEAMS] + [f'ball_{t}' for t in TEAMS] + ['target']

data = pd.DataFrame(rows, columns=columns)

print(f"Enhanced training data shape: {data.shape}")
print(f"Win rate in training data: {data['target'].mean():.3f}")

# Features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== Training Multiple Models ===")

# 1. Random Forest with hyperparameter tuning
print("Training Random Forest...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
rf_cv = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_cv.fit(X_train_scaled, y_train)
best_rf = rf_cv.best_estimator_

# 2. Gradient Boosting
print("Training Gradient Boosting...")
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

gb = GradientBoostingClassifier(random_state=42)
gb_cv = GridSearchCV(gb, gb_params, cv=5, scoring='accuracy', n_jobs=-1)
gb_cv.fit(X_train_scaled, y_train)
best_gb = gb_cv.best_estimator_

# 3. Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

# 4. Ensemble Model
print("Creating Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('gb', best_gb),
        ('lr', lr)
    ],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)

# Evaluate all models
models = {
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb,
    'Logistic Regression': lr,
    'Ensemble': ensemble
}

print("\n=== Model Performance Comparison ===")
best_model = None
best_score = 0

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    
    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print()
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = model

print(f"Best model: {best_score:.4f} accuracy")

# Save the best model and scaler
with open('win_predictor_improved.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('win_predictor_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nImproved win prediction model saved as win_predictor_improved.pkl")
print(f"Scaler saved as win_predictor_scaler.pkl")

# Feature importance (for Random Forest)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 10 Most Important Features ===")
    print(feature_importance.head(10)) 
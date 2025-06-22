import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the data
print("Loading data...")
deliveries_df = pd.read_csv('deliveries_2008-2024.csv')
matches_df = pd.read_csv('matches_2008-2024.csv')

# Clean column names
deliveries_df.columns = deliveries_df.columns.str.strip()
matches_df.columns = matches_df.columns.str.strip()

# Team name mapping (same as in score prediction)
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

# Clean team names in both dataframes
deliveries_df['batting_team'] = deliveries_df['batting_team'].apply(clean_team_name)
deliveries_df['bowling_team'] = deliveries_df['bowling_team'].apply(clean_team_name)
matches_df['team1'] = matches_df['team1'].apply(clean_team_name)
matches_df['team2'] = matches_df['team2'].apply(clean_team_name)
matches_df['winner'] = matches_df['winner'].apply(clean_team_name)

# Only keep matches with both teams in the model
deliveries_df = deliveries_df[deliveries_df['batting_team'].notnull() & deliveries_df['bowling_team'].notnull()]
matches_df = matches_df[matches_df['team1'].notnull() & matches_df['team2'].notnull() & matches_df['winner'].notnull()]

# Create a mapping from match_id to winner
match_winners = matches_df[['id', 'winner']].set_index('id')['winner'].to_dict()

# Team names for one-hot encoding
TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

# Create training data for win prediction
rows = []
for (match_id, batting_team, bowling_team), group in deliveries_df.groupby(['match_id', 'batting_team', 'bowling_team']):
    if match_id not in match_winners:
        continue
    
    winner = match_winners[match_id]
    # Target: 1 if batting team won, 0 if bowling team won
    target = 1 if winner == batting_team else 0
    
    group = group.sort_values(['over', 'ball'])
    
    # Create features for each over (starting from 5th over)
    for over in range(5, min(group['over'].max() + 1, 21)):  # Max 20 overs
        over_data = group[group['over'] <= over]
        current_runs = over_data['total_runs'].sum()
        wickets_out = over_data['player_dismissed'].notnull().sum()
        
        # Runs and wickets in last 5 overs
        run_lst_5 = group[(group['over'] > over - 5) & (group['over'] <= over)]['total_runs'].sum()
        wicket_lst_5 = group[(group['over'] > over - 5) & (group['over'] <= over)]['player_dismissed'].notnull().sum()
        
        # One-hot encoding for teams
        encoded_batting = [1 if batting_team == t else 0 for t in TEAMS]
        encoded_bowling = [1 if bowling_team == t else 0 for t in TEAMS]
        
        # Additional features
        balls_left = (20 - over) * 6
        wickets_left = 10 - wickets_out
        run_rate = current_runs / over if over > 0 else 0
        
        row = [current_runs, wickets_out, over, run_lst_5, wicket_lst_5, 
               balls_left, wickets_left, run_rate]
        row.extend(encoded_batting)
        row.extend(encoded_bowling)
        row.append(target)
        rows.append(row)

# Create DataFrame
columns = [
    'current_runs', 'wickets_out', 'over', 'run_lst_5', 'wicket_lst_5',
    'balls_left', 'wickets_left', 'run_rate'
] + [f'bat_{t}' for t in TEAMS] + [f'ball_{t}' for t in TEAMS] + ['target']

data = pd.DataFrame(rows, columns=columns)

print(f"Training data shape: {data.shape}")
print(f"Win rate in training data: {data['target'].mean():.3f}")

# Features and target
X = data.drop('target', axis=1)
y = data['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
print("Training win prediction model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
with open('win_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Win prediction model saved as win_predictor.pkl") 
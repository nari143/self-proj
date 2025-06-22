import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# Load the deliveries data
# Make sure deliveries_2008-2024.csv is in the same directory as this script
print("Loading data...")
df = pd.read_csv('deliveries_2008-2024.csv')
df.columns = df.columns.str.strip()
print(df.columns)

print("All batting teams in data:", df['batting_team'].unique())
print("All bowling teams in data:", df['bowling_team'].unique())

# Team name mapping (add all variants you see in your data)
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
    # The following teams are not in your model, so we can ignore them
    # 'Deccan Chargers': None,
    # 'Kochi Tuskers Kerala': None,
    # 'Pune Warriors': None,
    # 'Rising Pune Supergiants': None,
    # 'Rising Pune Supergiant': None,
    # 'Gujarat Lions': None,
    # 'Lucknow Super Giants': None,
    # 'Gujarat Titans': None,
}

def clean_team_name(name):
    return TEAM_NAME_MAP.get(name.strip(), None)

df['batting_team'] = df['batting_team'].apply(clean_team_name)
df['bowling_team'] = df['bowling_team'].apply(clean_team_name)

# Only keep rows where both teams are in the model
df = df[df['batting_team'].notnull() & df['bowling_team'].notnull()]

# Team names as per your Streamlit app
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

# Only keep matches with both teams in TEAMS
mask = df['batting_team'].isin(TEAMS) & df['bowling_team'].isin(TEAMS)
df = df[mask]

# Only use first innings
first_innings = df[df['inning'] == 1]

print("Unique innings:", df['inning'].unique())
print("Unique batting teams:", df['batting_team'].unique())
print("Unique bowling teams:", df['bowling_team'].unique())

# Group by match and over to create features
rows = []
for (match_id, batting_team, bowling_team), group in first_innings.groupby(['match_id', 'batting_team', 'bowling_team']):
    group = group.sort_values(['over', 'ball'])
    # Get final score for this first innings
    final_score = group['total_runs'].sum()
    for over in range(5, group['over'].max() + 1):  # Start from 5th over
        over_data = group[group['over'] <= over]
        current_runs = over_data['total_runs'].sum()
        wickets_out = over_data['player_dismissed'].notnull().sum()
        run_lst_5 = group[(group['over'] > over - 5) & (group['over'] <= over)]['total_runs'].sum()
        wicket_lst_5 = group[(group['over'] > over - 5) & (group['over'] <= over)]['player_dismissed'].notnull().sum()
        # One-hot encoding for teams
        encoded_batting = [1 if batting_team == t else 0 for t in TEAMS]
        encoded_bowling = [1 if bowling_team == t else 0 for t in TEAMS]
        row = [current_runs, wickets_out, over, run_lst_5, wicket_lst_5]
        row.extend(encoded_batting)
        row.extend(encoded_bowling)
        row.append(final_score)
        rows.append(row)

# Create DataFrame
columns = [
    'current_runs', 'wickets_out', 'over', 'run_lst_5', 'wicket_lst_5'
] + [f'bat_{t}' for t in TEAMS] + [f'ball_{t}' for t in TEAMS] + ['final_score']
data = pd.DataFrame(rows, columns=columns)

# Features and target
X = data.drop('final_score', axis=1)
y = data['final_score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Validation MAE: {mae:.2f}")

# Save model
with open('predict_ipl_1st_innings_score_etr.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as predict_ipl_1st_innings_score_etr.pkl") 
# -*- coding: utf-8 -*-
"""Group34_Sports_model_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15m0DJhJ_vjB1tQZhD1PlSz_zoYpiusm2
"""



import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import pickle


from google.colab import drive
drive.mount('/content/drive')

#loading the dataset for preproccesing and feature engineering
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/players_21.csv')
df.info()

all_features=[
    "sofifa_id", "player_url", "short_name", "long_name", "player_positions",
    "overall", "potential", "value_eur", "wage_eur", "age", "dob", "height_cm",
    "weight_kg", "club_team_id", "club_name", "league_name", "league_level",
    "club_position", "club_jersey_number", "club_loaned_from", "club_joined",
    "club_contract_valid_until", "nationality_id", "nationality_name",
    "nation_team_id", "nation_position", "nation_jersey_number", "preferred_foot",
    "weak_foot", "skill_moves", "international_reputation", "work_rate", "body_type",
    "real_face", "release_clause_eur", "player_tags", "player_traits", "pace", "shooting",
    "passing", "dribbling", "defending", "physic", "attacking_crossing", "attacking_finishing",
    "attacking_heading_accuracy", "attacking_short_passing", "attacking_volleys",
    "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing",
    "skill_ball_control", "movement_acceleration", "movement_sprint_speed",
    "movement_agility", "movement_reactions", "movement_balance", "power_shot_power",
    "power_jumping", "power_stamina", "power_strength", "power_long_shots",
    "mentality_aggression", "mentality_interceptions", "mentality_positioning",
    "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes", "goalkeeping_speed", "ls", "st", "rs",
    "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram", "lm", "lcm", "cm", "rcm", "rm", "lwb",
    "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb", "gk", "player_face_url",
    "club_logo_url", "club_flag_url", "nation_logo_url", "nation_flag_url"
]

correlation_matrix = df[all_features].corr()
correlation_matrix['overall'].sort_values(ascending=False)

selected_features=[
    "overall",
    "movement_reactions",
    "passing",
    "mentality_composure",
    "dribbling",
    "potential",
    "release_clause_eur",
    "wage_eur",
    "value_eur",
    "power_shot_power",
    "physic",
    "mentality_vision",
    "attacking_short_passing",
    "goalkeeping_speed",
    "shooting",
    "skill_long_passing",
    "age",
    "skill_ball_control",
    "international_reputation",
    "skill_curve",
    "attacking_crossing",
    "power_long_shots",
    "mentality_aggression",
    "skill_fk_accuracy",
    "power_stamina",
    "skill_moves",
    "skill_dribbling",
    "attacking_volleys",
    "defending",
    "power_strength",
    "mentality_positioning",
    "mentality_penalties",
    "attacking_heading_accuracy",
    "attacking_finishing",
    "defending_marking_awareness",
    "mentality_interceptions",
    "power_jumping",
    "movement_agility",
    "defending_standing_tackle",
    "defending_sliding_tackle",
    "weak_foot",
    "movement_sprint_speed",
    "movement_acceleration",
    "pace",
    "weight_kg",
    "movement_balance",
    "club_contract_valid_until",
    "height_cm",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
    "goalkeeping_handling",
    "goalkeeping_diving",
    "goalkeeping_kicking"
]

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/players_21.csv', usecols=selected_features)
df.info()

df.shape

df.head()

df.fillna(0, inplace=True)
df['overall'] = df['overall'].apply(lambda x: 92 if x == 93 else x)

shooting_features = [
    "shooting",
    "power_shot_power",
    "power_long_shots",
    "attacking_volleys",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_crossing"
]

df['shooting_skills'] = df[shooting_features].mean(axis=1)

df.drop(columns=shooting_features, inplace=True)

df.info()

mentality_attributes = [
    "mentality_composure",
    "mentality_vision",
    "mentality_aggression",
    "mentality_positioning",
    "mentality_penalties",
    "mentality_interceptions",
]

# Calculate the mean of 'mentality_attributes' for each row
df['mentality'] = df[mentality_attributes].mean(axis=1)

# Fill missing values in 'mentality' with the calculated mean
df['mentality'].fillna(df['mentality'].mean(), inplace=True)

# Drop the original 'mentality_attributes' columns
df.drop(columns=mentality_attributes, inplace=True)

df.info()



dribbling_attributes = [
    "dribbling",
    "skill_dribbling",
    "skill_ball_control",
    "movement_balance",
]



# Calculate the mean of 'dribbling_attributes' for each row
df['dribbling'] = df[dribbling_attributes].mean(axis=1)

# Fill missing values in 'dribbling' with the calculated mean
df['dribbling'].fillna(df['dribbling'].mean(), inplace=True)

# Drop the original 'dribbling_attributes' columns
df.drop(columns=dribbling_attributes, inplace=True)

df.info()

df.info()

df.isnull().sum()

y=df['overall']
x=df.drop('overall',axis=1)

sc=StandardScaler()
scaled=sc.fit_transform(x)

x=pd.DataFrame(scaled, columns=x.columns)

x.head()
x.info()

y.value_counts()

Xtrain,Xtest,Ytrain,Ytest=train_test_split(x,y,test_size=0.1,random_state=42,stratify = y)
Xtrain.shape

rf=RandomForestRegressor()

rf.fit(Xtrain, Ytrain)

y_pred=rf.predict(Xtest)
y_pred

mean_absolute_error(y_pred,Ytest)

xgb_model = XGBRegressor()

xgb_model.fit(Xtrain, Ytrain)

y_pred_xgb = xgb_model.predict(Xtest)

mae_xgb = mean_absolute_error(y_pred_xgb, Ytest)
mae_xgb

from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor()

gb_model.fit(Xtrain, Ytrain)

y_pred_gb = gb_model.predict(Xtest)

mae_gb = mean_absolute_error(y_pred_gb, Ytest)

mae_gb

from sklearn.ensemble import VotingRegressor

ensemble_model = VotingRegressor(estimators=[('rf', rf), ('xgb', xgb_model)])

# Training the ensemble model
ensemble_model.fit(Xtrain, Ytrain)

# Making predictions using the ensemble model
ensemble_predictions = ensemble_model.predict(Xtest)


mae_ensemble_new = mean_absolute_error(ensemble_predictions, Ytest)

mae_ensemble_new

#Calculating mean absolute error for the ensemble predictions
mae_ensemble = mean_absolute_error(ensemble_predictions, Ytest)

print("Mean Absolute Error for Random Forest: ", mean_absolute_error(y_pred, Ytest))
print("Mean Absolute Error for XGBoost: ", mae_xgb)
print("Mean Absolute Error for Ensemble Model: ", mae_ensemble)

new_set=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/players_22.csv', usecols=selected_features)

new_set.fillna(0, inplace=True)
new_set['overall'] = new_set['overall'].apply(lambda x: 92 if x in [92, 93] else x)

shooting_features = [
    "shooting",
    "power_shot_power",
    "power_long_shots",
    "attacking_volleys",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_crossing"
]

new_set['shooting_skills'] = new_set[shooting_features].mean(axis=1)

new_set.drop(columns=shooting_features, inplace=True)

new_set.info()

goalkeeping = [
    "goalkeeping_speed",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
    "goalkeeping_handling",
    "goalkeeping_diving",
    "goalkeeping_kicking",
]
# Calculate the mean of 'goalkeeping_attributes' for each row and assign it to a new column
new_set['goalkeeping_ability'] = new_set[goalkeeping].mean(axis=1)

# Drop the original 'goalkeeping_attributes' columns
new_set.drop(columns=goalkeeping, inplace=True)


new_set.info()

mentality_attributes = [
    "mentality_composure",
    "mentality_vision",
    "mentality_aggression",
    "mentality_positioning",
    "mentality_penalties",
    "mentality_interceptions",
]

# Calculate the mean of 'mentality_attributes' for each row
new_set['mentality'] = new_set[mentality_attributes].mean(axis=1)

# Fill missing values in 'mentality' with the calculated mean
new_set['mentality'].fillna(df['mentality'].mean(), inplace=True)

# Drop the original 'mentality_attributes' columns
new_set.drop(columns=mentality_attributes, inplace=True)

new_set.info()

dribbling_attributes = [
    "dribbling",
    "skill_dribbling",
    "skill_ball_control",
    "movement_balance",
]



# Calculate the mean of 'dribbling_attributes' for each row
new_set['dribbling'] = new_set[dribbling_attributes].mean(axis=1)

# Fill missing values in 'dribbling' with the calculated mean
new_set['dribbling'].fillna(new_set['dribbling'].mean(), inplace=True)

# Drop the original 'dribbling_attributes' columns
new_set.drop(columns=dribbling_attributes, inplace=True)

new_set.info()

y_1 = new_set['overall']
x_1 = new_set.drop('overall', axis=1)
x_1

scaled_x_1 = sc.fit_transform(x_1)
x_1 = pd.DataFrame(scaled_x_1, columns=x_1.columns)

x_1_train,x_1_test,y_1_train,y_1_test=train_test_split(x_1,y_1,test_size=0.1,random_state=42,stratify = y_1)

rf.fit(x_1_train, y_1_train)
y_pred_rf = rf.predict(x_1_test)

mean_absolute_error(y_pred_rf,y_1_test)

xgb_model.fit(x_1_train, y_1_train)
y_pred_xgb = xgb_model.predict(x_1_test)
mae_xgb_new = mean_absolute_error(y_pred_xgb, y_1_test)
mae_xgb_new

from sklearn.ensemble import VotingRegressor

ensemble_model = VotingRegressor(estimators=[('rf', rf), ('xgb', xgb_model)])

ensemble_model.fit(x_1_train, y_1_train)

ensemble_predictions = ensemble_model.predict(x_1_test)

mae_ensemble_new = mean_absolute_error(ensemble_predictions, y_1_test)

mae_ensemble_new

joblib.dump(sc, '/content/drive/My Drive/Colab Notebooks/scaler.pkl')

filename = '/content/drive/My Drive/Colab Notebooks/sports_prediction_model.pkl'
pickle.dump(rf, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))




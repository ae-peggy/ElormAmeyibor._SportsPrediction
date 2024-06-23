import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the XGBoost model
model = joblib.load('best_xgboost_model.pkl')

st.set_page_config(page_title="Player Rating Prediction App", layout="centered", initial_sidebar_state="expanded")

st.title('⚽ Player Rating Prediction App')
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
        color: #333;
        padding: 1rem;
    }
    .sidebar .sidebar-content {
        background: #dde1e7;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #FB6327;
        text-align: center;
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar input form
st.sidebar.header('⚙️ Player Features')

def user_input_features():
    movement_reactions = st.sidebar.slider('⚡ Movement Reactions', 0, 100, 50)
    potential = st.sidebar.slider('🌟 Potential', 0, 100, 50)
    passing = st.sidebar.slider('⚽ Passing', 0, 100, 50)
    wage_eur = st.sidebar.number_input('💷 Wage (EUR)', min_value=0)
    mentality_composure = st.sidebar.slider('🧠 Composure', 0, 100, 50)
    value_eur = st.sidebar.number_input('💷 Value (EUR)', min_value=0)
    dribbling = st.sidebar.slider('👟 Dribbling', 0, 100, 50)
    attacking_short_passing = st.sidebar.slider('⚽️ Short Passing', 0, 100, 50)
    mentality_vision = st.sidebar.slider('👁️ Vision', 0, 100, 50)
    international_reputation = st.sidebar.number_input('🌍 International Reputation', min_value=0, max_value=5)
    skill_long_passing = st.sidebar.slider('⚽️ Long Passing', 0, 100, 50)
    power_shot_power = st.sidebar.slider('💥 Shot Power', 0, 100, 50)
    physic = st.sidebar.slider('💪 Physic', 0, 100, 50)
    release_clause_eur = st.sidebar.number_input('💼 Release Clause (EUR)', min_value=0)
    age = st.sidebar.slider('🎂 Age', 16, 50, 25)
    skill_ball_control = st.sidebar.slider('⚽ Ball Control', 0, 100, 50)
    shooting = st.sidebar.slider('🎯 Shooting', 0, 100, 50)
    skill_curve = st.sidebar.slider('🔄 Curve', 0, 100, 50)
    power_long_shots = st.sidebar.slider('💥 Long Shots', 0, 100, 50)

    data = {
        'movement_reactions': movement_reactions,
        'potential': potential,
        'passing': passing,
        'wage_eur': wage_eur,
        'mentality_composure': mentality_composure,
        'value_eur': value_eur,
        'dribbling': dribbling,
        'attacking_short_passing': attacking_short_passing,
        'mentality_vision': mentality_vision,
        'international_reputation': international_reputation,
        'skill_long_passing': skill_long_passing,
        'power_shot_power': power_shot_power,
        'physic': physic,
        'release_clause_eur': release_clause_eur,
        'age': age,
        'skill_ball_control': skill_ball_control,
        'shooting': shooting,
        'skill_curve': skill_curve,
        'power_long_shots': power_long_shots
    }

    features = pd.DataFrame(data, index=[0])
    return features

# User input features
input_df = user_input_features()
st.subheader('Player Features Summary')
st.write(input_df)

# Predict the rating
prediction = model.predict(input_df)[0]

# Calculate residuals and confidence
def calculate_confidence(model, input_df):
    # Make predictions for all data to get residuals
    y_pred_all = model.predict(input_df)
    residuals = model.predict(input_df, output_margin=True) - y_pred_all

    # Standard deviation of residuals
    residual_std = np.std(residuals)

    # Calculate 99% prediction interval
    t_value = 2.576  # for 99% confidence level 
    lower_bound = prediction - t_value * residual_std
    upper_bound = prediction + t_value * residual_std

    return lower_bound, upper_bound

lower_confidence, upper_confidence = calculate_confidence(model, input_df)

# Display prediction and confidence
st.subheader('Prediction')
st.write(f"Predicted Rating: {prediction:.2f}")
st.write(f"99% Confidence Interval: [{lower_confidence:.2f}, {upper_confidence:.2f}]")

# Footer
st.markdown('<div class="footer">Made by Elorm Peggy Ameyibor</div>', unsafe_allow_html=True)

import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import json

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Restaurant Rating Predictor",
    page_icon="🍴",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    h1, h2, h3 {
        color: #00FFAA;
        text-align: center;
    }

    .stButton>button {
        width: 100%;
        background-color: #00FFAA;
        color: black;
        font-size: 18px;
        border-radius: 10px;
    }
            
    /* Mobile Optimization */

@media (max-width: 768px) {

    h1 {
        font-size: 35px !important;
    }

    h2 {
        font-size: 28px !important;
    }

    h3 {
        font-size: 22px !important;
    }

    .stMetric {
        text-align: center;
    }

    .stDataFrame {
        overflow-x: auto;
    }

}

    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

model = pickle.load(
    open("models/restaurant_model.pkl", "rb")
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    "Dataset .csv",
    encoding='utf-8',
    encoding_errors='ignore'
)

# =========================
# USER AUTHENTICATION
# =========================

users_file = "users.json"

# Load users
try:
    with open(users_file, "r") as file:
        users = json.load(file)
except:
    users = {}

st.sidebar.header("🔐 Login / Signup")

menu = st.sidebar.radio(
    "Select Option",
    ["Login", "Signup"]
)

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input(
    "Password",
    type="password"
)

logged_in = False

# Signup
if menu == "Signup":

    if st.sidebar.button("Create Account"):

        if username in users:
            st.sidebar.error("User already exists")

        else:
            users[username] = password

            with open(users_file, "w") as file:
                json.dump(users, file)

            st.sidebar.success(
                "Account Created Successfully"
            )

# Login
if menu == "Login":

    if st.sidebar.button("Login"):

        if username in users and users[username] == password:
            logged_in = True
            st.session_state.logged_in = True
            st.sidebar.success("Login Successful")

        else:
            st.sidebar.error("Invalid Credentials")
if not logged_in and not st.session_state.get("logged_in", False):
    st.warning("Please login to access the app")
    st.stop()

# Session check
if "logged_in" in st.session_state:
    logged_in = st.session_state.logged_in

# =========================
# TITLE
# =========================
if logged_in:

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False

        st.rerun()

    st.title("🍴 AI Restaurant Rating Predictor")    

# =========================
# KPI CARDS
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Restaurants",
        len(df)
    )

with col2:
    st.metric(
        "Average Rating",
        round(df['Aggregate rating'].mean(), 2)
    )

with col3:
    st.metric(
        "Average Cost",
        round(df['Average Cost for two'].mean(), 2)
    )

st.write("""
Predict restaurant ratings using Machine Learning and explore restaurant analytics.
""")

# =========================
# SIDEBAR
# =========================

st.sidebar.header("User Input")

price_range = st.sidebar.slider(
    "Price Range",
    1,
    4,
    2
)

votes = st.sidebar.number_input(
    "Votes",
    min_value=0,
    value=100
)

average_cost = st.sidebar.number_input(
    "Average Cost for Two",
    min_value=0,
    value=500
)

table_booking = st.sidebar.selectbox(
    "Table Booking",
    ["No", "Yes"]
)

online_delivery = st.sidebar.selectbox(
    "Online Delivery",
    ["No", "Yes"]
)

# =========================
# CONVERT INPUTS
# =========================

table_booking_num = 1 if table_booking == "Yes" else 0

online_delivery_num = 1 if online_delivery == "Yes" else 0

# =========================
# FILTERED DATAFRAME
# =========================

filtered_df = df[
    (df['Price range'] == price_range)
]

if online_delivery_num == 1:
    filtered_df = filtered_df[
        filtered_df['Has Online delivery'] == 'Yes'
    ]

if table_booking_num == 1:
    filtered_df = filtered_df[
        filtered_df['Has Table booking'] == 'Yes'
    ]

# =========================
# AI PREDICTION
# =========================

st.header("🤖 AI Prediction")

input_data = pd.DataFrame([[
    price_range,
    votes,
    average_cost,
    table_booking_num,
    online_delivery_num
]], columns=[
    'Price range',
    'Votes',
    'Average Cost for two',
    'Has Table Booking',
    'Has Online Delivery'
])

prediction = model.predict(input_data)

st.markdown(f"""
<div style="
padding:20px;
border-radius:15px;
background:linear-gradient(to right,#00FFAA,#00BFFF);
text-align:center;
font-size:30px;
font-weight:bold;
color:black;
">
⭐ Predicted Rating: {prediction[0]:.2f}
</div>
""", unsafe_allow_html=True)

# =========================
# SMART RECOMMENDATIONS
# =========================

st.header("🍽️ Smart Restaurant Recommendations")

recommended_df = filtered_df.sort_values(
    by='Aggregate rating',
    ascending=False
)

st.dataframe(
    recommended_df[
        [
            'Restaurant Name',
            'City',
            'Cuisines',
            'Aggregate rating',
            'Average Cost for two'
        ]
    ].head(10)
)

# =========================
# GOOGLE MAPS
# =========================

st.header("🗺️ Restaurant Location Map")

# Example coordinates
restaurant_map = folium.Map(
    location=[28.6139, 77.2090],
    zoom_start=5
)

folium.Marker(
    [28.6139, 77.2090],
    popup="Top Restaurant Area"
).add_to(restaurant_map)

st_folium(
    restaurant_map,
    width=700,
    height=500
)

# =========================
# PIE CHARTS
# =========================

st.header("🥧 Delivery & Booking Analysis")

col4, col5 = st.columns(2)

with col4:

    delivery_counts = filtered_df[
        'Has Online delivery'
    ].value_counts()

    fig4, ax4 = plt.subplots()

    ax4.pie(
        delivery_counts,
        labels=delivery_counts.index,
        autopct='%1.1f%%'
    )

    ax4.set_title("Online Delivery")

    st.pyplot(fig4)

with col5:

    booking_counts = filtered_df[
        'Has Table booking'
    ].value_counts()

    fig5, ax5 = plt.subplots()

    ax5.pie(
        booking_counts,
        labels=booking_counts.index,
        autopct='%1.1f%%'
    )

    ax5.set_title("Table Booking")

    st.pyplot(fig5)

# =========================
# FEATURE IMPORTANCE
# =========================

st.header("🧠 Feature Importance")

importance_df = pd.DataFrame({
    'Feature': [
        'Price Range',
        'Votes',
        'Average Cost',
        'Table Booking',
        'Online Delivery'
    ],
    'Importance': model.feature_importances_
})

fig3, ax3 = plt.subplots(figsize=(8,5))

importance_df.sort_values(
    by='Importance'
).plot(
    x='Feature',
    y='Importance',
    kind='barh',
    ax=ax3
)

st.pyplot(fig3)

# =========================
# RATING DISTRIBUTION
# =========================

st.header("📊 Distribution of Ratings")

fig, ax = plt.subplots(figsize=(8,5))

ax.hist(
    filtered_df['Aggregate rating'],
    bins=10
)

ax.set_title("Distribution of Ratings")

ax.set_xlabel("Ratings")
ax.set_ylabel("Count")

st.pyplot(fig)

# =========================
# CITY ANALYSIS
# =========================

st.header("🌍 City-wise Analysis")

filtered_city = filtered_df[
    'City'
].value_counts().head(10)

fig6, ax6 = plt.subplots(figsize=(10,5))

filtered_city.plot(
    kind='bar',
    ax=ax6
)

ax6.set_title(
    "Top Cities Based on Current Filters"
)

st.pyplot(fig6)

# =========================
# CUISINE ANALYSIS
# =========================

st.header("🍕 Top Cuisines")

top_cuisines = filtered_df[
    'Cuisines'
].value_counts().head(10)

fig7, ax7 = plt.subplots(figsize=(10,5))

top_cuisines.plot(
    kind='bar',
    ax=ax7
)

ax7.set_title(
    "Top Cuisines Based on Current Filters"
)

st.pyplot(fig7)

# =========================
# CITY FILTER
# =========================

st.header("🌍 Explore Restaurants by City")

selected_city = st.selectbox(
    "Select City",
    sorted(df['City'].unique())
)

city_df = df[
    df['City'] == selected_city
]

st.dataframe(
    city_df[
        [
            'Restaurant Name',
            'Cuisines',
            'Aggregate rating'
        ]
    ].head(20)
)

# =========================
# FOOTER
# =========================

st.markdown("""
---
### 🚀 Built with Streamlit, Python & Machine Learning
""")
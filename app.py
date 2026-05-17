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

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #050816;
    color: white;
}

.main {
    background: linear-gradient(
        135deg,
        #050816,
        #0b1026
    );
}

h1 {
    text-align: center;
    font-size: 60px !important;
    font-weight: bold;
    color: #00F5FF;
    text-shadow: 0px 0px 25px #00F5FF;
}

h2, h3 {
    color: #00FFAA;
    text-align: center;
}

.stButton>button {

    width: 100%;
    border-radius: 15px;
    border: none;

    background: linear-gradient(
        90deg,
        #00F5FF,
        #00FFAA
    );

    color: black;
    font-size: 18px;
    font-weight: bold;
    padding: 12px;

    transition: 0.3s;
}

.stButton>button:hover {

    transform: scale(1.03);

    box-shadow:
        0px 0px 20px #00F5FF;
}

[data-testid="metric-container"] {

    background: rgba(
        255,
        255,
        255,
        0.05
    );

    border-radius: 20px;

    padding: 20px;

    box-shadow:
        0px 0px 15px rgba(
            0,
            255,
            255,
            0.3
        );
}

.stDataFrame {

    border-radius: 20px;
    overflow: hidden;
}

section[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        #111827,
        #050816
    );
}

@media (max-width:768px){

    h1{
        font-size:40px !important;
    }

    h2{
        font-size:28px !important;
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

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# SIDEBAR AUTH
st.sidebar.header("🔐 Login / Signup")

# SHOW LOGIN/SIGNUP ONLY IF NOT LOGGED IN
if not st.session_state.logged_in:

    menu = st.sidebar.radio(
        "Select Option",
        ["Login", "Signup"]
    )

    username = st.sidebar.text_input(
        "Username"
    )

    password = st.sidebar.text_input(
        "Password",
        type="password"
    )

    # SIGNUP
    if menu == "Signup":

        if st.sidebar.button(
            "Create Account",
            key="signup_btn"
        ):

            if username in users:

                st.sidebar.error(
                    "User already exists"
                )

            else:

                users[username] = password

                with open(users_file, "w") as file:
                    json.dump(users, file)

                st.sidebar.success(
                    "Account Created Successfully"
                )

    # LOGIN
    if menu == "Login":

        if st.sidebar.button(
            "Login",
            key="login_btn"
        ):

            if (
                username in users
                and users[username] == password
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.rerun()

            else:

                st.sidebar.error(
                    "Invalid Credentials"
                )

# IF LOGGED IN
if st.session_state.logged_in:

    st.sidebar.success(
        f"Welcome {st.session_state.username} 👋"
    )

    if st.sidebar.button(
        "Logout",
        key="logout_btn"
    ):

        st.session_state.logged_in = False
        st.session_state.username = ""

        st.rerun()

# STOP APP IF NOT LOGGED IN
if not st.session_state.logged_in:

    st.markdown(
        '''
        <h1 style="
        text-align:center;
        color:#00FFAA;
        margin-top:100px;
        font-size:60px;
        ">
        🍴 AI Restaurant Rating Predictor
        </h1>
        ''',
        unsafe_allow_html=True
    )

    st.warning(
        "Please login to access the dashboard"
    )

    st.stop()  

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

price_options = {
    "Budget 🍔": 1,
    "Moderate 🍕": 2,
    "Expensive 🍽️": 3,
    "Luxury 🌟": 4
}

selected_price = st.sidebar.selectbox(
    "💰 Select Price Category",
    list(price_options.keys())
)

price_range = price_options[selected_price]

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
# DYNAMIC GOOGLE MAPS
# =========================

st.header("🗺️ Restaurant Location Map")

# City coordinates
city_coordinates = {

    "New Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Bangalore": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    "Pune": [18.5204, 73.8567],
    "Chennai": [13.0827, 80.2707],
    "Kolkata": [22.5726, 88.3639],
    "Jaipur": [26.9124, 75.7873],
    "London": [51.5074, -0.1278],
    "Dubai": [25.2048, 55.2708],
    "Istanbul": [41.0082, 28.9784],
    "Abu Dhabi": [24.4539, 54.3773],
    "Doha": [25.2854, 51.5310],
    "Sharjah": [25.3463, 55.4209]
}

# Use selected city from explore section

if selected_city in city_coordinates:

    lat, lon = city_coordinates[selected_city]

else:

    lat, lon = [28.6139, 77.2090]

# Create map
restaurant_map = folium.Map(
    location=[lat, lon],
    zoom_start=11
)

# Marker
folium.Marker(
    [lat, lon],

    popup=f"📍 {selected_city}",

    tooltip=selected_city,

    icon=folium.Icon(
        color="green",
        icon="cutlery",
        prefix="fa"
    )

).add_to(restaurant_map)

# Show map
st_folium(
    restaurant_map,
    width=900,
    height=500
)

# =========================
# FOOTER
# =========================

st.markdown("""
---
### 🚀 Built with ❤️ by Abhinav | AI Restaurant Rating Predictor
""")
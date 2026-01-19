import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import requests
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="OCC STL Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with light background and black text
st.markdown("""
<style>
    /* Global background - light and airy gradient */
    .stApp {
        background:
            radial-gradient(circle at 20% 0%, #ffffff 0%, rgba(255, 255, 255, 0) 35%),
            radial-gradient(circle at 80% 0%, #ffe0b2 0%, rgba(255, 224, 178, 0) 45%),
            linear-gradient(180deg, #e0f2ff 0%, #b3e5fc 40%, #e3f2fd 100%);
        color: #000000;
    }
    
    .main {
        background: transparent;
        color: #000000;
    }

    /* Headings - very visible black */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Markdown text - black */
    .stMarkdown, p, span, div, li {
        color: #000000 !important;
    }

    /* Metrics - black */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }

    /* Chart labels and text */
    .js-plotly-plot .plotly text {
        fill: #000000 !important;
    }

    /* Weather cards */
    .weather-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(12px);
        border: 2px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar - white background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 2px solid rgba(0, 0, 0, 0.1);
    }
    
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: #ffffff !important;
        border-radius: 10px;
        border: none;
        padding: 0.5em 1.2em;
        font-weight: 600;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
    }
    
    /* Selectbox - black text */
    .stSelectbox label, .stSelectbox div {
        color: #000000 !important;
    }
    
    /* Dataframe - better contrast */
    .dataframe {
        color: #000000 !important;
    }
    
    /* Info/Warning/Error boxes - black text */
    .stAlert {
        color: #000000 !important;
    }
    
    /* Tabs - black text */
    .stTabs [data-baseweb="tab"] {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #1d4ed8 !important;
        border-bottom: 3px solid #1d4ed8 !important;
    }
    
    /* ChatBot messages */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f1f8e9, #dcedc8);
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================
DATA_PATH = Path("data/processed_flights.parquet")
MODEL_PATH = Path("models/flight_risk_model.pkl")

@st.cache_data
def load_data():
    """Load processed data"""
    try:
        df = pd.read_parquet(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
        st.stop()

@st.cache_resource
def load_model():
    """Load the model"""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"⚠️ Model not loaded: {e}")
        return None

# Real-time weather function
def get_stl_weather():
    """Retrieve real-time weather for STL (KSTL)"""
    try:
        # STL Lambert Airport coordinates
        lat, lon = 38.7487, -90.37
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m,cloud_cover&timezone=America/Chicago"
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        
        # WMO weather codes
        weather_codes = {
            0: "☀️ Clear sky", 1: "🌤️ Mainly clear", 2: "⛅ Partly cloudy",
            3: "☁️ Overcast", 45: "🌫️ Fog", 48: "🌫️ Depositing rime fog",
            51: "🌧️ Light drizzle", 53: "🌧️ Moderate drizzle", 55: "🌧️ Dense drizzle",
            61: "🌧️ Slight rain", 63: "🌧️ Moderate rain", 65: "🌧️ Heavy rain",
            71: "🌨️ Slight snow", 73: "🌨️ Moderate snow", 75: "🌨️ Heavy snow",
            80: "🌦️ Slight showers", 81: "🌦️ Moderate showers", 82: "🌦️ Violent showers",
            95: "⛈️ Thunderstorm", 96: "⛈️ Thunderstorm with slight hail", 99: "⛈️ Thunderstorm with heavy hail"
        }
        
        weather_code = current.get('weather_code', 0)
        weather_desc = weather_codes.get(weather_code, "🌍 Unknown conditions")
        
        # Weather risk calculation
        wind_speed_kmh = current.get('wind_speed_10m', 0)
        wind_speed_kt = wind_speed_kmh * 0.539957
        precip = current.get('precipitation', 0)
        cloud_cover = current.get('cloud_cover', 0)
        
        wx_risk = 0.0
        if wind_speed_kt > 20:
            wx_risk += 0.4
        elif wind_speed_kt > 10:
            wx_risk += 0.2
        
        if precip > 1:
            wx_risk += 0.3
        elif precip > 0:
            wx_risk += 0.1
        
        if cloud_cover > 80:
            wx_risk += 0.1
        
        if weather_code >= 80 or weather_code in [61, 63, 65, 71, 73, 75, 95, 96, 99]:
            wx_risk += 0.3
        
        wx_risk = min(1.0, wx_risk)
        
        return {
            'temperature': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'wind_speed_kt': wind_speed_kt,
            'wind_direction': current.get('wind_direction_10m'),
            'precipitation': precip,
            'cloud_cover': cloud_cover,
            'weather_desc': weather_desc,
            'wx_risk_score': wx_risk,
            'success': True
        }
    except Exception as e:
        return {
            'temperature': None,
            'humidity': None,
            'wind_speed_kt': None,
            'wind_direction': None,
            'precipitation': None,
            'cloud_cover': None,
            'weather_desc': '❌ Data unavailable',
            'wx_risk_score': 0.3,
            'success': False,
            'error': str(e)
        }

# OCC ChatBot Class
class OCCChatBot:
    """ChatBot for STL OCC"""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 800):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        
        system_prompt = """You are an expert assistant for the Operations Control Center (OCC) at STL airport.
        
Your role is to help operators:
- Understand flight disruption risk predictions
- Interpret risk factors (weather, congestion, rotation, etc.)
- Suggest concrete operational actions
- Answer questions about historical performance
- Analyze trends and patterns

You respond in English, clearly, concisely and actionably.
You use the provided context (flight data, predictions, weather) to give accurate answers.
If asked general questions about aviation or airport operations, you can answer even without specific context."""
        
        self.messages = [{"role": "system", "content": system_prompt}]
        self.context = {}
    
    def set_context(self, context: dict) -> None:
        self.context = context
    
    def ask(self, question: str, include_context: bool = True) -> str:
        user_message = question
        
        if include_context and self.context:
            context_str = "\n\n**OPERATIONAL CONTEXT:**\n"
            context_str += json.dumps(self.context, indent=2, ensure_ascii=False)
            user_message = f"{question}\n{context_str}"
        
        self.messages.append({"role": "user", "content": user_message})
        
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "OCC STL ChatBot"
        }
        
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                return f"❌ API Error ({response.status_code}): {response.text}"
            
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            
            self.messages.append({"role": "assistant", "content": answer})
            return answer
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def reset(self) -> None:
        self.messages = [self.messages[0]]

# Loading
df = load_data()
model_bundle = load_model()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'prediction_context' not in st.session_state:
    st.session_state.prediction_context = {}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# ============================================================================
# AUTHENTICATION SYSTEM
# ============================================================================

# User database (replace with real DB in production)
USERS_DB = {
    "admin": {
        "password": "occ2024",  # Change in production!
        "role": "Administrator",
        "full_name": "OCC Administrator"
    },
    "operator": {
        "password": "stl2024",
        "role": "Operator",
        "full_name": "OCC Operator"
    },
    "viewer": {
        "password": "view2024",
        "role": "Viewer",
        "full_name": "Viewer"
    }
}

def verify_credentials(username, password):
    """Verify login credentials"""
    if username in USERS_DB:
        if USERS_DB[username]["password"] == password:
            return True, USERS_DB[username]
    return False, None

def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.chatbot = None
    st.session_state.chat_history = []
    st.session_state.prediction_context = {}

# LOGIN PAGE
if not st.session_state.authenticated:
    # Center login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3rem;'>✈️</h1>
            <h2>OCC STL Dashboard</h2>
            <p style='color: #666; margin-bottom: 2rem;'>Operations Control Center - St. Louis Lambert Airport</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔐 Secure Login")
        
        # Login form
        with st.form("login_form"):
            username_input = st.text_input("👤 Username", placeholder="Enter your username")
            password_input = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                login_button = st.form_submit_button("🚀 Login", use_container_width=True)
            with col_btn2:
                if st.form_submit_button("ℹ️ Help", use_container_width=True):
                    st.info("Contact the OCC administrator to obtain your credentials.")
        
        if login_button:
            if username_input and password_input:
                is_valid, user_info = verify_credentials(username_input, password_input)
                
                if is_valid:
                    st.session_state.authenticated = True
                    st.session_state.username = username_input
                    st.session_state.user_info = user_info
                    st.success(f"✅ Welcome {user_info['full_name']}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Incorrect credentials. Please try again.")
            else:
                st.warning("⚠️ Please fill in all fields.")
        
        # Test information
        st.markdown("---")
        with st.expander("🧪 Available test accounts"):
            st.markdown("""
            **To test the application:**
            
            - **Administrator**: `admin` / `occ2024`
            - **Operator**: `operator` / `stl2024`
            - **Viewer**: `viewer` / `view2024`
            
            ⚠️ **Note**: These credentials are for demonstration purposes only.
            """)
        
        st.markdown("""
        <div style='text-align: center; margin-top: 3rem; color: #666; font-size: 0.9em;'>
            <p>🔒 Secure connection | Confidential OCC data</p>
            <p>© 2024 STL Operations Control Center</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# ============================================================================
# HEADER (Displayed only if authenticated)
# ============================================================================
# Header with user info
col_title, col_user = st.columns([4, 1])

with col_title:
    st.title("✈️ OCC STL - Flight Disruption Dashboard")
    st.markdown("**Operations Control Center** - St. Louis Lambert International Airport")

with col_user:
    st.markdown(f"""
    <div style='text-align: right; padding: 1rem 0;'>
        <p style='margin: 0; color: #666; font-size: 0.9em;'>👤 Logged in as</p>
        <p style='margin: 0; font-weight: 700;'>{st.session_state.user_info['full_name']}</p>
        <p style='margin: 0; color: #666; font-size: 0.85em;'>🎭 {st.session_state.user_info['role']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()

st.markdown("---")

# ============================================================================
# SIDEBAR - FILTERS
# ============================================================================
st.sidebar.header("🎛️ Filters")

# Airline Filter
airlines = ["All"] + sorted(df['Airline'].dropna().unique().tolist())
selected_airline = st.sidebar.selectbox("Airline", airlines)

# Destination Filter
destinations = ["All"] + sorted(df['Dest_Airport'].dropna().unique().tolist())
selected_dest = st.sidebar.selectbox("Destination", destinations)

# Time Slot Filter
time_slots = ["All"] + sorted(df['DepSlot'].dropna().unique().tolist())
selected_slot = st.sidebar.selectbox("Time Slot", time_slots)

# Month Filter
months = ["All"] + sorted(df['Month'].dropna().unique().tolist())
selected_month = st.sidebar.selectbox("Month", months)

# Day of Week Filter
day_mapping = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday"
}

# Create list of available days
available_days = sorted(df['DayOfWeek'].dropna().unique().tolist())
day_options = ["All"] + [f"{day_mapping.get(day, f'Day {day}')} ({day})" for day in available_days]
selected_day = st.sidebar.selectbox("Day of Week", day_options)

# Apply filters
df_filtered = df.copy()

if selected_airline != "All":
    df_filtered = df_filtered[df_filtered['Airline'] == selected_airline]

if selected_dest != "All":
    df_filtered = df_filtered[df_filtered['Dest_Airport'] == selected_dest]

if selected_slot != "All":
    df_filtered = df_filtered[df_filtered['DepSlot'] == selected_slot]

if selected_month != "All":
    df_filtered = df_filtered[df_filtered['Month'] == selected_month]

if selected_day != "All":
    # Extract day number from selection (e.g., "Monday (1)" -> 1)
    day_num = int(selected_day.split('(')[1].split(')')[0])
    df_filtered = df_filtered[df_filtered['DayOfWeek'] == day_num]

# Check data after filtering
if len(df_filtered) == 0:
    st.error("⚠️ No data available with these filters. Please adjust your criteria.")
    st.stop()

# Test for NaN
nan_count = df_filtered.isna().sum().sum()
if nan_count > 0:
    st.sidebar.warning(f"⚠️ {nan_count} missing values detected in filtered data")

st.sidebar.markdown("---")
st.sidebar.info(f"📊 **{len(df_filtered):,}** flights in selection")

# ============================================================================
# REAL-TIME WEATHER STL
# ============================================================================
st.header("🌤️ Real-Time Weather - STL Lambert Airport")

weather_data = get_stl_weather()

if weather_data['success']:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🌡️ Temperature", 
                 f"{weather_data['temperature']:.1f}°C" if weather_data['temperature'] else "N/A")
    
    with col2:
        st.metric("💨 Wind", 
                 f"{weather_data['wind_speed_kt']:.1f} kt" if weather_data['wind_speed_kt'] else "N/A")
    
    with col3:
        st.metric("💧 Humidity", 
                 f"{weather_data['humidity']:.0f}%" if weather_data['humidity'] else "N/A")
    
    with col4:
        st.metric("🌧️ Precipitation", 
                 f"{weather_data['precipitation']:.1f} mm" if weather_data['precipitation'] is not None else "N/A")
    
    with col5:
        risk_level = "🔴 High" if weather_data['wx_risk_score'] > 0.6 else "🟡 Moderate" if weather_data['wx_risk_score'] > 0.3 else "🟢 Low"
        st.metric("⚠️ Weather Risk", risk_level)
    
    st.info(f"**Current conditions:** {weather_data['weather_desc']} | Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Weather alert if necessary
    if weather_data['wx_risk_score'] > 0.6:
        st.error("⚠️ **WEATHER ALERT** - Unfavorable conditions detected. Anticipate delays and coordinate with MET/ATC.")
    elif weather_data['wx_risk_score'] > 0.3:
        st.warning("🟡 Weather conditions to monitor. Plan extra margins for sensitive flights.")
else:
    st.warning(f"⚠️ Unable to retrieve weather data: {weather_data.get('error', 'Unknown error')}")

st.markdown("---")

# ============================================================================
# KEY PERFORMANCE INDICATORS
# ============================================================================
st.header("📊 Key Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_flights = len(df_filtered)
    st.metric("Total Flights", f"{total_flights:,}")

with col2:
    avg_delay = df_filtered['ArrDelay'].mean()
    st.metric("Average Delay", f"{avg_delay:.1f} min")

with col3:
    pct_delayed = (df_filtered['ArrDelay'] > 15).mean() * 100
    st.metric("Delayed Flights >15min", f"{pct_delayed:.1f}%")

with col4:
    high_risk_pct = (df_filtered['disruption_risk'] == 'HIGH_DISRUPTION').mean() * 100
    st.metric("High Risk", f"{high_risk_pct:.1f}%")

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Overview",
    "🗂️ Data",
    "🎯 Risks",
    "🔍 Prediction",
    "🗺️ Airport Map",
    "💬 OCC ChatBot",
    "📚 Guide"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.subheader("Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Disruption Risk - Simple pie
        risk_counts = df_filtered['disruption_risk'].value_counts()
        fig1 = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Disruption Risk",
            color_discrete_sequence=['#00D9FF', '#FFB800', '#FF4444']
        )
        fig1.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font_color='#FFFFFF'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Congestion Risk - Simple pie
        cong_counts = df_filtered['congestion_risk'].value_counts()
        fig2 = px.pie(
            values=cong_counts.values,
            names=cong_counts.index,
            title="Congestion Level",
            color_discrete_sequence=['#00FF88', '#FFB800', '#FF4444']
        )
        fig2.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font_color='#FFFFFF'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Delays by Month")
    
    monthly = df_filtered.groupby('Month')['ArrDelay'].mean().reset_index()
    fig3 = px.bar(
        monthly,
        x='Month',
        y='ArrDelay',
        title="Average Delay by Month",
        labels={'ArrDelay': 'Delay (min)', 'Month': 'Month'},
        color='ArrDelay',
        color_continuous_scale='Reds'
    )
    fig3.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#000000', size=12),
        title_font=dict(color='#000000', size=16, family='Arial Black')
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Performance by Airline")
    
    airline_perf = df_filtered.groupby('Airline').agg({
        'ArrDelay': 'mean',
        'Airline': 'count'
    }).rename(columns={'Airline': 'n_flights'}).reset_index()
    airline_perf = airline_perf.sort_values('ArrDelay', ascending=False).head(10)
    
    fig4 = px.bar(
        airline_perf,
        x='Airline',
        y='ArrDelay',
        title="Top 10 Airlines - Average Delay",
        labels={'ArrDelay': 'Delay (min)', 'Airline': 'Airline'},
        color='ArrDelay',
        color_continuous_scale='Blues'
    )
    fig4.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font_color='#FFFFFF'
    )
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================================
# TAB 2: DATA
# ============================================================================
with tab2:
    st.subheader("Database Description")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", f"{len(df.columns)}")
    with col3:
        st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    st.markdown("---")
    st.subheader("Data Preview")
    st.dataframe(df_filtered.head(100), use_container_width=True, height=400)
    
    st.markdown("---")
    st.subheader("Numerical Variables")
    
    num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    selected_num = st.selectbox("Choose a variable", num_cols)
    
    fig5 = px.histogram(
        df_filtered,
        x=selected_num,
        nbins=30,
        title=f"Distribution of {selected_num}",
        color_discrete_sequence=['#00D9FF']
    )
    fig5.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font_color='#FFFFFF'
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Descriptive Statistics")
    st.dataframe(df_filtered[num_cols].describe(), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Categorical Variables")
    
    cat_cols = ['Airline', 'Dest', 'UniqueCarrier', 'DepSlot', 'Origin', 
                'Org_Airport', 'Dest_Airport', 'disruption_risk', 'congestion_risk']
    cat_cols = [c for c in cat_cols if c in df_filtered.columns]
    selected_cat = st.selectbox("Choose a categorical variable", cat_cols)
    
    cat_counts = df_filtered[selected_cat].value_counts().head(10)
    fig6 = px.bar(
        x=cat_counts.index,
        y=cat_counts.values,
        title=f"Top 10 - {selected_cat}",
        labels={'x': selected_cat, 'y': 'Count'},
        color=cat_counts.values,
        color_continuous_scale='Viridis'
    )
    fig6.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font_color='#FFFFFF'
    )
    st.plotly_chart(fig6, use_container_width=True)

# ============================================================================
# TAB 3: RISKS
# ============================================================================
with tab3:
    st.subheader("Risk Analysis")
    
    st.markdown("#### Risk by Departure Hour")
    hourly = df_filtered.groupby('DepHour')['disruption_risk'].apply(
        lambda x: (x == 'HIGH_DISRUPTION').mean() * 100
    ).reset_index()
    hourly.columns = ['DepHour', 'Pct_High_Risk']
    
    fig7 = px.line(
        hourly,
        x='DepHour',
        y='Pct_High_Risk',
        title="% High Risk by Hour",
        labels={'Pct_High_Risk': '% High Risk', 'DepHour': 'Hour'},
        markers=True
    )
    fig7.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font_color='#FFFFFF'
    )
    st.plotly_chart(fig7, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### High-Risk Routes")
    
    route_risk = df_filtered.groupby('Dest_Airport')['disruption_risk'].apply(
        lambda x: (x == 'HIGH_DISRUPTION').mean() * 100
    ).reset_index()
    route_risk.columns = ['Dest', 'Pct_High_Risk']
    route_risk = route_risk.sort_values('Pct_High_Risk', ascending=False).head(10)
    
    fig8 = px.bar(
        route_risk,
        x='Dest',
        y='Pct_High_Risk',
        title="Top 10 Destinations - % High Risk",
        labels={'Pct_High_Risk': '% High Risk', 'Dest': 'Destination'},
        color='Pct_High_Risk',
        color_continuous_scale='Reds'
    )
    fig8.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font_color='#FFFFFF'
    )
    st.plotly_chart(fig8, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Weather Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig9 = px.scatter(
            df_filtered.sample(min(500, len(df_filtered))),
            x='wx_risk_score',
            y='ArrDelay',
            color='disruption_risk',
            title="Weather vs Delay",
            labels={'wx_risk_score': 'Weather Score', 'ArrDelay': 'Delay (min)'},
            color_discrete_sequence=['#00FF88', '#FFB800', '#FF4444'],
            opacity=0.6
        )
        fig9.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font_color='#FFFFFF'
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    with col2:
        fig10 = px.histogram(
            df_filtered,
            x='wx_risk_score',
            nbins=20,
            title="Weather Score Distribution",
            color_discrete_sequence=['#00D9FF']
        )
        fig10.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font_color='#FFFFFF'
        )
        st.plotly_chart(fig10, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Airport Congestion")
    
    fig11 = px.histogram(
        df_filtered,
        x='airport_congestion_score',
        nbins=20,
        title="Congestion Score Distribution",
        color_discrete_sequence=['#FFB800']
    )
    fig11.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font_color='#FFFFFF'
    )
    st.plotly_chart(fig11, use_container_width=True)

# ============================================================================
# TAB 4: PREDICTION
# ============================================================================
with tab4:
    st.subheader("Flight Prediction")
    
    if model_bundle is None:
        st.warning("⚠️ Model not available. This feature is disabled.")
    else:
        # Flight selection
        flight_options = [f"Flight #{i} - {row['Airline']} → {row['Dest_Airport']} at {row['DepHour']}h" 
                         for i, row in df_filtered.iterrows()]
        
        if len(flight_options) == 0:
            st.error("No flights available in selection")
        else:
            selected_flight = st.selectbox("Choose a flight", range(len(flight_options)), 
                                          format_func=lambda x: flight_options[x])
            
            flight_idx = df_filtered.index[selected_flight]
            sample_row = df_filtered.loc[flight_idx]
            
            # Prepare features
            num_features = model_bundle['num_features']
            cat_features = model_bundle['cat_features']
            all_features = num_features + cat_features
            
            X_sample = sample_row[all_features].to_frame().T
            
            # Prediction
            clf = model_bundle['pipeline']
            le = model_bundle['label_encoder']
            
            proba = clf.predict_proba(X_sample)[0]
            pred_idx = int(np.argmax(proba))
            pred_label = le.inverse_transform([pred_idx])[0]
            
            proba_dict = {cls: float(proba[i]) for i, cls in enumerate(le.classes_)}
            
            # Save context for chatbot with current weather
            st.session_state.prediction_context = {
                "flight": {
                    "airline": sample_row.get('Airline', 'N/A'),
                    "destination": sample_row.get('Dest', 'N/A'),
                    "departure_hour": int(sample_row.get('DepHour', 0)),
                    "distance": float(sample_row.get('Distance', 0)),
                    "time_slot": sample_row.get('DepSlot', 'N/A')
                },
                "prediction": {
                    "predicted_risk": pred_label,
                    "confidence": float(proba[pred_idx] * 100),
                    "probabilities": proba_dict
                },
                "risk_factors": {
                    "airport_congestion": float(sample_row.get('airport_congestion_score', 0)),
                    "weather_score": float(sample_row.get('wx_risk_score', 0)),
                    "late_inbound": bool(sample_row.get('late_inbound_flag', 0)),
                    "short_turnaround": bool(sample_row.get('short_turnaround_flag', 0)),
                    "delay_risk_index": float(sample_row.get('DelayRiskIndex', 0)),
                    "route_avg_delay": float(sample_row.get('route_mean_delay', 0)),
                    "carrier_delay_rate": float(sample_row.get('carrier_delay_rate', 0))
                },
                "operational_context": {
                    "flights_same_hour": int(sample_row.get('flights_origin_hour', 0)),
                    "additional_taxiout": float(sample_row.get('additional_taxiout', 0)),
                    "route_cancel_rate": float(sample_row.get('route_cancel_rate', 0))
                },
                "current_weather": weather_data
            }
            
            # Display
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Prediction")
                risk_emoji = {'NO_DISRUPTION': '🟢', 'RISK_DISRUPTION': '🟡', 'HIGH_DISRUPTION': '🔴'}
                st.markdown(f"## {risk_emoji.get(pred_label, '⚪')} {pred_label}")
                st.markdown(f"**Confidence: {proba[pred_idx]*100:.1f}%**")
            
            with col2:
                st.markdown("### Flight Details")
                st.write(f"**Airline:** {sample_row.get('Airline', 'N/A')}")
                st.write(f"**Destination:** {sample_row.get('Dest', 'N/A')}")
                st.write(f"**Hour:** {sample_row.get('DepHour', 'N/A')}h")
                st.write(f"**Distance:** {sample_row.get('Distance', 'N/A')} miles")
            
            with col3:
                st.markdown("### Probabilities")
                for cls in le.classes_:
                    st.write(f"{cls}: **{proba_dict[cls]*100:.1f}%**")
            
            st.markdown("---")
            
            # Probability chart
            fig12 = px.bar(
                x=list(proba_dict.keys()),
                y=list(proba_dict.values()),
                title="Probability Distribution",
                labels={'x': 'Risk', 'y': 'Probability'},
                color=list(proba_dict.values()),
                color_continuous_scale='RdYlGn_r'
            )
            fig12.update_layout(
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#FFFFFF',
                font_color='#FFFFFF'
            )
            st.plotly_chart(fig12, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Recommendations")
            
            if pred_label == 'HIGH_DISRUPTION':
                st.error("🔴 **HIGH ALERT** - Escalate to duty manager")
                st.warning("📞 Monitor this flight in real-time")
            elif pred_label == 'RISK_DISRUPTION':
                st.warning("🟡 **SURVEILLANCE** - Place under enhanced monitoring")
            else:
                st.success("🟢 **NORMAL** - Standard surveillance")
            
            if sample_row.get('airport_congestion_score', 0) > 0.7:
                st.info("🏢 High congestion - Coordinate with ATC")
            
            if sample_row.get('wx_risk_score', 0) > 0.6:
                st.info("🌧️ Weather risk - Plan extra margins")
            
            st.markdown("---")
            st.success("✅ **Context saved** - You can now query the OCC ChatBot in the next tab!")

# ============================================================================
# TAB 5: STL AIRPORT MAP
# ============================================================================
with tab5:
    st.subheader("🗺️ Interactive Map of STL Lambert Airport")
    st.markdown("Detailed view of facilities, terminals, runways, and operational areas")
    
    # General airport information
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏢 IATA Code", "STL")
        st.metric("📡 ICAO Code", "KSTL")
    
    with col2:
        st.metric("🛫 Runways", "2 (12L/30R, 12R/30L)")
        st.metric("📏 Max Length", "11,019 ft")
    
    with col3:
        st.metric("🏗️ Terminals", "2 (Terminal 1 & 2)")
        st.metric("🚪 Gates", "84")
    
    with col4:
        st.metric("⏰ Time Zone", "America/Chicago (CST)")
        st.metric("📍 Elevation", "618 ft / 188 m")
    
    st.markdown("---")
    
    # Interactive map with Folium/Plotly
    st.markdown("### 📍 Location and Infrastructure")
    
    # STL Lambert coordinates
    stl_lat, stl_lon = 38.7487, -90.3700
    
    # Create a map with Plotly
    fig_map = go.Figure()
    
    # Add the airport central point
    fig_map.add_trace(go.Scattermapbox(
        lat=[stl_lat],
        lon=[stl_lon],
        mode='markers+text',
        marker=dict(size=20, color='#3b82f6'),
        text=["STL Lambert Airport"],
        textposition="top center",
        name="Main Airport"
    ))
    
    # Add runways (approximation)
    # Runway 12L/30R
    runway_12L_30R_lat = [38.7387, 38.7587]
    runway_12L_30R_lon = [-90.3800, -90.3600]
    
    fig_map.add_trace(go.Scattermapbox(
        lat=runway_12L_30R_lat,
        lon=runway_12L_30R_lon,
        mode='lines+markers',
        line=dict(width=3, color='#f59e0b'),
        marker=dict(size=10, color='#f59e0b'),
        name="Runway 12L/30R",
        text=["12L", "30R"],
        hoverinfo='text'
    ))
    
    # Runway 12R/30L
    runway_12R_30L_lat = [38.7387, 38.7587]
    runway_12R_30L_lon = [-90.3650, -90.3450]
    
    fig_map.add_trace(go.Scattermapbox(
        lat=runway_12R_30L_lat,
        lon=runway_12R_30L_lon,
        mode='lines+markers',
        line=dict(width=3, color='#ef4444'),
        marker=dict(size=10, color='#ef4444'),
        name="Runway 12R/30L",
        text=["12R", "30L"],
        hoverinfo='text'
    ))
    
    # Points of interest
    poi = {
        "Terminal 1": (38.7497, -90.3710, "#10b981"),
        "Terminal 2": (38.7477, -90.3690, "#10b981"),
        "Control Tower": (38.7487, -90.3700, "#8b5cf6"),
        "Cargo Area": (38.7450, -90.3750, "#f97316"),
        "Maintenance Hangars": (38.7420, -90.3780, "#06b6d4"),
    }
    
    for name, (lat, lon, color) in poi.items():
        fig_map.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers+text',
            marker=dict(size=12, color=color),
            text=[name],
            textposition="bottom center",
            name=name,
            hoverinfo='text'
        ))
    
    # Map configuration
    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=stl_lat, lon=stl_lon),
            zoom=13
        ),
        showlegend=True,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#000000')
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    st.markdown("---")
    
    # Facility details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛫 Runways and Operations")
        
        runway_data = pd.DataFrame({
            'Runway': ['12L/30R', '12R/30L'],
            'Length (ft)': [11019, 9003],
            'Width (ft)': [150, 150],
            'Surface': ['Concrete', 'Concrete'],
            'ILS': ['Yes (CAT III)', 'Yes (CAT I)'],
            'Primary Use': ['Heavy Arrivals/Departures', 'General Traffic']
        })
        
        st.dataframe(runway_data, use_container_width=True, hide_index=True)
        
        st.info("💡 **Current configuration**: Simultaneous operations on both runways under VMC conditions")
    
    with col2:
        st.markdown("### 🏢 Terminals and Capacity")
        
        terminal_data = pd.DataFrame({
            'Terminal': ['Terminal 1', 'Terminal 2'],
            'Concourses': ['A, B, C, D', 'E'],
            'Gates': ['54', '30'],
            'Main Airlines': ['Southwest, American', 'Delta, United, others'],
            'Passenger Capacity/year': ['7.5M', '7.5M']
        })
        
        st.dataframe(terminal_data, use_container_width=True, hide_index=True)
        
        st.success("✅ **Status**: Terminals operating at normal capacity")
    
    st.markdown("---")
    
    # Real-time operational statistics
    st.markdown("### 📊 Operational Statistics (Filtered Data)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        daily_ops = len(df_filtered)
        st.metric("Daily Operations", f"{daily_ops:,}")
    
    with col2:
        avg_taxi_out = df_filtered['TaxiOut'].mean() if 'TaxiOut' in df_filtered.columns else 0
        st.metric("Average Taxi-Out", f"{avg_taxi_out:.1f} min")
    
    with col3:
        peak_hour = df_filtered.groupby('DepHour').size().idxmax() if len(df_filtered) > 0 else 0
        st.metric("Peak Hour", f"{peak_hour:02d}:00")
    
    with col4:
        congestion_avg = df_filtered['airport_congestion_score'].mean() if 'airport_congestion_score' in df_filtered.columns else 0
        st.metric("Average Congestion", f"{congestion_avg:.2f}")
    
    st.markdown("---")
    
    # Flight distribution chart by airline
    st.markdown("### 🚪 Flight Distribution by Airline")
    
    if len(df_filtered) > 0:
        airline_distribution = df_filtered.groupby('Airline').size().reset_index(name='Number of Flights')
        airline_distribution = airline_distribution.sort_values('Number of Flights', ascending=False).head(10)
        
        fig_airlines = px.bar(
            airline_distribution,
            x='Airline',
            y='Number of Flights',
            title="Top 10 Airlines - Number of Flights",
            color='Number of Flights',
            color_continuous_scale='Blues'
        )
        
        fig_airlines.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font=dict(color='#000000', size=12),
            title_font=dict(color='#000000', size=16, family='Arial Black')
        )
        
        st.plotly_chart(fig_airlines, use_container_width=True)
    
    st.markdown("---")
    
    # Service areas
    st.markdown("### 🔧 Service and Maintenance Areas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🛠️ Maintenance**
        - 3 main hangars
        - Capacity: Boeing 777
        - 24/7 services
        - FAA-certified teams
        """)
    
    with col2:
        st.markdown("""
        **⛽ Refueling**
        - 4 service points
        - Jet A-1 available
        - Capacity: 500,000 gal
        - Hydrant system
        """)
    
    with col3:
        st.markdown("""
        **📦 Cargo**
        - Area: 250,000 sq ft
        - 12 cargo stands
        - On-site customs
        - 24/7 operations
        """)
    
    st.markdown("---")
    
    # Contact and emergency information
    st.markdown("### 📞 Emergency Contacts and Coordination")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🚨 Airport Emergencies**
        - Fire/Rescue: Extension 911
        - Security: Extension 2222
        - Medical: Extension 3333
        - OCC Duty Manager: Extension 5000
        """)
    
    with col2:
        st.markdown("""
        **📡 Operational Coordination**
        - TWR (Tower): 118.5 MHz
        - GND (Ground): 121.9 MHz
        - ATIS: 132.65 MHz
        - UNICOM: 122.95 MHz
        """)
    
    st.markdown("---")
    
    # Map legend
    with st.expander("ℹ️ Map Legend"):
        st.markdown("""
        **Symbols and Colors:**
        
        - 🔵 **Blue**: Main airport
        - 🟠 **Orange**: Runway 12L/30R (primary)
        - 🔴 **Red**: Runway 12R/30L (secondary)
        - 🟢 **Green**: Passenger terminals
        - 🟣 **Purple**: ATC control tower
        - 🟠 **Dark orange**: Cargo area
        - 🔵 **Cyan**: Maintenance hangars
        
        **Navigation:**
        - Use the scroll wheel to zoom in/out
        - Click and drag to move the map
        - Click markers for more information
        """)
    
    st.markdown("---")
    # ========================================================================
    # DETAILED MAPPING OF THE AIRPORT COMPLEX
    # ========================================================================
    st.markdown("### 🏗️ Detailed Mapping of the Airport Complex")
    st.markdown("Schematic layout of zones, access points, and security perimeters")

    # Create an airport layout diagram with Plotly
    fig_layout = go.Figure()

    # Define airport zones (rectangles and polygons)

    # 1. AIRPORT PERIMETER (overall zone)
    perimeter = {
        'x': [0, 100, 100, 0, 0],
        'y': [0, 0, 80, 80, 0],
        'name': 'Airport Perimeter',
        'color': 'rgba(200, 200, 200, 0.3)'
    }

    fig_layout.add_trace(go.Scatter(
        x=perimeter['x'],
        y=perimeter['y'],
        fill='toself',
        fillcolor=perimeter['color'],
        line=dict(color='gray', width=3, dash='dash'),
        name=perimeter['name'],
        hoverinfo='name'
    ))

    # 2. RUNWAY 12L/30R (primary)
    runway1 = {
        'x': [10, 90, 90, 10, 10],
        'y': [35, 35, 40, 40, 35],
        'name': 'Runway 12L/30R',
        'color': 'rgba(245, 158, 11, 0.7)'
    }

    fig_layout.add_trace(go.Scatter(
        x=runway1['x'],
        y=runway1['y'],
        fill='toself',
        fillcolor=runway1['color'],
        line=dict(color='#f59e0b', width=2),
        name=runway1['name'],
        text='11,019 ft',
        hoverinfo='name+text'
    ))

    # Runway threshold markers
    fig_layout.add_trace(go.Scatter(
        x=[10, 90],
        y=[37.5, 37.5],
        mode='markers+text',
        marker=dict(size=12, color='white', symbol='square', line=dict(color='#f59e0b', width=2)),
        text=['12L', '30R'],
        textposition='top center',
        showlegend=False,
        hoverinfo='text'
    ))

    # 3. RUNWAY 12R/30L (secondary)
    runway2 = {
        'x': [15, 85, 85, 15, 15],
        'y': [50, 50, 55, 55, 50],
        'name': 'Runway 12R/30L',
        'color': 'rgba(239, 68, 68, 0.7)'
    }

    fig_layout.add_trace(go.Scatter(
        x=runway2['x'],
        y=runway2['y'],
        fill='toself',
        fillcolor=runway2['color'],
        line=dict(color='#ef4444', width=2),
        name=runway2['name'],
        text='9,003 ft',
        hoverinfo='name+text'
    ))

    # Runway threshold markers
    fig_layout.add_trace(go.Scatter(
        x=[15, 85],
        y=[52.5, 52.5],
        mode='markers+text',
        marker=dict(size=12, color='white', symbol='square', line=dict(color='#ef4444', width=2)),
        text=['12R', '30L'],
        textposition='top center',
        showlegend=False,
        hoverinfo='text'
    ))

    # 4. TERMINAL 1 (Southwest, American)
    terminal1 = {
        'x': [20, 35, 35, 20, 20],
        'y': [15, 15, 25, 25, 15],
        'name': 'Terminal 1',
        'color': 'rgba(16, 185, 129, 0.8)'
    }

    fig_layout.add_trace(go.Scatter(
        x=terminal1['x'],
        y=terminal1['y'],
        fill='toself',
        fillcolor=terminal1['color'],
        line=dict(color='#10b981', width=2),
        name=terminal1['name'],
        text='T1<br>54 Gates<br>Concourses A–D',
        hoverinfo='name+text'
    ))

    fig_layout.add_trace(go.Scatter(
        x=[27.5],
        y=[20],
        mode='text',
        text=['T1'],
        textfont=dict(size=14, color='white', family='Arial Black'),
        showlegend=False
    ))

    # 5. TERMINAL 2 (Delta, United)
    terminal2 = {
        'x': [40, 55, 55, 40, 40],
        'y': [15, 15, 25, 25, 15],
        'name': 'Terminal 2',
        'color': 'rgba(16, 185, 129, 0.8)'
    }

    fig_layout.add_trace(go.Scatter(
        x=terminal2['x'],
        y=terminal2['y'],
        fill='toself',
        fillcolor=terminal2['color'],
        line=dict(color='#10b981', width=2),
        name=terminal2['name'],
        text='T2<br>30 Gates<br>Concourse E',
        hoverinfo='name+text'
    ))

    fig_layout.add_trace(go.Scatter(
        x=[47.5],
        y=[20],
        mode='text',
        text=['T2'],
        textfont=dict(size=14, color='white', family='Arial Black'),
        showlegend=False
    ))

    # 6. CONTROL TOWER
    tower = {
        'x': [45, 50, 50, 45, 45],
        'y': [42, 42, 47, 47, 42],
        'name': 'ATC Control Tower',
        'color': 'rgba(139, 92, 246, 0.8)'
    }

    fig_layout.add_trace(go.Scatter(
        x=tower['x'],
        y=tower['y'],
        fill='toself',
        fillcolor=tower['color'],
        line=dict(color='#8b5cf6', width=2),
        name=tower['name'],
        text='ATC<br>TWR',
        hoverinfo='name+text'
    ))

    fig_layout.add_trace(go.Scatter(
        x=[47.5],
        y=[44.5],
        mode='text',
        text=['🗼'],
        textfont=dict(size=20),
        showlegend=False
    ))

    # 7. CARGO AREA
    cargo = {
        'x': [65, 85, 85, 65, 65],
        'y': [10, 10, 25, 25, 10],
        'name': 'Cargo Area',
        'color': 'rgba(249, 115, 22, 0.7)'
    }

    fig_layout.add_trace(go.Scatter(
        x=cargo['x'],
        y=cargo['y'],
        fill='toself',
        fillcolor=cargo['color'],
        line=dict(color='#f97316', width=2),
        name=cargo['name'],
        text='Cargo<br>250,000 sq ft<br>12 positions',
        hoverinfo='name+text'
    ))

    fig_layout.add_trace(go.Scatter(
        x=[75],
        y=[17.5],
        mode='text',
        text=['📦 CARGO'],
        textfont=dict(size=12, color='white', family='Arial Black'),
        showlegend=False
    ))

    # 8. MAINTENANCE HANGARS
    hangars = {
        'x': [5, 18, 18, 5, 5],
        'y': [60, 60, 72, 72, 60],
        'name': 'Maintenance Hangars',
        'color': 'rgba(6, 182, 212, 0.7)'
    }

    fig_layout.add_trace(go.Scatter(
        x=hangars['x'],
        y=hangars['y'],
        fill='toself',
        fillcolor=hangars['color'],
        line=dict(color='#06b6d4', width=2),
        name=hangars['name'],
        text='Maintenance<br>3 Hangars<br>Cap. B777',
        hoverinfo='name+text'
    ))

    fig_layout.add_trace(go.Scatter(
        x=[11.5],
        y=[66],
        mode='text',
        text=['🔧 MNT'],
        textfont=dict(size=12, color='white', family='Arial Black'),
        showlegend=False
    ))

    # 9. GENERAL AVIATION PARKING
    ga_parking = {
        'x': [85, 95, 95, 85, 85],
        'y': [60, 60, 70, 70, 60],
        'name': 'General Aviation',
        'color': 'rgba(168, 85, 247, 0.6)'
    }

    fig_layout.add_trace(go.Scatter(
        x=ga_parking['x'],
        y=ga_parking['y'],
        fill='toself',
        fillcolor=ga_parking['color'],
        line=dict(color='#a855f7', width=2),
        name=ga_parking['name'],
        text='GA Parking',
        hoverinfo='name+text'
    ))

    # 10. REFUELING AREA
    fuel = {
        'x': [22, 30, 30, 22, 22],
        'y': [58, 58, 64, 64, 58],
        'name': 'Refueling',
        'color': 'rgba(234, 179, 8, 0.7)'
    }

    fig_layout.add_trace(go.Scatter(
        x=fuel['x'],
        y=fuel['y'],
        fill='toself',
        fillcolor=fuel['color'],
        line=dict(color='#eab308', width=2),
        name=fuel['name'],
        text='⛽ Fuel<br>500K gal',
        hoverinfo='name+text'
    ))

    # 11. TAXIWAYS
    # Taxiway Alpha
    fig_layout.add_trace(go.Scatter(
        x=[10, 90],
        y=[30, 30],
        mode='lines',
        line=dict(color='yellow', width=3, dash='dot'),
        name='Taxiway Alpha',
        hoverinfo='name'
    ))

    # Taxiway Bravo
    fig_layout.add_trace(go.Scatter(
        x=[30, 30],
        y=[25, 60],
        mode='lines',
        line=dict(color='yellow', width=3, dash='dot'),
        name='Taxiway Bravo',
        hoverinfo='name'
    ))

    # 12. ROAD ACCESS
    # Main road
    fig_layout.add_trace(go.Scatter(
        x=[0, 20],
        y=[20, 20],
        mode='lines',
        line=dict(color='black', width=4),
        name='Main Access',
        hoverinfo='name'
    ))

    fig_layout.add_trace(go.Scatter(
        x=[10],
        y=[20],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='triangle-up'),
        text=['🚗 Entrance'],
        textposition='top center',
        showlegend=False
    ))

    # 13. SECURITY ZONES
    # Sterile zone (airside)
    security_zone = {
        'x': [10, 90, 90, 10, 10],
        'y': [28, 28, 75, 75, 28],
        'name': 'Secure Zone (Airside)',
        'color': 'rgba(220, 38, 38, 0.1)'
    }

    fig_layout.add_trace(go.Scatter(
        x=security_zone['x'],
        y=security_zone['y'],
        fill='toself',
        fillcolor=security_zone['color'],
        line=dict(color='red', width=2, dash='dash'),
        name=security_zone['name'],
        hoverinfo='name'
    ))

    # 14. PASSENGER PARKING
    parking_lots = [
        {'x': [5, 15, 15, 5, 5], 'y': [5, 5, 12, 12, 5], 'name': 'Parking A'},
        {'x': [58, 68, 68, 58, 58], 'y': [5, 5, 12, 12, 5], 'name': 'Parking B'},
    ]

    for i, parking in enumerate(parking_lots):
        fig_layout.add_trace(go.Scatter(
            x=parking['x'],
            y=parking['y'],
            fill='toself',
            fillcolor='rgba(100, 100, 100, 0.5)',
            line=dict(color='gray', width=1),
            name=parking['name'],
            text='🅿️',
            hoverinfo='name'
        ))

    # Layout configuration
    fig_layout.update_layout(
        title={
            'text': 'Schematic Plan – STL Lambert International Airport',
            'font': {'size': 18, 'color': '#000000', 'family': 'Arial Black'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-5, 105]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-5, 85],
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=10, color='#000000')
        ),
        height=700,
        paper_bgcolor='#f0f9ff',
        plot_bgcolor='#e0f2fe',
        font=dict(color='#000000'),
        hovermode='closest'
    )

    st.plotly_chart(fig_layout, use_container_width=True)

    # Additional zone information
    st.markdown("---")
    st.markdown("### 📋 Operational Zone Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🟢 Terminal Zone (Landside)**
        - Terminal 1: Concourses A, B, C, D
        - Terminal 2: Concourse E
        - Passenger services
        - Shops and restaurants
        - TSA security screening
        
        **🅿️ Parking**
        - Parking A: Short-term
        - Parking B: Long-term
        - Economy Parking
        - Total: 30,000 spaces
        """)

    with col2:
        st.markdown("""
        **🔴 Secure Zone (Airside)**
        - Restricted access
        - Badge required
        - Continuous monitoring
        - Security perimeter
        
        **🛫 Airfield Side**
        - Taxiways
        - Aprons
        - De-icing areas
        - Holding points
        """)

    with col3:
        st.markdown("""
        **🔧 Technical Zones**
        - Maintenance hangars
        - Specialized workshops
        - Fuel facilities
        - Equipment storage
        
        **📦 Freight & Cargo**
        - Cargo area
        - Warehouses
        - Customs
        - Handling operations
        """)

    st.markdown("---")

    # Capacity and traffic flow
    st.markdown("### 📊 Capacity and Traffic Flow")

    col1, col2 = st.columns(2)

    with col1:
        capacity_data = pd.DataFrame({
            'Zone': ['Terminal 1', 'Terminal 2', 'Cargo', 'General Aviation', 'Maintenance'],
            'Max Capacity/Day': [150, 120, 50, 30, 15],
            'Average Occupancy (%)': [75, 68, 85, 45, 90],
            'Staff': [850, 650, 200, 50, 120]
        })
        
        st.dataframe(capacity_data, use_container_width=True, hide_index=True)

    with col2:
        # Occupancy chart
        fig_capacity = px.bar(
            capacity_data,
            x='Zone',
            y='Average Occupancy (%)',
            title='Occupancy Rate by Zone',
            color='Average Occupancy (%)',
            color_continuous_scale='RdYlGn_r'
        )
        
        fig_capacity.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font=dict(color='#000000', size=11)
        )
        
        st.plotly_chart(fig_capacity, use_container_width=True)

    st.markdown("---")

    # Access points and circulation
    st.markdown("### 🚗 Access and Circulation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🚪 Vehicle Access Points**
        
        1. **Main Entrance (North Access)**
        - Interstate 70 Exit 239
        - Access to Terminals 1 & 2
        - Curbside drop-off
        
        2. **South Entrance (South Access)**
        - Natural Bridge Road
        - Economy Parking access
        - Cargo & Freight
        
        3. **Employee Entrance**
        - McDonnell Blvd
        - Badge required
        - Staff parking
        """)

    with col2:
        st.markdown("""
        **🚌 Public Transportation**
        
        - **MetroLink Red Line**
        - Station: Lambert Airport Terminal 1
        - Frequency: 15–20 min
        - Service: 5am–11pm
        
        - **Metro Transit Bus**
        - Routes: 11, 20, 95X
        - Multiple connections
        
        - **Hotel Shuttles**
        - Dedicated zone Terminal 1
        - 24/7 service
        """)

    st.info("💡 **Navigation**: Use the schematic map above to locate zones. Hover over elements for more details.")


# ============================================================================
# TAB 6: CHATBOT OCC
# ============================================================================
with tab6:
    st.subheader("💬 Intelligent OCC Assistant")
    st.markdown("Ask questions about predictions, operational risks or aviation in general.")
    
    # API configuration section
    with st.expander("⚙️ OpenRouter API Configuration", expanded=False):
        api_key_input = st.text_input(
            "OpenRouter API Key", 
            type="password",
            help="Get your key at https://openrouter.ai/keys"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox(
                "Model",
                ["openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
                help="GPT-4o-mini recommended for best value"
            )
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, 
                                   help="Higher = more creative")
        
        if st.button("💾 Save Configuration"):
            if api_key_input:
                st.session_state.chatbot = OCCChatBot(
                    api_key=api_key_input,
                    model=model_choice,
                    temperature=temperature
                )
                # Update context if available
                if st.session_state.prediction_context:
                    st.session_state.chatbot.set_context(st.session_state.prediction_context)
                st.success("✅ ChatBot configured and ready!")
            else:
                st.error("❌ Please enter an API key")
    
    st.markdown("---")
    
    # Check chatbot status
    if st.session_state.chatbot is None:
        st.info("👆 **First configure your OpenRouter API key above**")
        st.markdown("""
        **Suggested questions:**
        - What are the main risk factors for this flight?
        - What does a congestion score of 0.8 mean?
        - What action do you recommend for a high-risk flight?
        - Explain the impact of weather on delays
        - How to interpret a high DelayRiskIndex?
        """)
    else:
        # Display available context
        if st.session_state.prediction_context:
            with st.expander("📊 Current Operational Context", expanded=False):
                st.json(st.session_state.prediction_context)
        else:
            st.warning("⚠️ No prediction made. Go to 'Prediction' tab to generate context.")
        
        # Chat area
        st.markdown("### 💭 Conversation")
        
        # Display history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 OCC Assistant:</strong><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Input area
        col1, col2 = st.columns([5, 1])
        with col1:
            user_question = st.text_input(
                "Your question:",
                key="user_input",
                placeholder="e.g., What are the main risks for this flight?"
            )
        with col2:
            send_button = st.button("📤 Send", use_container_width=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Reset chat"):
                st.session_state.chatbot.reset()
                st.session_state.chat_history = []
                st.success("✅ Conversation reset")
                st.rerun()
        
        with col2:
            include_context = st.checkbox("Include context", value=True,
                                        help="Disable for general questions")
        
        with col3:
            if st.button("📥 Export conversation"):
                if st.session_state.chat_history:
                    export_text = ""
                    for msg in st.session_state.chat_history:
                        role_label = "👤 You" if msg['role'] == 'user' else "🤖 Assistant"
                        export_text += f"{role_label}:\n{msg['content']}\n\n{'='*50}\n\n"
                    
                    st.download_button(
                        label="💾 Download",
                        data=export_text,
                        file_name=f"chat_occ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        # Process question
        if send_button and user_question:
            with st.spinner("🤔 Thinking..."):
                # Add to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_question
                })
                
                # Get answer
                answer = st.session_state.chatbot.ask(user_question, include_context=include_context)
                
                # Add answer to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': answer
                })
                
                # Reload to display new messages
                st.rerun()
        
        # Suggested questions
        st.markdown("---")
        st.markdown("### 💡 Suggested Questions")
        
        suggestions = [
            "Explain the main risk factors",
            "What is the delay probability for this flight?",
            "How to improve STL flight punctuality?",
            "What to do in case of high airport congestion?",
            "Current weather impact on operations"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💬 {suggestion}", key=f"sug_{i}"):
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': suggestion
                    })
                    
                    with st.spinner("🤔 Thinking..."):
                        answer = st.session_state.chatbot.ask(suggestion, include_context=include_context)
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': answer
                        })
                    st.rerun()

# ============================================================================
# TAB 7: GUIDE
# ============================================================================
with tab7:
    st.subheader("📚 OCC ChatBot User Guide")
    
    st.markdown("""
    ### 🎯 Features
    
    The OCC ChatBot is an intelligent assistant designed to help you in your daily operations:
    
    #### 1️⃣ **Contextual Analysis**
    - Interprets risk predictions for each flight
    - Explains contributing factors (weather, congestion, rotation)
    - Suggests tailored corrective actions
    
    #### 2️⃣ **General Questions**
    - Answers questions about aviation and airport operations
    - Works even without prediction context
    - Covers a wide spectrum of operational topics
    
    #### 3️⃣ **History and Memory**
    - Maintains conversation context
    - Remembers previous exchanges
    - Enables natural and coherent dialogue
    
    ---
    
    ### 🚀 How to Use
    
    **Step 1: Configuration**
    1. Get an API key from [OpenRouter](https://openrouter.ai/keys)
    2. Enter the key in the "API Configuration" section
    3. Choose your model (GPT-4o-mini recommended)
    4. Click "Save Configuration"
    
    **Step 2: Generate Context** *(optional)*
    1. Go to "Prediction" tab
    2. Select a flight
    3. Context will be automatically saved
    
    **Step 3: Ask Your Questions**
    - Type your question in the text box
    - Enable "Include context" for personalized answers
    - Disable it for general questions
    
    ---
    
    ### 💡 Question Examples
    
    **With Context (Active Prediction):**
    - "Why does this flight have high risk?"
    - "What actions do you recommend to reduce risk?"
    - "Is a congestion score of 0.85 concerning?"
    - "How does current weather impact this flight?"
    
    **Without Context (General Questions):**
    - "What is a short turnaround and why is it risky?"
    - "How does airport slot management work?"
    - "What are the main KPIs of an OCC?"
    - "Explain typical causes of airline delays"
    
    ---
    
    ### ⚙️ Advanced Options
    
    **Temperature (0.0 - 1.0)**
    - **Low (0.0-0.3)**: Factual and precise answers
    - **Medium (0.4-0.7)**: Balance creativity/precision *(recommended)*
    - **High (0.8-1.0)**: More creative and varied answers
    
    **Model Choice**
    - **GPT-4o-mini**: Fast, economical, excellent for most cases
    - **GPT-4o**: More powerful, better complex reasoning
    - **Claude-3.5-Sonnet**: High-performance alternative, different style
    
    ---
    
    ### 🔒 Security & Privacy
    
    - Your API keys are stored only in your session
    - No data is saved server-side
    - Conversations are not persisted
    - Use "Reset chat" to clear history
    
    ---
    
    ### ⚠️ Limitations
    
    - Chatbot requires internet connection
    - Answers depend on API key quality
    - Context limited to 800 tokens by default
    - Use OpenRouter moderately (API costs)
    
    ---
    
    ### 🆘 Troubleshooting
    
    **Error "❌ API Error"**
    - Check your API key
    - Ensure you have credit on OpenRouter
    - Try another model
    
    **Slow responses**
    - Normal for GPT-4o (more complex)
    - Use GPT-4o-mini for more speed
    
    **Context not considered**
    - Check "Include context" is enabled
    - First make a prediction in tab 4
    
    ---
    
    ### 📞 Support
    
    For any questions or technical issues, contact the OCC technical team.
    """)
    
    # Usage statistics (if available)
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📊 Session Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_msgs = len(st.session_state.chat_history)
            st.metric("Total Messages", total_msgs)
        
        with col2:
            user_msgs = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
            st.metric("Your Questions", user_msgs)
        
        with col3:
            bot_msgs = len([m for m in st.session_state.chat_history if m['role'] == 'assistant'])
            st.metric("Bot Responses", bot_msgs)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #1f2937; padding: 1rem;'>
    <p><strong>OCC STL Dashboard v2.0</strong> | Powered by XGBoost, Streamlit & OpenRouter</p>
    <p style='font-size: 0.9em;'>OCC ChatBot integrated - Intelligent assistant for air operations</p>
</div>
""", unsafe_allow_html=True)
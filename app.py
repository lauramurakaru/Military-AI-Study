import streamlit as st
import pandas as pd
import joblib
import random
import os
import logging
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Initialize Logging Early
logging.basicConfig(level=logging.INFO)

# Page Layout Configuration
PAGE_CONFIG = {
    "layout": "centered",
    "page_title": "Military Decision-Making App",
    "page_icon": "⚔️",
    "initial_sidebar_state": "collapsed",
     "menu_items": {
        "Get Help": None,
        "Report a bug": None,
        "About": None
    }
}

st.set_page_config(**PAGE_CONFIG)

# Initialize session state variables
session_vars = [
    "step", "scenario", "user_decision", "model_prediction_label",
    "override_reason", "confirmation_feedback", "feedback_shared",
    "progress", "start_time", "decision_time",
    "submitted_decision", "submitted_feedback",
    "scenario_generated", "model_generated", "revealed_reasoning",
    "raw_model_prediction"
]

# Initialize all session state variables
for var in session_vars:
    if var not in st.session_state:
        if var == "step":
            st.session_state[var] = 1
        elif var in ["scenario_generated", "model_generated", "revealed_reasoning"]:
            st.session_state[var] = False
        else:
            st.session_state[var] = None

# Initialize time
if "time_remaining" not in st.session_state:
    st.session_state.time_remaining = 300
if "timer_active" not in st.session_state:
    st.session_state.timer_active = False
if "start" not in st.session_state or st.session_state.start is None:
    st.session_state.start = time.time()



if "start" not in st.session_state:
    st.session_state.start = None

# Styles for Markdown Elements
MARKDOWN_STYLE = {
    "header": "<h1 style='font-size: 25px; line-height: 1; text-align: center; color: #003366;'>",
    "subheader": "<h2 style='font-size: 23px; line-height: 1; color: #003366;'>",
    "normal_text": "<p style='font-size: 18px; line-height: 1;'>",
    "highlighted_text": "<p style='font-size: 18px; line-height: 1; color: #CC0000; font-weight: bold;'>",
    "decision_text": "<p style='font-size: 21px; line-height: 1; color: #003366; font-weight: bold;'>"
}

# Define columns to shuffle and score columns
columns_to_shuffle = [
    ['Target_Category', 'Target_Category_Score'],
    ['Target_Vulnerability', 'Target_Vulnerability_Score'],
    ['Terrain_Type', 'Terrain_Type_Score'],
    ['Civilian_Presence', 'Civilian_Presence_Score'],
    ['Damage_Assessment', 'Damage_Assessment_Score'],
    ['Time_Sensitivity', 'Time_Sensitivity_Score'],
    ['Weaponeering', 'Weaponeering_Score'],
    ['Friendly_Fire', 'Friendly_Fire_Score'],
    ['Politically_Sensitive', 'Politically_Sensitive_Score'],
    ['Legal_Advice', 'Legal_Advice_Score'],
    ['Ethical_Concerns', 'Ethical_Concerns_Score'],
    ['Collateral_Damage_Potential', 'Collateral_Damage_Potential_Score'],
    ['AI_Distinction (%)', 'AI_Distinction (%)_Score'],
    ['AI_Proportionality (%)', 'AI_Proportionality (%)_Score'],
    ['AI_Military_Necessity', 'AI_Military_Necessity_Score'],
    ['Human_Distinction (%)', 'Human_Distinction (%)_Score'],
    ['Human_Proportionality (%)', 'Human_Proportionality (%)_Score'],
    ['Human_Military_Necessity', 'Human_Military_Necessity_Score']
]

# Define score_columns based on columns_to_shuffle
score_columns = [pair[1] for pair in columns_to_shuffle]

# Label mapping for model predictions
label_mapping = {
    0: 'Do Not Engage',
    1: 'Ask Authorization',
    2: 'Do Not Know',
    3: 'Engage'
}

# Load the Model and Data
try:
    model_path = 'MDMP_model.joblib'
    features_path = 'MDMP_feature_columns.joblib'
    csv_path = 'dataset_with_all_category_scores.csv'
    
    rf_model_loaded = joblib.load(model_path)
    trained_feature_columns = joblib.load(features_path)
    df = pd.read_csv(csv_path)
    
    # Debug prints
    print("Loaded data columns:", df.columns.tolist())
    print("Trained feature columns:", trained_feature_columns)
    
    logging.info("Model and data loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    logging.error(f"Error loading model or data: {e}")
    st.stop()

# Utility Functions
def get_markdown_text(text, style_key):
    """Format text with predefined markdown styles."""
    tag_mapping = {
        "header": "h1",
        "subheader": "h2",
        "normal_text": "p",
        "highlighted_text": "p",
        "decision_text": "p"
    }
    tag = tag_mapping.get(style_key, "p")
    style = MARKDOWN_STYLE.get(style_key, MARKDOWN_STYLE['normal_text'])
    if style.endswith('>'):
        style = style[:-1]
    return f"{style}>{text}</{tag}>"

def convert_civilian_presence(value):
    """Convert civilian presence values to consistent format."""
    if isinstance(value, str) and '-' in value:
        return value
    try:
        return str(int(value))
    except ValueError:
        return "0"

def shuffle_dataset(df):
    """Shuffle the dataset while maintaining related columns together."""
    print("Original columns:", df.columns.tolist())  # Debug print
    df_shuffled = df.copy()
    
    for related_columns in columns_to_shuffle:
        print(f"Processing columns: {related_columns}")  # Debug print
        shuffled_subset = df[related_columns].sample(
            frac=1, random_state=random.randint(0, 10000)
        ).reset_index(drop=True)
        df_shuffled[related_columns] = shuffled_subset
    
    df_shuffled['Total_Score'] = df_shuffled[score_columns].sum(axis=1)
    print("Final columns:", df_shuffled.columns.tolist())  # Debug print
    return df_shuffled

def get_random_scenario(df):
    """Get a random scenario from the dataset."""
    random_index = random.randint(0, len(df) - 1)
    return df.iloc[random_index]

def calculate_percentages(scores):
    """Calculate percentages based on absolute scores."""
    abs_scores = {k: abs(v) for k, v in scores.items()}
    total_abs = sum(abs_scores.values())
    
    if total_abs == 0:
        return {k: 0 for k in scores}
    
    raw_percentages = {k: (v / total_abs) * 100 for k, v in abs_scores.items()}
    rounded_percentages = {}
    total_rounded = 0
    items = list(raw_percentages.items())
    
    for k, v in items[:-1]:
        rounded = round(v, 2)
        rounded_percentages[k] = rounded
        total_rounded += rounded
    
    last_key = items[-1][0]
    rounded_percentages[last_key] = round(100 - total_rounded, 2)
    
    signed_percentages = {
        key: pct if scores[key] >= 0 else -pct 
        for key, pct in rounded_percentages.items()
    }
    
    return signed_percentages

def get_score_display(score, percentage):
    """Format score display with color coding."""
    if score > 0:
        color = "#28a745"  # Green for positive
    elif score < 0:
        color = "#dc3545"  # Red for negative
    else:
        color = "#6c757d"  # Gray for neutral
    return f"<b>{score}</b> (<span style='color:{color}'>{percentage:.2f}%</span>)"

def assign_final_decision(total_score):
    """Assign final decision based on total score thresholds."""
    if total_score >= 30:
        return 'Engage'
    elif total_score >= 22.5:
        return 'Ask Authorization'
    elif total_score >= 15:
        return 'Do Not Know'
    else:
        return 'Do Not Engage'

def verify_scenario_data(scenario):
    """Verify that all required columns are present in the scenario data."""
    required_columns = [col[0] for col in columns_to_shuffle]
    
    # If data is in the index, it's valid - return empty list (no missing columns)
    if isinstance(scenario, pd.Series) or any(col in scenario.index for col in required_columns):
        return []
    
    # For DataFrame, check columns
    if isinstance(scenario, pd.DataFrame):
        missing_columns = [col for col in required_columns if col not in scenario.columns]
        return missing_columns
        
    return []  # Default case - assume data is valid

def get_final_prediction(scenario_df, model):
    """Generate final prediction following the sequence."""
    try:
        # 1. Calculate total score
        if 'Total_Score' not in scenario_df.columns or pd.isna(scenario_df['Total_Score']).all():
            scenario_df['Total_Score'] = scenario_df[score_columns].sum(axis=1)
        total_score = scenario_df['Total_Score'].iloc[0]
        
        # 2. Check override rules first
        override_decision, override_reason = apply_override_rules(scenario_df.iloc[0])
        
        # 3. Get model's raw prediction for reference
        try:
            model_pred = model.predict(scenario_df)[0]
            model_label = label_mapping.get(model_pred, "Unknown")
        except Exception as e:
            logging.error(f"Error in model prediction: {e}")
            model_label = None
        
        # 4. Make final decision
        if override_decision:
            # Only include OVERRIDE APPLIED prefix for actual overrides
            return override_decision, f"OVERRIDE APPLIED: {override_reason}", model_label
        else:
            # For score-based decisions, don't include any prefix
            score_based_decision = assign_final_decision(total_score)
            return score_based_decision, "", model_label  # Empty string for reason when score-based
            
    except Exception as e:
        logging.error(f"Error in get_final_prediction: {e}")
        return None, f"Error in prediction: {e}", None

def apply_override_rules(row):
    """Apply override rules to determine final decision."""
    try:
        # Get both category and score columns from the row
        categories = {col: row[col] for col in row.index if not col.endswith('_Score')}
        
        # Ensure Total_Score is calculated
        if 'Total_Score' not in row or pd.isna(row['Total_Score']):
            row['Total_Score'] = row[[col for col in row.index if col.endswith('_Score')]].sum()
            logging.info("Calculated Total_Score in apply_override_rules")

        # RULE 1: DO NOT ENGAGE OVERRIDES
        if categories['Target_Category'] in ["Chapel", "Medical Installation", "Medical Vehicle"]:
            return "Do Not Engage", f"Protected Target_Category '{categories['Target_Category']}'"
        
        if (categories['Terrain_Type'] in ["Urban Center", "Residential Area"]) and \
           (categories['Target_Category'] not in ["High-Value Target", "Battalion HQ", "Battlegroup HQ", "Brigade HQ", "Division HQ"]):
            return "Do Not Engage", f"Non-priority target in {categories['Terrain_Type']}"
        
        if categories['Ethical_Concerns'] == 'Immoral' and row['Total_Score'] >= 30:
            return "Do Not Engage", "Ethical concerns override high score"
        
        try:
            civilian_presence = categories['Civilian_Presence']
            if isinstance(civilian_presence, str) and '-' in civilian_presence:
                civilian_presence = int(civilian_presence.split('-')[0])
            else:
                civilian_presence = int(civilian_presence)
            
            if civilian_presence >= 100:
                return "Do Not Engage", f"High civilian presence: {categories['Civilian_Presence']}"
        except (ValueError, TypeError):
            logging.warning(f"Could not parse Civilian_Presence value: {categories['Civilian_Presence']}")
        
        if categories['Collateral_Damage_Potential'] == 'Very_High' and \
           any(str(categories['Civilian_Presence']).startswith(x) for x in ['50', '100']):
            return "Do Not Engage", "High collateral damage risk with significant civilian presence"
        
        if categories['Friendly_Fire'] == "Very_High" and categories['Collateral_Damage_Potential'] == 'Very_High':
            return "Do Not Engage", "Multiple high-risk factors present"

        # RULE 2: ASK AUTHORIZATION OVERRIDES
        try:
            cp_value = categories['Civilian_Presence']
            if isinstance(cp_value, str) and '-' in cp_value:
                cp_value = int(cp_value.split('-')[0])
            else:
                cp_value = int(cp_value)
                
            if cp_value > 30 and categories['Weaponeering'] in ["Incendiary Weapon", "Thermobaric Munition", "White Phosphorus Bomb"]:
                return "Ask Authorization", "Special weapons with civilian presence > 30"
        except (ValueError, TypeError):
            logging.warning(f"Could not parse Civilian_Presence value for special weapons check")

        if categories['Legal_Advice'] in ['It depends', 'Questionable'] or \
           (categories['Ethical_Concerns'] == 'Immoral' and cp_value > 50):
            return "Ask Authorization", "Legal/ethical concerns require authorization"
        
        if categories['Politically_Sensitive'] == "High" and categories['Terrain_Type'] == "Critical Infrastructure Area":
            return "Ask Authorization", "Sensitive infrastructure engagement"

        # RULE 3: DO NOT KNOW OVERRIDES
        if categories['Weaponeering'] == "Anti-Personnel Mine" and \
           categories['Target_Category'] in ["Fighter Aircraft", "Frigate", "Ship Maintenance Facility", "Naval Base"]:
            return "Do Not Know", "Inappropriate weapon for target type"
        
        if categories['Weaponeering'] == "Torpedo" and \
           categories['Target_Category'] not in ["Ship Maintenance Facility", "Naval Base", "Frigate"]:
            return "Do Not Know", "Torpedo inappropriate for non-naval target"

        return None, "No override rules applied"

    except KeyError as e:
        logging.error(f"Missing required column in override rules: {e}")
        return None, "No override rules applied"  # Return None instead of error
    except Exception as e:
        logging.error(f"Unexpected error in apply_override_rules: {e}")
        return None, "No override rules applied"  # Return None instead of error

def get_google_sheet():
    """Connect to Google Sheets."""
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            'https://www.googleapis.com/auth/spreadsheets',
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            st.secrets["gcp_service_account"], scope
        )
        client = gspread.authorize(creds)
        sheet = client.open("Study_data").sheet1
        return sheet
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        logging.error(f"Error connecting to Google Sheets: {e}")
        return None

def save_data_to_google_sheet(data):
    """Save data to Google Sheets."""
    sheet = get_google_sheet()
    if sheet:
        try:
            # Include scenario categories and values as a single string
            scenario_details = ", ".join(
                f"{key}: {value}" for key, value in data.get('scenario', {}).items()
            )
            
            # Prepare the row data
            row = [
                scenario_details,  # Column 1: Scenario categories and values
                data.get('Participant Decision', ''),  # Column 2
                data.get('Model Prediction', ''),  # Column 3
                data.get('Decision Time (seconds)', ''),  # Column 4
                data.get('Confirmation Feedback', ''),  # Column 5
                data.get('Additional Feedback', ''),  # Column 6
            ]
            
            # Append the row to the Google Sheet
            sheet.append_row(row)
            logging.info("Data appended to Google Sheets successfully.")
        except Exception as e:
            st.error(f"Error saving data to Google Sheets: {e}")
            logging.error(f"Error saving data to Google Sheets: {e}")

def display_scenario_with_scores(scenario, feature_importances=None, override_reason=None):
    """Display the scenario with scores and analysis."""
    columns_to_display = [col[0] for col in columns_to_shuffle]

    # For Steps 2-5, show clean scenario display
    if st.session_state.step < 6:
        for column in columns_to_display:
            value = scenario[column] if column in scenario and pd.notna(scenario[column]) else "Unknown"
            st.markdown(f"""
                <div style='font-size: 16px; margin-bottom: 1px;'>
                    <b>{column}</b>: {value}
                </div>
            """, unsafe_allow_html=True)
    
    # For Step 6, show integrated display with scores and percentages
    else:
        scores = {f"{col}_Score": scenario[f"{col}_Score"] 
                 for col in columns_to_display if f"{col}_Score" in scenario}
        percentages = calculate_percentages(scores)

        # Then display parameters with scores
        for score_col, score_val in scores.items():
            pct = percentages.get(score_col, 0)
            parameter = score_col.replace('_Score', '')
            score_display = get_score_display(score_val, pct)
            
            st.markdown(f"""
                <div style='display: flex; justify-content: flex-start; align-items: center; margin-bottom: 2px;'>
                    <span style='font-weight: bold; margin-right: 5px; font-size: 20px;'>{parameter}:</span>
                    <span style='margin-right: 5px; font-size: 20px;'>{scenario[parameter]}</span>
                    <span style='font-size: 20px;'><b>{score_val}</b> ({pct:.2f}%)</span>
            </div>
            <div class='dotted-line'></div>
        """, unsafe_allow_html=True)

        # Display total score at the bottom
        total_score = sum(scores.values())
        st.markdown(f"""
            <div style='margin-top: 15px; color: #CC0000; font-weight: bold;'>
                Total Score: {total_score}
            </div>
        """, unsafe_allow_html=True)

# Navigation Functions
def next_step():
    """Move to the next step."""
    total_steps = 9
    if st.session_state.step < total_steps:
        st.session_state.step += 1
        logging.info(f"Moved to Step {st.session_state.step}")

def prev_step():
    """Move to the previous step."""
    if st.session_state.step > 1:
        st.session_state.step -= 1
        st.session_state.timer_active = False
        st.session_state.time_remaining = 300
        logging.info(f"Moved back to Step {st.session_state.step}")

# Feedback Handling Functions
def handle_submit_feedback():
    """Handle submission of user feedback."""
    feedback = st.session_state.get('feedback_box', '').strip()
    if feedback == "":
        st.warning("Please provide feedback before submitting.")
    else:
        data = {
            "scenario": st.session_state.scenario,
            "Participant Decision": st.session_state.user_decision,
            "Model Prediction": st.session_state.model_prediction_label,
            "Decision Time (seconds)": round(st.session_state.decision_time),
            "Confirmation Feedback": st.session_state.confirmation_feedback,
            "Additional Feedback": feedback
        }
        save_data_to_google_sheet(data)
        st.success("Your responses have been recorded. Thank you!")
        logging.info("Data saved successfully.")
        next_step()


def handle_timeout_decision():
    """Handle case when time runs out without a decision."""

    st.session_state.user_decision = "No Decision - Time Expired"
    st.session_state.decision_time = 300

    return {
        'Participant Decision': "No Decision - Time Expired",
        'Model Prediction': st.session_state.model_prediction_label,
        'Override Reason': st.session_state.override_reason,
        'Confirmation Feedback': "N/A - Timeout",
        'Additional Feedback': "Participant did not complete decision within time limit",
        'Decision Time (seconds)': 300
    }
def handle_skip_feedback():
    """Handle skipping the feedback section."""

    feedback_text = st.session_state.get("feedback_box", "")
    data = {
        "scenario": st.session_state.scenario,
        "Participant Decision": st.session_state.user_decision,
        "Model Prediction": st.session_state.model_prediction_label,
        "Decision Time (seconds)": round(st.session_state.decision_time),
        "Confirmation Feedback": st.session_state.confirmation_feedback,
        "Additional Feedback": feedback_text
    }
    save_data_to_google_sheet(data)
    st.success("Your responses have been recorded. Thank you!")
    logging.info("Data saved successfully.")
    next_step()

def main():
    """Main application function."""
    # Add custom CSS
    st.markdown("""
        <style>
            .step-title {
                font-size: 20px;
                font-weight: bold;
                color: #003366;
                margin-bottom: 2px;  /* Reduce spacing below the title */
            }
            .scenario-guide {
                margin-top: -15px;  /* Reduce spacing above Scenario Guide */
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Increase spacing around st.button */
    .stButton > button {
        margin-top: 1rem;  /* top spacing */
        margin-bottom: 1rem;  /* bottom spacing */
    }
    </style>
    """, unsafe_allow_html=True)


    st.markdown("""
        <style>
            /* Hide Streamlit and GitHub elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .reportview-container .main footer {visibility: hidden;}
            iframe {display: none;}
            div[data-testid="stDecoration"] {display: none;}
            .element-container iframe {display: none;}
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
            }
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
            }
            ::-webkit-scrollbar-thumb {
                background: #888;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
    
    /* Header */
    .main-header {
        font-size: 32px;
        color: #003366;
        text-align: center;
        margin-top: -50px;
        padding: 10px 0;
    }
    
    /* Cards */
    .card {
        background-color: #F0F8FF;
        padding: 5px;
        border-radius: 10px;
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .card h3 {
        color: #003366;
        margin-top: 0;
        line-height: 1.2;
        font-size: 20px;
        font-weight: bold;
    }
    
    .card p {
        font-size: 16px;
        margin: 2px 0;
        line-height: 1;
    }
    
    .card-right {
        background-color: #F0F8FF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
        margin-top: 0;
    }
    
    /* Time Remaining Display */
    .time-remaining {
        font-size: 16px;
        color: #003366;
        font-weight: bold;
        text-align: center;
        padding: 5px;
        background-color: #F0F8FF;
        border-radius: 5px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton button {
        background-color: #003366;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    
    .stButton button:hover {
        background-color: #002244;
    }
    
    /* Radio buttons */
    .stRadio label {
        font-size: 15px;
        padding: 1px 0;
   line-height: 0.5; 
    }

  /* Add this to further reduce space between radio options */
  .stRadio > div {
    gap: 1px !important;  # Reduces space between radio options
  }

    /* Progress bar */
    .stProgress .st-ba {
        background-color: #003366;
    }
    
    /* Decision text styling */
    .decision-text {
        font-size: 16px;
        color: #003366;
        font-weight: bold;
        margin: 15px 0;
        padding: 10px;
        background-color: #F0F8FF;
        border-radius: 5px;
    }
    
    /* Score display */
    .score-text {
        font-size: 18px;
        line-height: 1;
        margin: 4px 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 16px;
        color: #003366;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(get_markdown_text("Military Decision-Making App", "header"), unsafe_allow_html=True)
    logging.info("App started.")

    # Progress Indicator
    total_steps = 9
    st.session_state.progress = (st.session_state.step - 1) / (total_steps - 1)
    st.progress(st.session_state.progress)
    logging.info(f"Progress: {st.session_state.progress}, Step: {st.session_state.step}")

# Step 1: Introduction and Scenario Guide
    if st.session_state.step == 1:
        logging.info("Entered Step 1: Introduction and Scenario Guide.")
        st.markdown("<div class='step-title'>Step 1: Introduction</div>", unsafe_allow_html=True)
        
        st.markdown(get_markdown_text("""
        
The App explores AI-augmented decision making and human-machine teaming in military contexts.
*Review the instructions below for scenario parameters, steps, and tutorials.*
        """, "normal_text"), unsafe_allow_html=True)

        st.markdown("""
            <details>
            <summary><strong>Scenario Guide</strong></summary>
            <p>Please familiarize yourself with the parameters used in the scenarios:</p>
            <ol>
            <li><strong>Target_Category</strong>: The exact object or multiple targets to be engaged. Examples include <strong>Brigade HQ</strong>, <strong>Artillery Unit</strong>, <strong>Unmanned Aerial Vehicle</strong>, etc.</li>
            <li><strong>Target_Vulnerability</strong>: Represents the susceptibility of the target to damage from an attack. Values range from <strong>Very Low</strong> to <strong>Very High</strong>.</li>
            <li><strong>Terrain_Type</strong>: The area in which the target is present. Examples include <strong>Transportation Hub</strong>, <strong>Electric Power Grid Network</strong>, <strong>Residential Area</strong>, etc.</li>
            <li><strong>Civilian_Presence</strong>: An approximate estimate of civilians in the area. Values include ranges like <strong>0</strong>, <strong>11-29</strong>, <strong>50-99</strong>, etc.</li>
            <li><strong>Damage_Assessment</strong>: Indicates the expected ease and productivity of the Battle Damage Assessment (BDA) process after the attack. Values range from <strong>Very Low</strong> to <strong>Very High</strong>.</li>
            <li><strong>Time_Sensitivity</strong>: Urgency of action required. Values include <strong>High</strong>, <strong>Immediate</strong>, <strong>Normal</strong>.</li>
            <li><strong>Weaponeering</strong>: The type of weapon or asset available for engagement. Examples include <strong>Precision Guided Munition</strong>, <strong>155mm Artillery</strong>, <strong>SOF Unit</strong>, etc.</li>
            <li><strong>Friendly_Fire</strong>: Risk of friendly fire incidents. Values range from <strong>Very Low</strong> to <strong>Very High</strong>.</li>
            <li><strong>Politically_Sensitive</strong>: Indicates the level of political tension and strategic considerations regarding the use of force. Values include <strong>Low</strong>, <strong>Medium</strong>, <strong>High</strong>.</li>
            <li><strong>Legal_Advice</strong>: Legal interpretations that may affect the decision. Values include <strong>Lawful</strong>, <strong>Questionable</strong>, <strong>It depends</strong>, etc.</li>
            <li><strong>Ethical_Concerns</strong>: How the use of force reflects moral values and beliefs about right and wrong. Values include <strong>Unlikely</strong>, <strong>Immoral</strong>, <strong>No</strong>, etc.</li>
            <li><strong>Collateral_Damage_Potential</strong>: Potential for unintended damage. Values range from <strong>Very Low</strong> to <strong>Very High</strong>.</li>
            <li><strong>AI_Distinction (%)</strong>: AI-driven system's estimation of Positive Identification (PID) of a target, on a scale of <strong>1-100%</strong>.</li>
            <li><strong>AI_Proportionality (%)</strong>: AI-driven system's estimation of proportionality, on a scale of <strong>1-100%</strong>.</li>
            <li><strong>AI_Military_Necessity</strong>: Whether the model assesses the action as necessary for achieving military objectives. Values include <strong>Yes</strong>, <strong>Open to Debate</strong>.</li>
            <li><strong>Human_Distinction (%)</strong>: Human estimation of PID of a target, based on sensor data or direct observation, ranging from <strong>30-100%</strong>.</li>
            <li><strong>Human_Proportionality (%)</strong>: Human estimation of proportionality, based on sensor data or direct observation, ranging from <strong>30-100%</strong>.</li>
            <li><strong>Human_Military_Necessity</strong>: Human assessment of whether the action is necessary for achieving military objectives. Values include <strong>Yes</strong>, <strong>Open to Debate</strong>.</li>
            </ol>
            </details>
            """, unsafe_allow_html=True)
        st.markdown("""
            <details>
            <summary><strong>Background</strong></summary>
            <ol>
            <li>As the commander of an infantry unit, your mission is to secure an object and protect it from potential destruction caused by enemy action.</li>
            <li>Higher command will provide you with intelligence and resources to influence targets and achieve desired effects within your area of responsibility.</li>
            <li>You will engage in 10 scenarios designed to rehearse decision-making with the assistance of an AI-driven model.</li>
            <li>The information presented in each scenario may be conflicting, requiring you to carefully evaluate its reliability.</li>
            <li>It is your responsibility to decide whether to trust the model's recommendations or rely on your own judgment.</li>
            </ol>
            </details>
            """, unsafe_allow_html=True)
        st.markdown("""
            <details>
            <summary><strong>Steps</strong></summary>
            <ul>
            <li>Step 1: Introduction</li>
            <li>Step 2: Generate Scenario</li>
            <li>Step 3: Review Scenario</li>
            <li>Step 4: Submit Decision</li>
            <li>Step 5: Generate Model Prediction</li>
            <li>Step 6: Reveal Model Reasoning</li>
            <li>Step 7: Provide Confirmation Feedback</li>
            <li>Step 8: Share Additional Feedback</li>
            </ul>
            <p><strong>Note:</strong></p>
            <ul>
            <li>Each scenario's parameters are randomized, which may lead to contradictory data. However, it is important to make decisions based on the available data in the given context and justify your judgment by providing feedback, if necessary.</li>
            <li>A 5-minute timer is provided for submitting decisions, intended solely for research purposes.</li>
            <li>In the 10-scenario loop, odd-numbered scenarios allow you to submit decisions before the model's prediction. In even-numbered scenarios, the model predicts first. This alternating sequence aims to calibrate trust in the AI model for research purposes.</li>
            </ul>
            </details>
            """, unsafe_allow_html=True)
        st.markdown("""
            <details>
            <summary><strong>Getting Started</strong></summary>
            <ol>
                <li>Refer to the "Scenario Guide" if necessary to refresh your understanding of parameter definitions.</li>
                <li>Take time to review scenario details.</li>
                <li>Submit your decision. If the timer expires, a decision will be auto-submitted.</li>
                <li>Next, you'll interact with a pre-trained AI model trained on hypothetical scenarios.</li>
            </ol>
            </details>
        """, unsafe_allow_html=True)

        st.button("Proceed to Scenario Generation", key="proceed_to_scenario_generation", on_click=next_step)

    # Step 2: Generate Scenario
    elif st.session_state.step == 2:
        logging.info("Entered Step 2: Generate Scenario.")
        st.markdown("<div class='step-title'>Step 2: Generate Scenario</div>", unsafe_allow_html=True)

        # Intro text (fixed closing </i>)
        st.markdown(get_markdown_text(
            "<i>Click the button below to generate a new scenario.</i>",
            "normal_text"
        ), unsafe_allow_html=True)
  
        
        generate_button = st.button("Generate Scenario", key="generate_scenario")
        if generate_button:
            try:
                logging.info("Starting scenario generation")
                st.session_state.df_shuffled = shuffle_dataset(df)
                logging.info("Dataset shuffled successfully")

                st.session_state.scenario = get_random_scenario(st.session_state.df_shuffled)
                logging.info("Random scenario selected successfully")
            
            # Ensure Total_Score is calculated
                if 'Total_Score' not in st.session_state.scenario or pd.isna(st.session_state.scenario['Total_Score']):
                    st.session_state.scenario['Total_Score'] = st.session_state.scenario[score_columns].sum()
                    logging.info("Calculated Total_Score for the scenario.")
            
                st.session_state.start_time = time.time()
                st.session_state.scenario_generated = True
                st.success("Scenario generated successfully!")
                logging.info("Generated new scenario.")
            except Exception as e:
                logging.error(f"Error in scenario generation: {e}")
                st.error(f"Failed to generate scenario: {e}")

        # Navigation Buttons
        col_back, col_next = st.columns(2)
        with col_back:
            st.button("Back", key="back_step2", on_click=prev_step)
        with col_next:
            if st.session_state.scenario_generated:
                st.button("Next", key="next_step2", on_click=next_step)
            else:
                st.button("Next", key="next_step2_disabled", on_click=next_step, disabled=True)

    # Step 3: Review Scenario
    elif st.session_state.step == 3:
        logging.info("Entered Step 3: Review Scenario.")
        st.markdown("<div class='step-title'>Step 3: Review Scenario</div>", unsafe_allow_html=True)
        display_scenario_with_scores(st.session_state.scenario)
        
        # Navigation Buttons
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Add space between the scenario and buttons
        col_back, col_next = st.columns(2)
        with col_back:
            st.button("Back", key="back_step3", on_click=prev_step)
        with col_next:
            st.button("Proceed to Decision Making", key="proceed_to_decision_step3", on_click=next_step)

    # Step 4: Submit Decision
    elif st.session_state.step == 4:
        logging.info("Entered Step 4: Submit Decision.")

    # Intro text (fixed closing </i>)
        st.markdown(get_markdown_text(
            "<i>Please review the scenario and select your decision below.</i>",
            "normal_text"
        ), unsafe_allow_html=True)

    # Initialize timer if not active
        if not st.session_state.timer_active:
            st.session_state.time_remaining = 300
            st.session_state.timer_active = True
            st.session_state.start = time.time()

    # Show the current countdown
        mins, secs = divmod(st.session_state.time_remaining, 60)
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 20px; font-weight: bold; color: #003366;">
                    Step 4: Submit Decision
                </div>
                <div style="font-size: 18px; color: #8B0000;">
                    Time remaining - {mins:02d}:{secs:02d}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Display the scenario
        display_scenario_with_scores(st.session_state.scenario)

    # Only allow radio selection if there's still time
        if st.session_state.time_remaining > 0:
            user_decision = st.radio(
                "",
                ["Engage", "Do Not Engage", "Ask Authorization", "Do Not Know"],
                key="decision",
                help="Select the most appropriate decision based on the scenario."
            )
            if user_decision:
                st.session_state.user_decision = user_decision
        else:
            st.warning("Time's up! No more decisions allowed.")

    # Navigation buttons
        col_back, col_submit = st.columns(2)
        with col_back:
            st.button("Back", key="back_step4", on_click=prev_step)
        with col_submit:
        # Only show the "Submit Decision" button if time > 0
            if st.session_state.time_remaining > 0:
                submit_decision = st.button("Submit Decision", key="submit_decision")
                if submit_decision:
                # Record how many seconds left when user submits
                    if isinstance(st.session_state.start, float):
                        st.session_state.decision_time = 300 - (time.time() - st.session_state.start)
                    else:
                        st.session_state.start = time.time()
                        st.session_state.decision_time = 300

                    st.session_state.submitted_decision = True
                    st.session_state.timer_active = False
                    st.success("Decision submitted successfully!")

    # If decision was submitted, show "Next" button
        if st.session_state.submitted_decision:
            st.button("Next", key="next_step4", on_click=next_step)

    # === MANUAL COUNTDOWN LOOP (runs last) ===
        if st.session_state.timer_active and st.session_state.time_remaining > 0:
        # Sleep for 1 second, then decrement time_remaining
            time.sleep(1)
            st.session_state.time_remaining -= 1

        # If timer just hit 0, auto-submit and go next
            if st.session_state.time_remaining == 0:
                if not st.session_state.submitted_decision:
                    data = handle_timeout_decision()
                    save_data_to_google_sheet(data)
                    st.warning("Time's up! Decision auto-submitted.")
                    st.session_state.submitted_decision = True
                    st.session_state.timer_active = False

                st.session_state.step += 1

            st.rerun()

    # Step 5: Generate Model Prediction
    elif st.session_state.step == 5:
        logging.info("Entered Step 5: Generate Model Prediction.")
        st.markdown("<div class='step-title'>Step 5: Generate Model Prediction</div>", unsafe_allow_html=True)
        st.write(get_markdown_text(f"<b>Your Decision</b>: {st.session_state.user_decision}", "decision_text"), unsafe_allow_html=True)
        
        generate_prediction = st.button("Generate Model Prediction", key="generate_prediction")
        
        if generate_prediction:
            try:
                # Create DataFrame with only the required score columns
                scenario_data = pd.DataFrame([{
                    col: st.session_state.scenario[col]
                    for col in trained_feature_columns
                }])
                
                # Get final prediction
                final_decision, reason, raw_model_pred = get_final_prediction(scenario_data, rf_model_loaded)
                
                if final_decision:
                    # Store results in session state
                    st.session_state.model_prediction_label = final_decision
                    st.session_state.override_reason = reason
                    st.session_state.raw_model_prediction = raw_model_pred
                    st.session_state.model_generated = True
                    
                    st.success("Model prediction generated!")
                    st.write(get_markdown_text(f"<b>Model Decision</b>: {final_decision}", "decision_text"), unsafe_allow_html=True)
                    
                    
                    logging.info(f"Model prediction generated - Final: {final_decision}, Reason: {reason}")
                else:
                    st.error("Could not generate prediction")
                    
            except Exception as e:
                st.error(f"An error occurred during model prediction: {e}")
                st.write("Error details:", str(e))
                logging.error(f"Exception in Step 5: {e}")

        # Navigation Buttons
        col_back, col_next = st.columns(2)
        with col_back:
            st.button("Back", key="back_step5", on_click=prev_step)
        with col_next:
            if st.session_state.model_generated:
                st.button("Next", key="next_step5", on_click=next_step)
            else:
                st.button("Next", key="next_step5_disabled", on_click=next_step, disabled=True)
    
    # Step 6: Reveal Model Reasoning
    elif st.session_state.step == 6:
        logging.info("Entered Step 6: Reveal Model Reasoning.")
        st.markdown("<div class='step-title'>Step 6: Reveal Model Reasoning</div>", unsafe_allow_html=True)

        st.markdown(f"""
            <div style='color: #003366; font-size: 20px; margin-bottom: 20px;'>
                <p style='margin: 5px 0;'>Your Decision: {st.session_state.user_decision}</p>
                <p style='margin: 5px 0;'>Model Prediction: {st.session_state.model_prediction_label}</p>
            </div>
        """, unsafe_allow_html=True)

        # Only show override rules when applicable
        if "OVERRIDE APPLIED:" in st.session_state.override_reason:
            st.markdown(get_markdown_text(
                f"**Override Rule Applied:** {st.session_state.override_reason.replace('OVERRIDE APPLIED: ', '')}", 
                "highlighted_text"
            ), unsafe_allow_html=True)
                
        # Display scores with new explanation focus
        display_scenario_with_scores(
            st.session_state.scenario,
            feature_importances=rf_model_loaded.feature_importances_ if hasattr(rf_model_loaded, 'feature_importances_') else None,
            override_reason=st.session_state.override_reason
        )

        # In step 6, modify the expander section to:
        help_container = st.container()
        with help_container:
            col1, col2 = st.columns([0.97, 0.03])
            with col1:
                with st.expander("Total Score Meaning"):
                    st.markdown("""
                        **Score Ranges:**
                        - **≥ 30**: Generally favorable conditions
                        - **22.5-30**: Conditions that might require additional authorization
                        - **15-22.5**: Situations with significant uncertainty
                        - **< 15**: Generally unfavorable conditions
            
                        Note: These ranges are scenario reference points rather than strict rules.  
                    """)
        
                with st.expander("Model Decision Logic"):
                    st.markdown("""
                        1. **Pattern Recognition**: The model analyzes patterns from its training data
                        2. **Context Analysis**: Considers the complete scenario context
                        3. **Feature Interaction**: Evaluates how different factors influence each other
                        4. **Score Guidance**: Uses scores as reference points, not rules
                        5. **Override Rules**: Applies critical legal and ethical constraints when necessary
                    """)

        # Set revealed_reasoning to True since we're showing everything directly
        st.session_state.revealed_reasoning = True

        # Navigation Buttons
        col_back, col_next = st.columns(2)
        with col_back:
            st.button("Back", key="back_step6", on_click=prev_step)
        with col_next:
            st.button("Next", key="next_step6", on_click=next_step)

    # Step 7: Provide Confirmation Feedback
    elif st.session_state.step == 7:
        st.markdown("<div class='step-title'>Step 7: Provide Confirmation Feedback</div>", unsafe_allow_html=True)
        
        # Keep only this version with tight line spacing
        st.markdown(f"""<div style='line-height: 1.2;'>
                <p style='color: #003366; font-size: 20px; margin: 12px 0;'>
                    Your Decision: {st.session_state.user_decision}<br>
                    Model Prediction: {st.session_state.model_prediction_label}
                </p>
        </div>""", unsafe_allow_html=True)
        
        if st.session_state.override_reason and "No override rules applied" not in st.session_state.override_reason:
            st.markdown(get_markdown_text(f"Override Rule Applied: {st.session_state.override_reason}", "highlighted_text"), unsafe_allow_html=True)
        
        st.markdown(get_markdown_text("Do you agree with the model's prediction?", "normal_text"), unsafe_allow_html=True)
        feedback_options = [
            "Strongly Disagree",
            "Disagree",
            "Neither Agree Nor Disagree",
            "Agree",
            "Strongly Agree"
        ]
        confirmation_feedback = st.radio(
            "",
            feedback_options,
            key="confirmation_feedback_radio",
            help="Your feedback helps us improve the model."
        )

        # Navigation Buttons
        col_back, col_submit = st.columns(2)
        with col_back:
            st.button("Back", key="back_step7", on_click=prev_step)
        with col_submit:
            submit_feedback = st.button("Submit Feedback", key="submit_feedback")
            if submit_feedback and confirmation_feedback:
                st.session_state.confirmation_feedback = confirmation_feedback
                st.session_state.submitted_feedback = True
                st.success("Thank you for your feedback!")
                logging.info(f"User feedback submitted: {confirmation_feedback}")

        if st.session_state.submitted_feedback:
            st.button("Next", key="next_step7", on_click=next_step)

    # Step 8: Share Additional Feedback
    elif st.session_state.step == 8:
        logging.info("Entered Step 8: Share Additional Feedback.")
        st.markdown("<div class='step-title'>Step 8: Share Additional Feedback</div>", unsafe_allow_html=True)
        st.markdown(get_markdown_text("Please provide any additional thoughts or comments below.", "normal_text"), unsafe_allow_html=True)
        
        st.text_area(
            "",
            key="feedback_box",
            help="Share any additional thoughts or comments."
        )

        # Navigation Buttons
        col_back, col_submit = st.columns(2)
        with col_back:
            st.button("Back", key="back_step8", on_click=prev_step)
        with col_submit:
            st.button("Submit Additional Feedback", key="submit_feedback_additional", on_click=handle_submit_feedback)
            st.button("Skip", key="skip_feedback", on_click=handle_skip_feedback)

    # Step 9: Completion
    elif st.session_state.step == 9:
        logging.info("Entered Step 9: Completion.")
        st.markdown(get_markdown_text("You have completed all steps.", "subheader"), unsafe_allow_html=True)
        st.write("Thank you for participating in this study.")
        message_placeholder = st.empty()
        if st.button("Start New Scenario", key="start_new_scenario_button"):
            # Reset session state variables
            st.session_state.step = 2
            st.session_state.scenario = None
            st.session_state.user_decision = None
            st.session_state.model_prediction_label = None
            st.session_state.override_reason = None
            st.session_state.confirmation_feedback = None
            st.session_state.feedback_shared = False
            st.session_state.progress = 0
            st.session_state.start_time = None
            st.session_state.decision_time = None
            st.session_state.submitted_decision = False
            st.session_state.submitted_feedback = False
            st.session_state.scenario_generated = False
            st.session_state.model_generated = False
            st.session_state.revealed_reasoning = False
            st.session_state.raw_model_prediction = None
            logging.info("Starting new scenario.")
            st.session_state['rerun_counter'] = st.session_state.get('rerun_counter', 0) + 1
            message_placeholder.success("New scenario initialized.")

    else:
        st.markdown("Other steps here...")

# Add at the very end of the file:
if __name__ == '__main__':
    main()

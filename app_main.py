import streamlit as st
import pandas as pd
import joblib
import random
import os
import logging
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---------------------------
# Logging & Page Configuration
# ---------------------------
logging.basicConfig(level=logging.INFO)
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
# ---------------------------
# Session State Initialization (including multi-scenario variables)
# ---------------------------
session_vars = [
    "step", "scenario", "user_decision", "model_prediction_label",
    "override_reason", "confirmation_feedback", "feedback_shared",
    "progress", "start_time", "decision_time",
    "submitted_decision", "submitted_feedback",
    "scenario_generated", "model_generated", "revealed_reasoning",
    "raw_model_prediction", "scenario_count", "flow", "new_step_index"
]
for var in session_vars:
    if var not in st.session_state:
        if var == "step":
            st.session_state[var] = 1
        elif var in ["scenario_generated", "model_generated", "revealed_reasoning"]:
            st.session_state[var] = False
        else:
            st.session_state[var] = None

if st.session_state.scenario_count is None:
    st.session_state.scenario_count = 1  # Start with scenario 1
if st.session_state.flow is None:
    # Use "original" flow for scenarios 1–5; "reordered" for scenarios 6–10
    if st.session_state.scenario_count <= 5:
        st.session_state.flow = "original"
    else:
        st.session_state.flow = "reordered"
if st.session_state.new_step_index is None:
    st.session_state.new_step_index = 0

if "time_remaining" not in st.session_state:
    st.session_state.time_remaining = 300
if "timer_active" not in st.session_state:
    st.session_state.timer_active = False
if "start" not in st.session_state or st.session_state.start is None:
    st.session_state.start = time.time()

# ---------------------------
# Styles for Markdown Elements
# ---------------------------
MARKDOWN_STYLE = {
    "header": "<h1 style='font-size: 25px; line-height: 1; text-align: center; color: #003366;'>",
    "subheader": "<h2 style='font-size: 23px; line-height: 1; color: #003366;'>",
    "normal_text": "<p style='font-size: 18px; line-height: 1;'>",
    "highlighted_text": "<p style='font-size: 18px; line-height: 1; color: #CC0000; font-weight: bold;'>",
    "decision_text": "<p style='font-size: 21px; line-height: 1; color: #003366; font-weight: bold;'>"
}

def get_markdown_text(text, style_key):
    tag_mapping = {
        "header": "h1",
        "subheader": "h2",
        "normal_text": "p",
        "highlighted_text": "p",
        "decision_text": "p"
    }
    tag = tag_mapping.get(style_key, "p")
    style = MARKDOWN_STYLE.get(style_key, MARKDOWN_STYLE["normal_text"])
    if style.endswith('>'):
        style = style[:-1]
    return f"{style}>{text}</{tag}>"

def convert_civilian_presence(value):
    if isinstance(value, str) and '-' in value:
        return value
    try:
        return str(int(value))
    except ValueError:
        return "0"

# ---------------------------
# Data Columns & Model Files
# ---------------------------
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
score_columns = [pair[1] for pair in columns_to_shuffle]
label_mapping = {
    0: 'Do Not Engage',
    1: 'Ask Authorization',
    2: 'Do Not Know',
    3: 'Engage'
}

try:
    model_path = 'MDMP_model.joblib'
    features_path = 'MDMP_feature_columns.joblib'
    csv_path = 'dataset_with_all_category_scores.csv'
    
    rf_model_loaded = joblib.load(model_path)
    trained_feature_columns = joblib.load(features_path)
    df = pd.read_csv(csv_path)
    print("Loaded data columns:", df.columns.tolist())
    print("Trained feature columns:", trained_feature_columns)
    logging.info("Model and data loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    logging.error(f"Error loading model or data: {e}")
    st.stop()

def shuffle_dataset(df):
    print("Original columns:", df.columns.tolist())
    df_shuffled = df.copy()
    for related_columns in columns_to_shuffle:
        print(f"Processing columns: {related_columns}")
        shuffled_subset = df[related_columns].sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
        df_shuffled[related_columns] = shuffled_subset
    df_shuffled['Total_Score'] = df_shuffled[score_columns].sum(axis=1)
    print("Final columns:", df_shuffled.columns.tolist())
    return df_shuffled

def get_random_scenario(df):
    random_index = random.randint(0, len(df) - 1)
    return df.iloc[random_index]

def calculate_percentages(scores):
    total_abs = sum(abs(v) for k, v in scores.items() if k != "Total_Score")
    if total_abs == 0:
        return {k: 0 for k in scores if k != "Total_Score"}
    percentages = {
        k: round((abs(v) / total_abs) * 100, 2) * (1 if v >= 0 else -1)
        for k, v in scores.items() if k != "Total_Score"
    }
    return percentages


def get_score_display(score, percentage):
    if score > 0:
        color = "#28a745"
    elif score < 0:
        color = "#dc3545"
    else:
        color = "#6c757d"
    return f"<b>{score}</b> (<span style='color:{color}'>{percentage:.2f}%</span>)"

def assign_final_decision(total_score):
    if total_score >= 30:
        return 'Engage'
    elif total_score >= 22.5:
        return 'Ask Authorization'
    elif total_score >= 15:
        return 'Do Not Know'
    else:
        return 'Do Not Engage'

def verify_scenario_data(scenario):
    required_columns = [col[0] for col in columns_to_shuffle]
    if isinstance(scenario, pd.Series) or any(col in scenario.index for col in required_columns):
        return []
    if isinstance(scenario, pd.DataFrame):
        missing_columns = [col for col in required_columns if col not in scenario.columns]
        return missing_columns
    return []

def get_final_prediction(scenario_df, model):
    try:
        if 'Total_Score' not in scenario_df.columns or pd.isna(scenario_df['Total_Score']).all():
            scenario_df['Total_Score'] = scenario_df[score_columns].sum(axis=1)
        total_score = scenario_df['Total_Score'].iloc[0]
        override_decision, override_reason = apply_override_rules(scenario_df.iloc[0])
        try:
            model_pred = model.predict(scenario_df)[0]
            model_label = label_mapping.get(model_pred, "Unknown")
        except Exception as e:
            logging.error(f"Error in model prediction: {e}")
            model_label = None
        if override_decision:
            return override_decision, f"OVERRIDE APPLIED: {override_reason}", model_label
        else:
            score_based_decision = assign_final_decision(total_score)
            return score_based_decision, "", model_label
    except Exception as e:
        logging.error(f"Error in get_final_prediction: {e}")
        return None, f"Error in prediction: {e}", None



def apply_override_rules(row):
    try:
        categories = {col: row[col] for col in row.index if not col.endswith('_Score')}
        if 'Total_Score' not in row or pd.isna(row['Total_Score']):
            row['Total_Score'] = row[[col for col in row.index if col.endswith('_Score')]].sum()
            logging.info("Calculated Total_Score in apply_override_rules")
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
        if categories['Weaponeering'] == "Anti-Personnel Mine" and \
           categories['Target_Category'] in ["Fighter Aircraft", "Frigate", "Ship Maintenance Facility", "Naval Base"]:
            return "Do Not Know", "Inappropriate weapon for target type"
        if categories['Weaponeering'] == "Torpedo" and \
           categories['Target_Category'] not in ["Ship Maintenance Facility", "Naval Base", "Frigate"]:
            return "Do Not Know", "Torpedo inappropriate for non-naval target"
        return None, "No override rules applied"
    except KeyError as e:
        logging.error(f"Missing required column in override rules: {e}")
        return None, "No override rules applied"
    except Exception as e:
        logging.error(f"Unexpected error in apply_override_rules: {e}")
        return None, "No override rules applied"

def get_google_sheet():
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
        client = gspread.authorize(creds)
        sheet = client.open("Study_data").sheet1
        return sheet
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        logging.error(f"Error connecting to Google Sheets: {e}")
        return None

def save_data_to_google_sheet(data):
    sheet = get_google_sheet()
    if sheet:
        try:
            scenario_details = ", ".join(f"{key}: {value}" for key, value in data.get('scenario', {}).items())
            row = [
                scenario_details,
                data.get('Participant Decision', ''),
                data.get('Model Prediction', ''),
                data.get('Decision Time (seconds)', ''),
                data.get('Confirmation Feedback', ''),
                data.get('Additional Feedback', ''),
            ]
            sheet.append_row(row)
            logging.info("Data appended to Google Sheets successfully.")
        except Exception as e:
            st.error(f"Error saving data to Google Sheets: {e}")
            logging.error(f"Error saving data to Google Sheets: {e}")

def display_scenario_with_scores(scenario, feature_importances=None, override_reason=None):
    columns_to_display = [col[0] for col in columns_to_shuffle]
    if st.session_state.step < 6:
        for column in columns_to_display:
            value = scenario[column] if column in scenario and pd.notna(scenario[column]) else "Unknown"
            st.markdown(f"""
                <div style='font-size: 16px; margin-bottom: 1px;'>
                    <b>{column}</b>: {value}
                </div>
            """, unsafe_allow_html=True)
    else:
        scores = {f"{col}_Score": scenario[f"{col}_Score"] for col in columns_to_display if f"{col}_Score" in scenario}
        percentages = calculate_percentages(scores)
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
        total_score = sum(scores.values())
        st.markdown(f"""
            <div style='margin-top: 15px; color: #CC0000; font-weight: bold;'>
                Total Score: {total_score}
            </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Navigation Functions (with updated multi-scenario logic)
# ---------------------------
def next_step():
    if st.session_state.flow == "original":
        if st.session_state.step < 9:
            st.session_state.step += 1
            logging.info(f"Original flow: Moved to Step {st.session_state.step}")
        else:
            # End of scenario in original flow: increment scenario_count
            st.session_state.scenario_count += 1
            logging.info(f"Completed scenario {st.session_state.scenario_count - 1} in original flow")
            if st.session_state.scenario_count > 5:
                st.session_state.flow = "reordered"
                st.session_state.new_step_index = 0
                st.session_state.step = 2
            else:
                st.session_state.step = 2
            reset_scenario_states()
            st.rerun()
    else:  # Reordered flow for scenarios 6–10
        # Updated reorder flow now includes Step 7: [2, 5, 6, 3, 4, 7, 8, 9]
        reorder_flow = [2, 5, 6, 3, 4, 7, 8, 9]
        if st.session_state.new_step_index < len(reorder_flow) - 1:
            st.session_state.new_step_index += 1
            st.session_state.step = reorder_flow[st.session_state.new_step_index]
            logging.info(f"Reordered flow: Moved to Step {st.session_state.step} (index {st.session_state.new_step_index})")
        else:
            st.session_state.scenario_count += 1
            logging.info(f"Completed scenario {st.session_state.scenario_count - 1} in reordered flow")
            if st.session_state.scenario_count > 10:
                st.info("Study completed. Please refresh the page for the next round.")
                st.stop()
            else:
                st.session_state.new_step_index = 0
                st.session_state.step = reorder_flow[0]
            reset_scenario_states()
            st.rerun()

def prev_step():
    if st.session_state.flow == "original":
        if st.session_state.step > 1:
            st.session_state.step -= 1
            st.session_state.timer_active = False
            st.session_state.time_remaining = 300
            logging.info(f"Original flow: Moved back to Step {st.session_state.step}")
    else:
        reorder_flow = [2, 5, 6, 3, 4, 7, 8, 9]
        if st.session_state.new_step_index > 0:
            st.session_state.new_step_index -= 1
            st.session_state.step = reorder_flow[st.session_state.new_step_index]
            logging.info(f"Reordered flow: Moved back to Step {st.session_state.step} (index {st.session_state.new_step_index})")

def reset_scenario_states():
    st.session_state.scenario = None
    st.session_state.user_decision = None
    st.session_state.model_prediction_label = None
    st.session_state.override_reason = None
    st.session_state.confirmation_feedback = None
    st.session_state.feedback_shared = False
    st.session_state.start_time = None
    st.session_state.decision_time = None
    st.session_state.submitted_decision = False
    st.session_state.submitted_feedback = False
    st.session_state.scenario_generated = False
    st.session_state.model_generated = False
    st.session_state.revealed_reasoning = False
    st.session_state.raw_model_prediction = None
    st.session_state.time_remaining = 300
    st.session_state.timer_active = False
    st.session_state.start = None

# ---------------------------
# Feedback Handling Functions
# ---------------------------
def handle_submit_feedback():
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

# ---------------------------
# Main Application Function
# ---------------------------
def main():
    # Custom CSS
    st.markdown("""
        <style>
            .step-title {
                font-size: 20px;
                font-weight: bold;
                color: #003366;
                margin-bottom: 2px;
            }
            .scenario-guide {
                margin-top: -15px;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
        .stButton > button {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .reportview-container .main footer {visibility: hidden;}
            iframe {display: none;}
            div[data-testid="stDecoration"] {display: none;}
            .element-container iframe {display: none;}
            ::-webkit-scrollbar { width: 10px; }
            ::-webkit-scrollbar-track { background: #f1f1f1; }
            ::-webkit-scrollbar-thumb { background: #888; }
            ::-webkit-scrollbar-thumb:hover { background: #555; }
            .main-header { font-size: 32px; color: #003366; text-align: center; margin-top: -50px; padding: 10px 0; }
            .card { background-color: #F0F8FF; padding: 5px; border-radius: 10px; box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1); }
            .card h3 { color: #003366; margin-top: 0; line-height: 1.2; font-size: 20px; font-weight: bold; }
            .card p { font-size: 16px; margin: 2px 0; line-height: 1; }
            .card-right { background-color: #F0F8FF; padding: 15px; border-radius: 10px; box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1); margin-top: 0; }
            .time-remaining { font-size: 16px; color: #003366; font-weight: bold; text-align: center; padding: 5px; background-color: #F0F8FF; border-radius: 5px; margin: 5px 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }
            .stButton button { background-color: #003366; color: white; border-radius: 10px; padding: 10px 20px; font-weight: bold; transition: background-color 0.3s; }
            .stButton button:hover { background-color: #002244; }
            .stRadio label { font-size: 15px; padding: 1px 0; line-height: 0.5; }
            .stRadio > div { gap: 1px !important; }
            .stProgress .st-ba { background-color: #003366; }
            .decision-text { font-size: 16px; color: #003366; font-weight: bold; margin: 15px 0; padding: 10px; background-color: #F0F8FF; border-radius: 5px; }
            .score-text { font-size: 18px; line-height: 1; margin: 4px 0; }
            .streamlit-expanderHeader { font-size: 16px; color: #003366; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    # Always show the title and scenario counter at the top
    st.markdown(get_markdown_text("Military Decision-Making App", "header"), unsafe_allow_html=True)
    logging.info("App started.")
    scenario_num = st.session_state.scenario_count
    st.markdown(f"<h6 style='text-align:center; color:#003366;'>Scenario {scenario_num} of 10</h4>", unsafe_allow_html=True)
    
    # Progress Indicator
    total_steps = 9
    st.session_state.progress = (st.session_state.step - 1) / (total_steps - 1)
    st.progress(st.session_state.progress)
    logging.info(f"Progress: {st.session_state.progress}, Step: {st.session_state.step}")

    # ---------------------------
    # Step-based Logic
    # ---------------------------
    # Step 1: Introduction and Scenario Guide (only for scenario 1 in original flow)
    if st.session_state.step == 1 and st.session_state.flow == "original":
        logging.info("Entered Step 1: Introduction and Scenario Guide.")
        st.markdown("<div class='step-title'>Step 1: Introduction</div>", unsafe_allow_html=True)
        st.markdown(get_markdown_text("""
The App explores human-machine teaming in military contexts.

*Review the instructions below:*
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
<summary><strong>Tutorial</strong></summary>

<p>The following is an example to guide you through decision-making steps to cooperate with the model.</p>

<ol>
  <li>
    <strong>Tutorial Scenario Details:</strong>
    <ul>
      <li>Target_Category: Artillery Unit</li>
      <li>Terrain_Type: Open Field</li>
      <li>Civilian_Presence: 0</li>
      <li>Time_Sensitivity: Immediate</li>
      <li>Weaponeering: Precision Guided Munition</li>
      <li>Friendly_Fire: Very Low</li>
      <li>Legal_Advice: Lawful</li>
    </ul>
  </li>
  <li>
    <strong>Practice Decision:</strong>
    <ul>
      <li>Review the parameters above and select the most appropriate decision: 
        <em>Engage, Do Not Engage, Ask Authorization, or Do Not Know</em> within a 5-minute time limit.</li>
      <li>Please note: The randomized parameters may occasionally contradict each other 
        (for example, <em>Target_Category: Medical Depot</em> with <em>Legal_Advice: Lawful</em>).</li>
      <li>In case of any illogical outcomes, make the best possible decision based on the given context.</li>
      <li>In the following steps, you will review the AI prediction scores and familiarize yourself with 
        the total score model methodology.</li>
      <li>Provide confirmation feedback when prompted.</li>
    </ul>
    <p>What would be your decision for this practice scenario?</p>
    <ul>
      <li>Engage</li>
      <li>Do Not Engage</li>
      <li>Ask Authorization</li>
      <li>Do Not Know</li>
    </ul>
    <p><strong>NB! This is a tutorial. Any option could be valid in this context</strong></p>
  </li>
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
            <li>For scenarios 6–10, the step order changes to: Steps 2, 5, 6, 3, 4, 7, 8, 9. In these scenarios, the model generates its decision first, and you are then allowed to review the scenario and submit your own decision, allowing you to either rely on or challenge the model's recommendation.</li>
            </ul>
            </details>
        """, unsafe_allow_html=True)
        st.markdown("""
            <details>
            <summary><strong>Getting Started</strong></summary>
            <ol>
                <li>Refer to the "Scenario Guide" if necessary to refresh your understanding of parameter definitions.</li>
                <li>Review the scenario details carefully.</li>
                <li>Submit your decision. If the timer expires, a decision will be auto-submitted.</li>
                <li>Then, you'll interact with a pre-trained AI model.</li>
            </ol>
            </details>
        """, unsafe_allow_html=True)
        st.button("Proceed to Scenario Generation", key="proceed_to_scenario_generation", on_click=next_step)

    # Step 2: Generate Scenario
    elif st.session_state.step == 2:
        logging.info("Entered Step 2: Generate Scenario.")
        st.markdown("<div class='step-title'>Step 2: Generate Scenario</div>", unsafe_allow_html=True)
        st.markdown(get_markdown_text("<i>Click the button below to generate a new scenario.</i>", "normal_text"), unsafe_allow_html=True)
        generate_button = st.button("Generate Scenario", key="generate_scenario")
        if generate_button:
            try:
                logging.info("Starting scenario generation")
                st.session_state.df_shuffled = shuffle_dataset(df)
                logging.info("Dataset shuffled successfully")
                st.session_state.scenario = get_random_scenario(st.session_state.df_shuffled)
                logging.info("Random scenario selected successfully")
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
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        col_back, col_next = st.columns(2)
        with col_back:
            st.button("Back", key="back_step3", on_click=prev_step)
        with col_next:
            st.button("Proceed to Decision Making", key="proceed_to_decision_step3", on_click=next_step)

    # Step 4: Submit Decision
    elif st.session_state.step == 4:
        logging.info("Entered Step 4: Submit Decision.")
        st.markdown(get_markdown_text("<i>Please review the scenario and select your decision below.</i>", "normal_text"), unsafe_allow_html=True)
        if not st.session_state.timer_active:
            st.session_state.time_remaining = 300
            st.session_state.timer_active = True
            st.session_state.start = time.time()
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
        display_scenario_with_scores(st.session_state.scenario)
        if st.session_state.time_remaining > 0:
            user_decision = st.radio("", ["Engage", "Do Not Engage", "Ask Authorization", "Do Not Know"],
                                     key="decision", help="Select the most appropriate decision based on the scenario.")
            if user_decision:
                st.session_state.user_decision = user_decision
        else:
            st.warning("Time's up! No more decisions allowed.")
        col_back, col_submit = st.columns(2)
        with col_back:
            st.button("Back", key="back_step4", on_click=prev_step)
        with col_submit:
            if st.session_state.time_remaining > 0:
                submit_decision = st.button("Submit Decision", key="submit_decision")
                if submit_decision:
                    if isinstance(st.session_state.start, float):
                        st.session_state.decision_time = 300 - (time.time() - st.session_state.start)
                    else:
                        st.session_state.start = time.time()
                        st.session_state.decision_time = 300
                    st.session_state.submitted_decision = True
                    st.session_state.timer_active = False
                    st.success("Decision submitted successfully!")
        if st.session_state.submitted_decision:
            st.button("Next", key="next_step4", on_click=next_step)
        if st.session_state.timer_active and st.session_state.time_remaining > 0:
            time.sleep(1)
            st.session_state.time_remaining -= 1
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
                scenario_data = pd.DataFrame([{col: st.session_state.scenario[col] for col in trained_feature_columns}])
                final_decision, reason, raw_model_pred = get_final_prediction(scenario_data, rf_model_loaded)
                if final_decision:
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

        display_scenario_with_scores(st.session_state.scenario, feature_importances=rf_model_loaded.feature_importances_ if hasattr(rf_model_loaded, 'feature_importances_') else None, override_reason=st.session_state.override_reason)
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
                        1. **Pattern Recognition**: The model analyzes patterns from its training data.
                        2. **Context Analysis**: Considers the complete scenario context.
                        3. **Feature Interaction**: Evaluates how different factors influence each other.
                        4. **Score Guidance**: Uses scores as reference points, not rules.
                        5. **Override Rules**: Applies critical legal and ethical constraints when necessary.
                    """)
        st.session_state.revealed_reasoning = True
        col_back, col_next = st.columns(2)
        with col_back:
            st.button("Back", key="back_step6", on_click=prev_step)
        with col_next:
            st.button("Next", key="next_step6", on_click=next_step)

    # Step 7: Provide Confirmation Feedback (appears in both flows)
    elif st.session_state.step == 7:
        st.markdown("<div class='step-title'>Step 7: Provide Confirmation Feedback</div>", unsafe_allow_html=True)
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
        confirmation_feedback = st.radio("", feedback_options, key="confirmation_feedback_radio", help="Your feedback helps us improve the model.")
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
        st.text_area("", key="feedback_box", help="Share any additional thoughts or comments.")
        col_back, col_submit = st.columns(2)
        with col_back:
            st.button("Back", key="back_step8", on_click=prev_step)
        with col_submit:
            st.button("Submit Additional Feedback", key="submit_feedback_additional", on_click=handle_submit_feedback)
            st.button("Skip", key="skip_feedback", on_click=handle_skip_feedback)

    # Step 9: Completion – update scenario counter here
    elif st.session_state.step == 9:
        logging.info("Entered Step 9: Completion.")
        st.markdown(get_markdown_text("You have completed all steps for this scenario.", "subheader"), unsafe_allow_html=True)
        st.write("Thank you for participating in this scenario.")
        message_placeholder = st.empty()
        if st.button("Start New Scenario", key="start_new_scenario_button"):
            st.session_state.scenario_count += 1
            if st.session_state.scenario_count > 10:
                st.info("Study completed. Please refresh the page for the next round.")
                st.stop()
            else:
                if st.session_state.scenario_count <= 5:
                    st.session_state.flow = "original"
                else:
                    st.session_state.flow = "reordered"
                    st.session_state.new_step_index = 0
                st.session_state.step = 2
                reset_scenario_states()
                st.rerun()
    else:
        st.markdown("Other steps here...")


if __name__ == '__main__':
    main()

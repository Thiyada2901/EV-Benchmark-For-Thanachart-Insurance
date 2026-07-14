import streamlit as st
import pandas as pd
import numpy as np

# --- Set Streamlit Theme (Orange-White) ---
st.set_page_config(
    page_title="EV benchmark App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Main background color */
    .stApp { background-color: #FFFFFF; } /* White */
    /* Sidebar background color */
    .st-emotion-cache-1ldk86k { background-color: #FFE0B2; } /* Lighter Orange for sidebar */
    .st-emotion-cache-1ldk86k .st-emotion-cache-vk3370 { background-color: #FFCC80; } /* Medium Orange for sidebar header */

    /* Primary button color (orange) */
    .stButton>button { background-color: #FFA500; color: white; border: none; }
    .stButton>button:hover { background-color: #FF8C00; color: white; }

    /* Text color */
    body { color: #333333; } /* Dark Grey */
    h1, h2, h3, h4, h5, h6 { color: #FF6F00; } /* More vibrant orange for headings */

    /* Expander background */
    .streamlit-expanderHeader { background-color: #FFF5E0; } /* Lightest Orange for expander */
    .streamlit-expanderContent { background-color: #FFFFFF; } /* White */

    /* Custom CSS for similarity badges */
    .similarity-badge {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 60px; /* Adjust size as needed */
        height: 60px;
        border-radius: 50%;
        color: white;
        font-weight: bold;
        font-size: 1.1em;
        background-color: grey; /* Default */
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .similarity-badge-green { background-color: #28a745; } /* Green */
    .similarity-badge-darkyellow { background-color: #ffc107; color: black; } /* Dark Yellow */
    .similarity-badge-lightyellow { background-color: #fff3cd; color: black; } /* Light Yellow */

    /* Basic table styling for the custom HTML table */
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 10px;
        text-align: left;
        vertical-align: middle;
    }
    th {
        background-color: #FFF5E0; /* Lightest Orange for table headers */
        color: #333333;
    }
    tr:nth-child(even) { background-color: #f9f9f9; }
    tr:hover { background-color: #f1f1f1; }

    </style>
    """,
    unsafe_allow_html=True
)

# --- Add Thanachart Insurance Logo ---
try:
    # Create columns to push the image to the right
    col1, col2 = st.columns([0.8, 0.2]) # Adjust ratios as needed
    with col2:
        st.image('thanachart_logo.jpg', width=150) # Adjust width as needed
except FileNotFoundError:
    st.warning("Thanachart Insurance logo (thanachart_logo.jpg) not found. Please upload the image file.")

st.title("EV benchmark for Thanachart insurance")
st.write("ค้นหารถยนต์ไฟฟ้าที่คล้ายคลึงกันตามคุณสมบัติที่คุณเลือก")

# --- โหลดข้อมูล (ควรอยู่ในไฟล์ Streamlit โดยตรงเมื่อใช้งานจริง) ---
try:
    df = pd.read_excel('database.xlsx')
    df.columns = df.columns.str.strip()
    # Strip whitespace from categorical columns
    for col in ['body_type', 'Battery Manufacturer', 'Battery Material']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
except FileNotFoundError:
    st.error("Error: 'database.xlsx' not found. Please make sure the file is in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading data: {e}")
    st.stop()

# เตรียมข้อมูล: กำจัดแถวที่มีค่าว่างในคอลัมน์สำคัญที่อาจส่งผลต่อการดึงตัวเลือก
df_cleaned_for_options = df.dropna(subset=['body_type', 'Battery Manufacturer', 'Battery Material'])

# *** สำคัญ: แก้ชื่อคอลัมน์ตรงนี้ให้ตรงกับไฟล์ Excel จริงของคุณ ***
FEATURE_WEIGHTS = {
    "Vehicle Sales Price (in THB)": 0.163,
    "Battery Capacity(kWh)": 0.163,
    "Electric Motor (Motor power) (kW)": 0.163,
    "Rang(WLTP)": 0.054, # Reduced weight to accommodate NEDC
    "Rang(NEDC)": 0.055, # Added NEDC with a similar weight. Total range weight ~0.109
    "Motor Torque": 0.076,
    "Vehicle weight": 0.076,
    "ความยาว(mm)": 0.054,
    "ความกว้าง(mm)": 0.054,
    "ความสูง(mm)": 0.054,
    "ระยะฐานล้อ(mm)": 0.054,
    "Ground Clearance": 0.033
}

CATEGORICAL_FEATURES = {
    "body_type": 0.3,
    "Battery Manufacturer": 0.4,
    "Battery Material": 0.3
}

def normalize_features(df_input, features):
    df_norm = df_input.copy()
    for col in features:
        df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce').fillna(0)
        min_val, max_val = df_norm[col].min(), df_norm[col].max()
        if max_val - min_val == 0:
            df_norm[col] = 0.5
        else:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def calculate_similarity(input_specs, database_df,
                         numerical_weights=FEATURE_WEIGHTS,
                         categorical_weights=CATEGORICAL_FEATURES,
                         w_market_segment=0.33,
                         w_vehicle_char=0.34,
                         w_pricing_cost=0.33,
                         alpha_num_cat=0.7,
                         top_n=5):

    # Define features for each category using the global weights
    market_segment_num_feats = {f: numerical_weights[f] for f in ['Vehicle Sales Price (in THB)'] if f in numerical_weights}
    market_segment_cat_feats = {f: categorical_weights[f] for f in ['body_type'] if f in categorical_weights}

    # Updated to include both Rang(WLTP) and Rang(NEDC) for processing
    vehicle_char_num_feats = {f: numerical_weights[f] for f in ['ความกว้าง(mm)', 'ความยาว(mm)', 'ความสูง(mm)', 'ระยะฐานล้อ(mm)', 'Battery Capacity(kWh)', 'Electric Motor (Motor power) (kW)', 'Motor Torque', 'Rang(WLTP)', 'Rang(NEDC)', 'Vehicle weight', 'Ground Clearance'] if f in numerical_weights}
    vehicle_char_cat_feats = {f: categorical_weights[f] for f in ['Battery Manufacturer', 'Battery Material'] if f in categorical_weights}

    pricing_cost_num_feats = {f: numerical_weights[f] for f in ['Vehicle Sales Price (in THB)'] if f in numerical_weights}
    pricing_cost_cat_feats = {}

    # --- Helper function to calculate sub-similarity for a given set of features ----
    def _calculate_sub_similarity(sub_numerical_weights, sub_categorical_weights):
        sub_numerical_features = list(sub_numerical_weights.keys())
        sub_categorical_features = list(sub_categorical_weights.keys())

        # Numerical part
        sub_numerical_scores = [0] * len(database_df) # Default to 0 if no numerical features or input issues
        if sub_numerical_features:
            combined_numerical_subset = database_df.copy()
            for col in sub_numerical_features:
                # Only include columns actually present in the database_df for normalization
                if col not in combined_numerical_subset.columns:
                    # st.warning(f"DEBUG: Column '{col}' not found for numerical subset. Skipping.")
                    continue # Skip if column not in database_df
                combined_numerical_subset[col] = pd.to_numeric(combined_numerical_subset[col], errors='coerce').fillna(0)

            input_row_numerical_subset = {k: v for k, v in input_specs.items() if k in sub_numerical_features}
            if input_row_numerical_subset:
                # Ensure all numerical features for this sub-category are in the input row, fill with 0 if missing
                for f in sub_numerical_features:
                    if f not in input_row_numerical_subset:
                        input_row_numerical_subset[f] = 0.0 # Use 0.0 for numerical consistency

                # Add temporary ID for input row before concatenating
                input_row_for_concat = input_row_numerical_subset.copy()
                input_row_for_concat['__TEMP_ID__'] = '__INPUT__'

                combined_numerical_subset_with_input = pd.concat([combined_numerical_subset, pd.DataFrame([input_row_for_concat])], ignore_index=True)
                combined_norm_subset = normalize_features(combined_numerical_subset_with_input, sub_numerical_features)

                input_norm_subset = combined_norm_subset[combined_norm_subset["__TEMP_ID__"] == "__INPUT__"].iloc[0]
                db_norm_subset = combined_norm_subset[combined_norm_subset["__TEMP_ID__"] != "__INPUT__"].reset_index(drop=True)

                sub_numerical_scores = []
                for idx, row in db_norm_subset.iterrows():
                    dist_sq = sum(sub_numerical_weights[f] * (row[f] - input_norm_subset[f]) ** 2 for f in sub_numerical_features if f in row.index and f in input_norm_subset.index) if sub_numerical_features else 0
                    distance = np.sqrt(dist_sq)
                    similarity_pct = max(0, (1 - distance)) * 100
                    sub_numerical_scores.append(similarity_pct)
            else:
                 sub_numerical_scores = [0] * len(database_df)

        # Categorical part
        sub_categorical_scores = [0] * len(database_df) # Default to 0
        total_sub_cat_weight = sum(sub_categorical_weights.values())

        if total_sub_cat_weight > 0:
            sub_categorical_scores = []
            for idx, db_row in database_df.iterrows():
                cat_sim_sum = 0
                for cat_feat, weight in sub_categorical_weights.items():
                    if cat_feat in input_specs and cat_feat in db_row:
                        # Ensure consistency by stripping spaces from both input and database values
                        if str(input_specs[cat_feat]).strip().lower() == str(db_row[cat_feat]).strip().lower():
                            cat_sim_sum += weight
                sub_categorical_scores.append((cat_sim_sum / total_sub_cat_weight) * 100)

        # Combine numerical and categorical for this sub-category
        combined_sub_scores = []
        for i in range(len(database_df)):
            score = (alpha_num_cat * sub_numerical_scores[i]) + ((1 - alpha_num_cat) * sub_categorical_scores[i])
            combined_sub_scores.append(score)
        return combined_sub_scores

    # --- Calculate scores for each main category ---
    market_segment_scores = _calculate_sub_similarity(market_segment_num_feats, market_segment_cat_feats)
    vehicle_char_scores = _calculate_sub_similarity(vehicle_char_num_feats, vehicle_char_cat_feats)
    pricing_cost_scores = _calculate_sub_similarity(pricing_cost_num_feats, pricing_cost_cat_feats)

    # --- Combine the main category scores ---
    final_combined_scores = []
    for i in range(len(database_df)):
        score = (w_market_segment * market_segment_scores[i]) + \
                (w_vehicle_char * vehicle_char_scores[i]) + \
                (w_pricing_cost * pricing_cost_scores[i])
        final_combined_scores.append(score)

    result = database_df.copy()
    result["Similarity_Score"] = final_combined_scores
    result = result.sort_values("Similarity_Score", ascending=False).reset_index(drop=True)
    result["Similarity_Score"] = result["Similarity_Score"].round(1)
    return result.head(top_n)

# Helper function to generate HTML for similarity badge
def get_similarity_badge_html(score):
    if score >= 80:
        color_class = "similarity-badge-green"
    elif 50 <= score < 80:
        color_class = "similarity-badge-darkyellow"
    else:
        color_class = "similarity-badge-lightyellow"
    return f'<div class="similarity-badge {color_class}">{score:.1f}%</div>'

# --- Streamlit UI for Input ---
st.sidebar.header("คุณสมบัติรถยนต์ EV ที่ต้องการ")

with st.sidebar.expander("1. Market Segment", expanded=True):
    body_type_options = [''] + sorted(df_cleaned_for_options['body_type'].astype(str).unique().tolist())
    input_body_type = st.selectbox("ประเภทตัวถัง (body_type)", body_type_options)
    input_sales_price = st.number_input("ราคาขายรถยนต์ (Vehicle Sales Price (in THB))", min_value=0, value=1500000, step=10000)

with st.sidebar.expander("2. Vehicle Characteristics", expanded=True):
    input_length = st.number_input("ความยาว(mm)", min_value=0, value=4455, step=10)
    input_width = st.number_input("ความกว้าง(mm)", min_value=0, value=1875, step=10)
    input_height = st.number_input("ความสูง(mm)", min_value=0, value=1615, step=10)
    input_wheelbase = st.number_input("ระยะฐานล้อ(mm)", min_value=0, value=2720, step=10)
    input_ground_clearance = st.number_input("Ground Clearance", min_value=0, value=175, step=5)
    input_battery_capacity = st.number_input("Battery Capacity(kWh)", min_value=0, value=50, step=1)

    battery_manuf_options = [''] + sorted(df_cleaned_for_options['Battery Manufacturer'].astype(str).unique().tolist())
    input_battery_manufacturer = st.selectbox("ผู้ผลิตแบตเตอรี่ (Battery Manufacturer)", battery_manuf_options)

    input_motor_power = st.number_input("กำลังมอเตอร์ (Electric Motor (Motor power) (kW))", min_value=0, value=150, step=10)
    input_motor_torque = st.number_input("แรงบิดมอเตอร์ (Motor Torque)", min_value=0, value=310, step=10)

    # New UI for selecting Range Type and inputting Range Value
    range_type_choice = st.radio("เลือกประเภทระยะการขับขี่", ['Rang(WLTP)', 'Rang(NEDC)'])
    input_driving_range = st.number_input(f"ระยะการขับขี่ ({range_type_choice})", min_value=0, value=410, step=10)

    input_vehicle_weight = st.number_input("น้ำหนักรถยนต์ (Vehicle weight)", min_value=0, value=2090, step=10)

    # Apply .str.strip() here to clean Battery Material options
    battery_material_options = [''] + sorted(df_cleaned_for_options['Battery Material'].astype(str).str.strip().unique().tolist())
    input_battery_material = st.selectbox("วัสดุแบตเตอรี่ (Battery Material)", battery_material_options)

# Create a dictionary for the new EV from user inputs
new_ev_input = {
    "ความยาว(mm)": input_length,
    "ความกว้าง(mm)": input_width,
    "ความสูง(mm)": input_height,
    "ระยะฐานล้อ(mm)": input_wheelbase,
    "Ground Clearance": input_ground_clearance,
    "Battery Capacity(kWh)": input_battery_capacity,
    "Electric Motor (Motor power) (kW)": input_motor_power,
    "Motor Torque": input_motor_torque,
    "Vehicle Sales Price (in THB)": input_sales_price,
    "Vehicle weight": input_vehicle_weight,
}

# Conditionally add the chosen range type and value
if range_type_choice == 'Rang(WLTP)':
    new_ev_input["Rang(WLTP)"] = input_driving_range
elif range_type_choice == 'Rang(NEDC)':
    new_ev_input["Rang(NEDC)"] = input_driving_range

# Add categorical features if selected
if input_body_type: new_ev_input["body_type"] = input_body_type
if input_battery_manufacturer: new_ev_input["Battery Manufacturer"] = input_battery_manufacturer
# Ensure the input battery material is also stripped of spaces for matching
if input_battery_material: new_ev_input["Battery Material"] = input_battery_material.strip()

if st.sidebar.button("ค้นหารถยนต์ที่คล้ายกัน"):
    if not any(new_ev_input.values()):
        st.warning("กรุณากรอกข้อมูลคุณสมบัติรถยนต์อย่างน้อยหนึ่งรายการเพื่อทำการค้นหา")
    else:
        # Check if the selected range column exists in the dataframe before proceeding
        if range_type_choice not in df.columns:
            st.warning(f"คำเตือน: คอลัมน์ '{range_type_choice}' ไม่มีอยู่ในฐานข้อมูล. การคำนวณความคล้ายคลึงกันจะไม่ได้ใช้ค่านี้")

        st.subheader("ผลลัพธ์รถยนต์ที่คล้ายคลึงกันที่สุด")
        results_df = calculate_similarity(new_ev_input, df, top_n=5)
        results_df['Rank'] = results_df.reset_index(drop=True).index + 1

        # Apply the badge HTML to the Similarity_Score
        results_df['Similarity_Score_HTML'] = results_df['Similarity_Score'].apply(get_similarity_badge_html)

        # Build custom HTML table to display Brand, model_name, variant_name and custom score badge
        html_table = "<table><thead><tr>"
        html_table += "<th>Rank</th>"
        html_table += "<th>Brand</th>"
        html_table += "<th>Model Name</th>"
        html_table += "<th>Variant Name</th>"
        html_table += "<th>Similarity Score</th>"
        html_table += "</tr></thead><tbody>"

        for index, row in results_df.iterrows():
            html_table += "<tr>"
            html_table += f"<td>{row['Rank']}</td>"
            html_table += f"<td>{row['Brand']}</td>"
            html_table += f"<td>{row['model_name']}</td>"
            html_table += f"<td>{row['variant_name']}</td>"
            html_table += f"<td>{row['Similarity_Score_HTML']}</td>"
            html_table += "</tr>"
        html_table += "</tbody></table>"

        st.markdown(html_table, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
from datetime import datetime

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Inventory Discrepancy Prediction System",
    layout="wide"
)

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------

model = joblib.load("model/xgboost_inventory_model.pkl")
explainer = shap.TreeExplainer(model)
model_columns = joblib.load('model/model_columns.pkl')
encoder = joblib.load("model/encoder.pkl")

dataset = pd.read_csv(
    "dataset/jeddah_inventory_discrepancy_dataset.csv"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

.main {
    background-color: #f5f7fb;
}

h1 {
    color: #0b1b5e;
    font-weight: 800;
}

h2, h3 {
    color: #0b1b5e;
}

.stButton>button {
    background-color: #1565ff;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
    margin-top:0px
}

.result-box {
    background-color: #dbeafe;
    padding: 25px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    color: #0b4aa2;
}
.stSelectbox div[data-baseweb="select"] > div {
    min-height: 40px;
}

.stNumberInput input {
    height: 40px;
}

.stDateInput input {
    height: 40px;
}

.metric-card {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
}

/* KPI CARD */
[data-testid="stMetric"]{
    background:white;
    padding:20px;
    border-radius:18px;
    border-top:5px solid #111827;
    box-shadow:0px 2px 8px rgba(0,0,0,0.05);
    text-align:left;
}

/* KPI VALUE */
[data-testid="stMetricValue"]{
    font-size:38px;
    font-weight:700;
    color:#111827;
}

/* KPI LABEL */
[data-testid="stMetricLabel"]{
    font-size:16px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    """
    <h1 style='
        font-size:40px;
        color:#1d2340;
        font-weight:600;
        margin-bottom:0px;
    '>
    Inventory Discrepancy Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2 = st.tabs([
    "Prediction",
    "Analytics Dashboard"
])

# ===================================================
# PREDICTION PAGE
# ===================================================

with tab1:
    col1, col2 = st.columns(2)

    # --------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------

    with col1:
        shipment_date = st.date_input(
            "Shipment Date"
        )
        
        is_peak_season = st.selectbox(
            "Peak Season",
            ["Yes", "No"]
        )

        # If Peak Season = Yes
        if is_peak_season == "Yes":
            peak_season_type = st.selectbox(
                "Peak Season Type",
                [
                    "Ramadan",
                    "Hajj",
                    "Eid al-Adha",
                    "Eid al-Fitr",
                    "Back-to-School",
                    "Year-End Retail",
                    "Summer Surge"
                ]
            )

        # If Peak Season = No
        else:
            peak_season_type = "Normal Operations"
            st.text_input(
                "Peak Season Type",
                value=peak_season_type,
                disabled=True
            )

        shift = st.selectbox(
            "Shift",
            ["Morning", "Evening", "Night"]
        )
        
        warehouse_utilization = st.slider(
            "Warehouse Utilization Percent",
            0,
            100,
            75
        )


    # --------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------

    with col2:
        handling_complexity = st.selectbox(
            "Handling Complexity",
            ["Low", "Medium", "High"]
        )
        
        dwell_time = st.number_input(
            "Dwell Time Hours",
            min_value=1,
            max_value=300,
            value=25
        )

        temperature_sensitive = st.selectbox(
            "Temperature Sensitive",
            ["Yes", "No"]
        )
        
        scan_count = st.number_input(
            "Scan Count",
            min_value=1,
            max_value=15,
            value=1
        )

        worker_experience = st.slider(
            "Worker Experience Years",
            0,
            25,
            5
        )




    # ------------------------------------------------
    # DATE FEATURE EXTRACTION
    # ------------------------------------------------

    day_name = shipment_date.strftime("%A")

    is_weekend = (
        "Yes"
        if day_name in ["Saturday", "Friday"]
        else "No"
    )

    month = shipment_date.month

    # ------------------------------------------------
    # PREDICT BUTTON
    # ------------------------------------------------

    if st.button("Predict Inventory Discrepancy"):

        # ----------------------------------------
        # INPUT DATAFRAME
        # ----------------------------------------

        input_data = pd.DataFrame({
            'Month': [month],
            'Is_Peak_Season': [is_peak_season],
            'Peak_Season_Type': [peak_season_type],
            'Temperature_Sensitive': [
                temperature_sensitive
            ],
            'Scan_Count': [scan_count],
            'Dwell_Time_Hours': [dwell_time],
            'Warehouse_Utilization_Percent': [
                warehouse_utilization
            ],
            'Worker_Experience_Years': [
                worker_experience
            ],
            'Shift': [shift],
            'Handling_Complexity': [
                handling_complexity
            ],
            'Is_Weekend': [is_weekend]
        })

        # ----------------------------------------
        # REQUIRED EXTRA COLUMNS
        # ----------------------------------------

        input_data['Day_Name'] = day_name
        input_data['Cargo_Type'] = "General"
        input_data['Storage_Zone'] = "General Zone"
        input_data['Damage_History'] = "No"
        input_data['Missing_History'] = "No"
        input_data['Movement_Count'] = 5
        input_data['Manual_Handling_Count'] = 1
        # ----------------------------------------
        # COLUMN ORDER
        # ----------------------------------------

        final_columns = [
            'Month',
            'Day_Name',
            'Is_Weekend',
            'Is_Peak_Season',
            'Peak_Season_Type',
            'Cargo_Type',
            'Handling_Complexity',
            'Storage_Zone',
            'Shift',
            'Worker_Experience_Years',
            'Scan_Count',
            'Movement_Count',
            'Dwell_Time_Hours',
            'Temperature_Sensitive',
            'Damage_History',
            'Missing_History',
            'Warehouse_Utilization_Percent',
            'Manual_Handling_Count'
        ]

        input_data = input_data[final_columns]

        # ----------------------------------------
        # ENCODING
        # ----------------------------------------

        categorical_cols = input_data.select_dtypes(
            include='object'
        ).columns

        encoded = encoder.transform(
            input_data[categorical_cols]
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(
                categorical_cols
            )
        )

        numerical_df = input_data.drop(
            columns=categorical_cols
        )

        final_input = pd.concat(
            [numerical_df, encoded_df],
            axis=1
        )

        # ----------------------------------------
        # ALIGN COLUMNS
        # ----------------------------------------
        final_input = final_input.reindex(
            columns=model_columns,
            fill_value=0
        )

        # ----------------------------------------
        # PREDICTION
        # ----------------------------------------

        prediction = model.predict(final_input)[0]

        probability = model.predict_proba(
            final_input
        )[0][1]
        
        result_col, chart_col=st.columns([1,1])
        
        # ----------------------------------------
        # OUTPUT
        # ----------------------------------------
        with result_col:
            st.subheader("Prediction Results")

            if prediction == 1:

                st.markdown(f"""
                <div class="result-box">
                ⚠️ High Risk of Inventory Discrepancy

                <br><br>

                Probability:
                {probability:.2%}
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-box">
                ✅ Low Risk of Inventory Discrepancy

                <br><br>

                Probability:
                {probability:.2%}
                </div>
                """, unsafe_allow_html=True)
                
        with chart_col:
            st.subheader(
                "Feature Impact Analysis"
            )

            # ==========================================
            # SHAP VALUES
            # ==========================================

            shap_values = explainer.shap_values(
                final_input
            )

            # SHAP DataFrame
            shap_df = pd.DataFrame({

                'Feature': final_input.columns,

                'Impact': abs(shap_values[0])

            })

            # ==========================================
            # MAP ENCODED FEATURES
            # TO ORIGINAL FEATURES
            # ==========================================

            feature_mapping = {
                'Month': 'Month',
                'Worker_Experience_Years': 'Worker Experience',
                'Scan_Count': 'Scan Count',
                'Dwell_Time_Hours': 'Dwell Time',
                'Warehouse_Utilization_Percent': 'Warehouse Utilization',
                'Is_Weekend': 'Weekend',
                'Is_Peak_Season': 'Peak Season',
                'Peak_Season_Type': 'Peak Season Type',
                'Handling_Complexity': 'Handling Complexity',
                'Shift': 'Shift',
                'Temperature_Sensitive': 'Temperature Sensitive',

            }

            # ==========================================
            # MERGE ENCODED FEATURES
            # ==========================================

            cleaned_data = []

            for feature, impact in zip(
                shap_df['Feature'],
                shap_df['Impact']
            ):
                matched = False
                for key in feature_mapping:
                    if feature.startswith(key):
                        cleaned_data.append([
                            feature_mapping[key],
                            impact
                        ])
                        matched = True
                        break
            # Create DataFrame
            clean_df = pd.DataFrame(
                cleaned_data,
                columns=['Feature', 'Impact']
            )

            # Group Same Features
            clean_df = clean_df.groupby(
                'Feature'
            )['Impact'].sum().reset_index()

            # Sort
            clean_df = clean_df.sort_values(
                by='Impact',
                ascending=False
            )

            # Top 10
            clean_df = clean_df.head(10)

            # ==========================================
            # PLOT
            # ==========================================

            fig, ax = plt.subplots(
                figsize=(7,5)
            )

            ax.barh(
                clean_df['Feature'],
                clean_df['Impact']
            )

            ax.invert_yaxis()

            ax.set_xlabel("Impact")

            ax.set_title(
                "Top Features Affecting Prediction"
            )
            st.pyplot(fig)

# ===================================================
# ANALYTICS DASHBOARD
# ===================================================

with tab2:
    # =========================================================
    # DATASET DATE RANGE
    # =========================================================

    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset_start_date = dataset['Date'].min()
    dataset_end_date = dataset['Date'].max()
    
    # --------------------------------------------
    # FILTERS
    # --------------------------------------------
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    # ---------------------------------------------------------
    # FROM DATE
    # ---------------------------------------------------------
    with filter_col1:
        from_date = st.date_input(
            "From Date",
            value=dataset_start_date,
            min_value=dataset_start_date,
            max_value=dataset_end_date
    )

    # ---------------------------------------------------------
    # TO DATE
    # ---------------------------------------------------------
    with filter_col2:
        to_date = st.date_input(
            "To Date",
            value=dataset_end_date,
            min_value=dataset_start_date,
            max_value=dataset_end_date
        )

    with filter_col3:
        quick_filter = st.selectbox(
            "Quick Filter",[
                "All Time",
                "Last 30 Days",
                "Last 90 Days",
                "Last 6 Months",
                "Last 1 Year"]
        )
        
    # =========================================================
    # DATE FILTERING
    # =========================================================

    filtered_data = dataset.copy()
    # Convert Date Column
    filtered_data['Date'] = pd.to_datetime(
        filtered_data['Date']
    )
    # Apply Quick Filter
    today = filtered_data['Date'].max()
    if quick_filter == "Last 30 Days":
        from_date = today - pd.Timedelta(days=30)
    elif quick_filter == "Last 90 Days":
        from_date = today - pd.Timedelta(days=90)
    elif quick_filter == "Last 6 Months":
        from_date = today - pd.DateOffset(months=6)
    elif quick_filter == "Last 1 Year":
        from_date = today - pd.DateOffset(years=1)
    # Final Filtering
    filtered_data = filtered_data[
        (filtered_data['Date']>= pd.to_datetime(from_date))
        &
        (filtered_data['Date']<= pd.to_datetime(to_date))
    ]
    # =========================================================
    # KPI VALUES
    # =========================================================
    total_shipments = len(filtered_data)
    high_risk = filtered_data[
        filtered_data['Inventory_Discrepancy'] == 1
    ].shape[0]
    risk_rate = (
        high_risk / total_shipments
    ) * 100
    avg_dwell = filtered_data[
        'Dwell_Time_Hours'
    ].mean()
    avg_utilization = filtered_data[
        'Warehouse_Utilization_Percent'
    ].mean()

    # =========================================================
    # KPI CARDS
    # =========================================================
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric(
            label="Total Shipments",
            value=f"{total_shipments:,}"
        )

    with kpi2:
        st.metric(
            label="High Risk Rate",
            value=f"{round(risk_rate,1)}%"
        )

    with kpi3:
        st.metric(
            label="Discrepancy Count",
            value=f"{high_risk:,}"
        )

    with kpi4:
        st.metric(
            label="Avg Dwell Time",
            value=f"{round(avg_dwell,1)} hrs"
        )

    with kpi5:
        st.metric(
            label="Avg Utilization",
            value=f"{round(avg_utilization,1)}%"
        )

    # =========================================================
    # ROW 1
    # =========================================================
    chart1, chart2 = st.columns(2)

    # =========================================================
    # 1. MONTHLY DISCREPANCY TREND
    # =========================================================

    with chart1:
        monthly_risk = (
            filtered_data
            .groupby(filtered_data["Date"].dt.month_name().str[:3])
            ["Inventory_Discrepancy"]
            .mean()
            .reset_index()
        )

        month_order = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ]

        monthly_risk["Date"] = pd.Categorical(
            monthly_risk["Date"],
            categories=month_order,
            ordered=True
        )
        monthly_risk = monthly_risk.sort_values("Date")
        monthly_risk["Inventory_Discrepancy"] = (
            monthly_risk["Inventory_Discrepancy"] * 100
        )

        fig1 = px.line(
            monthly_risk,
            x="Date",
            y="Inventory_Discrepancy",
            markers=True
        )

        fig1.update_traces(
            mode="lines+markers+text",
            text=monthly_risk["Inventory_Discrepancy"].round(1),
            textposition="top center",           
            
            line=dict(
                color="#0b66c3",
                width=3
            ),
            marker=dict(
                size=8,
                color="#0b66c3"
            ),
            

            textfont=dict(
                size=12,
                color="#111827"
            )
        )

        fig1.update_layout(
            title="1. Monthly Discrepancy Trend",
            title_font_size=24,
            title_font_color="#111827",
            height=420,
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",

            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            ),

            xaxis=dict(
                title="Month",
                showgrid=False,
                tickfont=dict(size=14)
            ),

            yaxis=dict(
                title="Average Percentage of Discrepancy",
                gridcolor="#E5E7EB",
                range=[0, monthly_risk["Inventory_Discrepancy"].max() + 5],
                tickfont=dict(size=14)
            )
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # =========================================================
    # 2. SHIFT-WISE RISK ANALYSIS
    # =========================================================

    with chart2:

        # High discrepancy count by shift
        shift_risk = (
            filtered_data[filtered_data["Inventory_Discrepancy"] == 1]
            .groupby("Shift")
            .size()
            .reset_index(name="High_Risk_Count")
        )

        # Create color shades manually
        shift_risk["Color"] = [
            "#BFDBFE",   # Light Blue
            "#60A5FA",   # Medium Blue
            "#1D4ED8"    # Dark Blue
        ]

        # Create chart
        fig2 = px.bar(
            shift_risk,
            x="Shift",
            y="High_Risk_Count",
            text="High_Risk_Count"
        )

        # Apply different blue shades
        fig2.update_traces(
            marker_color=shift_risk["Color"],
            textposition="outside"
        )

        # Layout
        fig2.update_layout(
            title="2. Shift-wise Discrepancy Count",
            title_font_size=24,
            title_font_color="#111827",
            height=420,
            template="plotly_white",

            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',

            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),

            xaxis=dict(
                title="Shift",
                tickfont=dict(size=15)
            ),

            yaxis=dict(
                title="Discrepancy Count",
                gridcolor="#E5E7EB",
                tickfont=dict(size=10),

                range=[
                    0,
                    shift_risk["High_Risk_Count"].max() + 500
                ]
            ),

            showlegend=False
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={"displayModeBar": False}
        )
    # =========================================================
    # ROW 2
    # =========================================================

    chart3, chart4, chart5 = st.columns(3)

# =========================================================
# 3. WEEKEND VS WEEKDAY
# =========================================================

    with chart3:

        # Calculate discrepancy percentage
        weekend_data = (
            filtered_data
            .groupby("Is_Weekend")["Inventory_Discrepancy"]
            .mean()
            .reset_index()
        )

        # Convert to percentage
        weekend_data["Discrepancy_Percentage"] = (
            weekend_data["Inventory_Discrepancy"] * 100
        ).round(1)

        # Rename values
        weekend_data["Is_Weekend"] = weekend_data["Is_Weekend"].map({
            0: "Weekday",
            1: "Weekend",
            False: "Weekday",
            True: "Weekend",
            "No": "Weekday",
            "Yes": "Weekend"
        })

        # Create chart
        fig3 = px.bar(
            weekend_data,
            x="Is_Weekend",
            y="Discrepancy_Percentage",
            color="Is_Weekend",
            text="Discrepancy_Percentage",

            color_discrete_map={
                "Weekday": "#689ff8",
                "Weekend": "#374cee"
            }
        )

        # Text on top
        fig3.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            cliponaxis=False
        )

        # Layout
        fig3.update_layout(
            title="3. Discrepancy for Weekends <br> Vs Weekdays",
            title_font_size=24,
            title_font_color="#111827",
            height=420,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="",
            yaxis=dict(
                title="Discrepancy Rate (%)",
                autorange=False,
                range=[
                    0,
                    float(
                        weekend_data["Discrepancy_Percentage"].max()
                    ) + 5    
                ],
                gridcolor="#E5E7EB",
                tickfont=dict(size=14),
                fixedrange=True
            )
        )
        

        st.plotly_chart(
            fig3,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # =========================================================
    # 4. PEAK SEASON RISK DISTRIBUTION
    # =========================================================

    with chart4:

        peak_data = (
            filtered_data
            .groupby("Peak_Season_Type")["Inventory_Discrepancy"]
            .mean()
            .reset_index()
        )

        # Convert to percentage
        peak_data["Risk_Percentage"] = (
            peak_data["Inventory_Discrepancy"] * 100
        ).round(1)

        # Remove Normal Operations
        peak_data = peak_data[
            peak_data["Peak_Season_Type"] != "Normal Operations"
        ]

        # Top 5 seasons
        peak_data = (
            peak_data
            .sort_values(
                by="Risk_Percentage",
                ascending=False
            )
            .head(5)
        )

        fig4 = px.pie(

            peak_data,

            names="Peak_Season_Type",
            values="Risk_Percentage",

            hole=0.55,

            color_discrete_sequence=[
                "#1d4ed8",
                "#2563eb",
                "#3b82f6",
                "#60a5fa",
                "#93c5fd"
            ]
        )

        fig4.update_traces(

            textposition='outside',
            textinfo='percent+label',
            hoverinfo="skip",
            hovertemplate=None,
            pull=[0.03, 0, 0, 0, 0],
            marker=dict(
                line=dict(
                    color='white',
                    width=2
                )
            )
        )

        fig4.update_layout(

            title=dict(
                text="4. Peak Season Risk Distribution",
                x=0,
                font=dict(
                    size=22,
                    color="#111827"
                )
            ),

            height=420,

            template="plotly_white",

            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',

            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig4,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # =========================================================
    # 5. HANDLING COMPLEXITY
    # =========================================================

    with chart5:

        complexity_data = (
            filtered_data
            .groupby("Handling_Complexity")
            ["Inventory_Discrepancy"]
            .mean()
            .reset_index()
        )

        # Convert to percentage
        complexity_data["Inventory_Discrepancy"] = (
            complexity_data["Inventory_Discrepancy"] * 100
        ).round(1)

        # Sort properly
        complexity_order = ["Low", "Medium", "High"]

        complexity_data["Handling_Complexity"] = pd.Categorical(
            complexity_data["Handling_Complexity"],
            categories=complexity_order,
            ordered=True
        )

        complexity_data = complexity_data.sort_values(
            "Handling_Complexity"
        )

        # Create chart
        fig5 = px.bar(
            complexity_data,

            x="Inventory_Discrepancy",
            y="Handling_Complexity",

            orientation="h",

            text="Inventory_Discrepancy",

            title="5. Handling Complexity vs Risk",

            color="Handling_Complexity",

            color_discrete_map={
                "Low": "#bfdbfe",
                "Medium": "#60a5fa",
                "High": "#1d4ed8"
            }
        )

        # Text formatting
        fig5.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )

        # Layout
        fig5.update_layout(

            height=420,

            template="plotly_white",

            showlegend=False,

            paper_bgcolor="white",
            plot_bgcolor="white",

            title_font_size=24,
            title_font_color="#111827",

            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            ),

            xaxis=dict(
                title="Discrepancy Rate (%)",
                gridcolor="#E5E7EB",
                tickfont=dict(size=13),
                range=[0,35]
            ),

            yaxis=dict(
                title="",
                tickfont=dict(size=15)
            )
        )

        st.plotly_chart(
            fig5,
            use_container_width=True,
            config={"displayModeBar": False}
        )
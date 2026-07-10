"""Streamlit Dashboard for the Financial Crime Detection Engine.

This dashboard provides a compliance analyst with a view of the
pipeline's outputs, including high-level metrics, an alert queue
of high-risk accounts, and deep-dive capabilities for individual accounts.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.abspath('.'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.db import query
from src.features import build_features
from src.model import train_model, score_anomalies
from src.scoring import compute_risk_scores
from src.explain import generate_explanations

# --- Configuration ---
st.set_page_config(
    page_title="Financial Crime Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #1f77b4;
    }
    .metric-card.alert {
        border-top: 4px solid #d62728;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f77b4;
    }
    .metric-value.alert {
        color: #d62728;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading (Cached for performance) ---
@st.cache_data
def load_data():
    """Load and process the full pipeline."""
    # 1. Load Raw Data
    txns = query('SELECT * FROM transactions')
    accounts = query('SELECT * FROM accounts')
    
    # Ensure timestamps are parsed
    txns['timestamp'] = pd.to_datetime(txns['timestamp'])
    
    # 2. Features
    features = build_features(txns, accounts)
    
    # 3. Model & Score
    model, scaler = train_model(features)
    anomaly_df = score_anomalies(model, scaler, features)
    risk_df = compute_risk_scores(anomaly_df)
    
    # 4. Explanations
    explained_df = generate_explanations(risk_df, features)
    
    # 5. Merge account info
    full_risk_df = risk_df.merge(accounts[['account_id', 'customer_id', 'status']], on='account_id')
    
    return txns, accounts, features, full_risk_df, explained_df

# --- Application State ---
try:
    with st.spinner("Initializing Analytics Engine..."):
        txns, accounts, features, risk_df, explained_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you have run `python -m src.data_generation` to populate the database.")
    st.stop()


# --- Sidebar ---
st.sidebar.title("Command Center")
st.sidebar.markdown("---")
view_mode = st.sidebar.radio(
    "Navigation Menu",
    ["Executive Overview", "High-Risk Alert Queue", "Account Deep Dive"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Financial Crime Detection Engine v1.0")

# --- View 1: Executive Overview Dashboard ---
if view_mode == "Executive Overview":
    st.title("Executive Overview")
    st.markdown("Macro-level view of transaction velocity and risk distribution across the network.")
    
    # Top Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(txns):,}</div><div class="metric-label">Total Transactions</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(accounts):,}</div><div class="metric-label">Active Accounts</div></div>', unsafe_allow_html=True)
    with col3:
        high_risk = len(risk_df[risk_df['risk_band'] == 'High'])
        st.markdown(f'<div class="metric-card alert"><div class="metric-value alert">{high_risk}</div><div class="metric-label">High-Risk Alerts</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">R {txns["amount"].sum():,.0f}</div><div class="metric-label">Total Volume (ZAR)</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row 1: Time Series & Cross-Border
    r1c1, r1c2 = st.columns([2, 1])
    
    with r1c1:
        st.subheader("Transaction Volume Over Time")
        # Aggregate by week
        weekly_vol = txns.set_index('timestamp').resample('W')['amount'].sum().reset_index()
        fig_time = px.line(weekly_vol, x='timestamp', y='amount', 
                           labels={'timestamp': 'Date', 'amount': 'Volume (Rands)'})
        fig_time.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig_time, use_container_width=True)
        
    with r1c2:
        st.subheader("Cross-Border Exposure")
        cb_vol = txns.groupby('is_international')['amount'].sum().reset_index()
        cb_vol['is_international'] = cb_vol['is_international'].map({0: 'Domestic', 1: 'International'})
        fig_donut = px.pie(cb_vol, values='amount', names='is_international', hole=0.5,
                           color='is_international',
                           color_discrete_map={'Domestic': '#2ca02c', 'International': '#ff7f0e'})
        st.plotly_chart(fig_donut, use_container_width=True)

    # Charts Row 2: Risk Distributions
    r2c1, r2c2 = st.columns(2)
    
    with r2c1:
        st.subheader("Network Risk Score Distribution")
        fig_hist = px.histogram(
            risk_df, x="risk_score", color="risk_band",
            color_discrete_map={"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"},
            nbins=30, opacity=0.8,
            category_orders={"risk_band": ["Low", "Medium", "High"]}
        )
        fig_hist.update_layout(bargap=0.1, xaxis_title="Risk Score", yaxis_title="Number of Accounts")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with r2c2:
        st.subheader("Risk Band Breakdown")
        band_counts = risk_df['risk_band'].value_counts().reset_index()
        fig_pie = px.pie(
            band_counts, values='count', names='risk_band',
            color='risk_band',
            color_discrete_map={"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# --- View 2: Alert Queue ---
elif view_mode == "High-Risk Alert Queue":
    st.title("High-Risk Alert Queue")
    st.markdown("Prioritized list of accounts flagged by the anomaly detection ensemble.")
    
    if explained_df.empty:
        st.success("No high-risk accounts currently in the queue.")
    else:
        # Display as a dataframe
        display_df = explained_df.copy()
        
        # Merge total spend for context
        spend = txns.groupby('account_id')['amount'].sum().reset_index()
        display_df = display_df.merge(spend, on='account_id')
        
        display_df['account_id_short'] = display_df['account_id'].str[:8] + "..."
        display_df = display_df.sort_values('risk_score', ascending=False)
        
        # Move columns
        cols = ['account_id_short', 'risk_score', 'amount', 'explanation', 'account_id']
        display_df = display_df[cols]
        
        st.dataframe(
            display_df.drop(columns=['account_id']), 
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config={
                "account_id_short": "Account ID",
                "risk_score": st.column_config.ProgressColumn(
                    "Risk Score", help="0-100", format="%f", min_value=0, max_value=100,
                ),
                "amount": st.column_config.NumberColumn(
                    "Total Exposure (ZAR)", format="R %.2f"
                ),
                "explanation": "Primary Anomaly Drivers"
            }
        )

# --- View 3: Account Deep Dive ---
elif view_mode == "Account Deep Dive":
    st.title("Account Deep Dive & Forensics")
    
    # Account selector
    all_accounts = risk_df.sort_values('risk_score', ascending=False)['account_id'].tolist()
    selected_account = st.selectbox(
        "Search or Select Account ID to Investigate:", 
        options=all_accounts,
        format_func=lambda x: f"{x[:8]}... (Risk Score: {risk_df[risk_df['account_id']==x]['risk_score'].iloc[0]:.1f})"
    )
    
    st.markdown("---")
    
    if selected_account:
        acc_info = risk_df[risk_df['account_id'] == selected_account].iloc[0]
        acc_txns = txns[txns['account_id'] == selected_account].copy()
        acc_features = features.loc[selected_account]
        
        # Header
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Account ID", f"{selected_account[:8]}...")
        with col2:
            st.metric("Risk Score", f"{acc_info['risk_score']:.1f}/100")
        with col3:
            st.metric("Total Transactions", len(acc_txns))
        with col4:
            st.metric("Total Spend (ZAR)", f"R {acc_txns['amount'].sum():,.2f}")
            
        if acc_info['risk_band'] == 'High':
            st.error("**STATUS: HIGH RISK ALERT**")
            explanation = explained_df[explained_df['account_id'] == selected_account]['explanation'].iloc[0]
            st.warning(f"**Automated Forensics Report:** {explanation}")
        elif acc_info['risk_band'] == 'Medium':
            st.warning("**STATUS: MEDIUM RISK (MONITOR)**")
        else:
            st.success("**STATUS: LOW RISK (NORMAL)**")
            
        st.markdown("---")
        
        # Deep Dive Charts Row 1
        r1c1, r1c2 = st.columns([1, 1])
        
        with r1c1:
            st.subheader("Feature Profile (vs. Network Median)")
            st.markdown("Compares this account's behavior against normal network activity.")
            
            # Normalize features for radar chart
            cols_for_radar = ['transaction_count', 'avg_amount', 'max_daily_spend', 
                              'cross_border_ratio', 'night_ratio', 'merchant_diversity']
            
            # Get median of network
            network_median = features[cols_for_radar].median()
            
            # Calculate ratio of account to median (cap at 3x for visualization)
            acc_vals = acc_features[cols_for_radar]
            # Avoid division by zero
            safe_median = network_median.replace(0, 0.01)
            ratios = (acc_vals / safe_median).clip(lower=0, upper=3)
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                  r=[1]*len(cols_for_radar),
                  theta=cols_for_radar,
                  fill='toself',
                  name='Network Median (Baseline)',
                  line_color='rgba(44, 160, 44, 0.5)',
                  fillcolor='rgba(44, 160, 44, 0.2)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                  r=ratios.values,
                  theta=cols_for_radar,
                  fill='toself',
                  name='This Account',
                  line_color='rgba(214, 39, 40, 0.8)',
                  fillcolor='rgba(214, 39, 40, 0.4)'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=False, range=[0, 3])),
                showlegend=True,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with r1c2:
            st.subheader("Transaction Volume Timeline")
            st.markdown("Detailed transaction history highlighting large or international spikes.")
            
            acc_txns = acc_txns.sort_values('timestamp')
            fig_timeline = px.scatter(
                acc_txns, x="timestamp", y="amount", 
                color="is_international", size="amount",
                hover_data=["merchant_id", "country"],
                labels={"is_international": "Cross Border", "amount": "Amount (R)", "timestamp": "Date"},
                color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'}
            )
            fig_timeline.update_layout(xaxis_title="", yaxis_title="Amount (Rands)", margin=dict(t=20))
            st.plotly_chart(fig_timeline, use_container_width=True)
            
        # Transaction History Table
        st.subheader("Forensic Transaction Log")
        log_df = acc_txns[['timestamp', 'amount', 'merchant_id', 'country', 'is_international', 'transaction_type']].copy()
        log_df = log_df.sort_values('timestamp', ascending=False)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

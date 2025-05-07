import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Set page config
st.set_page_config(
    page_title="IPL Data Analysis Dashboard",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f7f7f7;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>IPL Data Analysis Dashboard (2008-2022)</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/1200px-Indian_Premier_League_Official_Logo.svg.png", width=200)
    st.markdown("## Upload & Configuration")
    
    # Dataset upload
    uploaded_file = st.file_uploader("Upload IPL Dataset (CSV)", type=["csv"])
    
    # Sample data option
    use_sample_data = st.checkbox("Use sample data instead", value=False)
    
    # Visualization options
    st.markdown("## Visualization Settings")
    color_theme = st.selectbox(
        "Color Theme",
        ["viridis", "plasma", "inferno", "magma", "cividis", "Paired", "Set1", "coolwarm", "Blues"]
    )
    
    # Analysis filters
    st.markdown("## Analysis Filters")
    
    # Season filter will be populated after data is loaded
    seasons_filter = st.empty()
    
    # Team filter will be populated after data is loaded
    teams_filter = st.empty()
    
    # Player filter will be populated after data is loaded
    players_filter = st.empty()
    
    # Advanced options
    st.markdown("## Advanced Options")
    remove_outliers_option = st.checkbox("Remove Outliers", value=False)
    normalize_data = st.checkbox("Normalize Data", value=False)
    
    # Export options
    st.markdown("## Export Options")
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
    
    if st.button("Export Processed Data"):
        if 'df' in locals() or 'df' in globals():
            st.success("Data exported successfully! (Demo)")

# Helper Functions
@st.cache_data
def load_sample_data():
    # This is a mock sample - in a real app, you would include a small sample IPL dataset
    return pd.read_csv('IPL Dataset 2008-2022.csv')

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def get_summary_stats(df):
    summary = df.describe().T
    summary['IQR'] = summary['75%'] - summary['25%']
    return summary

# Data Processing
df = None

# Load data
if uploaded_file is not None:
    try:
        with st.spinner('Loading and processing your data...'):
            df = pd.read_csv(uploaded_file)
            st.session_state['data_loaded'] = True
            st.success(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
elif use_sample_data:
    try:
        with st.spinner('Loading sample data...'):
            df = load_sample_data()
            st.session_state['data_loaded'] = True
            st.success("Sample IPL dataset loaded successfully.")
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
else:
    st.info("Please upload an IPL dataset or use sample data to get started.")
    st.session_state['data_loaded'] = False

# Only proceed if data is loaded
if df is not None:
    # Populate filters based on data
    if 'Season' in df.columns:
        seasons = sorted(df['Season'].unique())
        selected_seasons = seasons_filter.multiselect("Select Seasons", seasons, default=seasons)
        df = df[df['Season'].isin(selected_seasons)]
    
    if 'BattingTeam' in df.columns:
        teams = sorted(df['BattingTeam'].unique())
        selected_teams = teams_filter.multiselect("Select Teams", teams, default=teams)
        
    if 'batter' in df.columns:
        all_players = sorted(df['batter'].unique())
        # Get top 20 players by runs for the default selection
        if len(all_players) > 20:
            top_players = df.groupby('batter')['batsman_run'].sum().nlargest(20).index.tolist()
            selected_players = players_filter.multiselect("Select Players", all_players, default=top_players)
    
    # Apply outlier removal if selected
    if remove_outliers_option:
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        columns_to_clean = st.multiselect(
            "Select columns for outlier removal",
            numerical_columns,
            default=[]
        )
        
        if columns_to_clean:
            original_shape = df.shape[0]
            for col in columns_to_clean:
                df = remove_outliers(df, col)
            st.info(f"Removed {original_shape - df.shape[0]} outliers ({((original_shape - df.shape[0])/original_shape)*100:.2f}%).")

    # Dashboard Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🏏 Team Analysis", 
        "👨‍💼 Player Analysis", 
        "🔍 Advanced Metrics",
        "📈 Predictive Insights"
    ])

    # Tab 1: Overview
    with tab1:
        st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
        
        # Dataset info in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Matches", df['ID'].nunique() if 'ID' in df.columns else "N/A")
        with col2:
            st.metric("Total Runs", f"{df['batsman_run'].sum():,}" if 'batsman_run' in df.columns else "N/A")
        with col3:
            st.metric("Total Wickets", f"{df['isWicketDelivery'].sum():,}" if 'isWicketDelivery' in df.columns else "N/A")
            
        # Summary Statistics (Expandable)
        with st.expander("View Summary Statistics"):
            summary_stats = get_summary_stats(df.select_dtypes(include=['number']))
            st.dataframe(summary_stats)
        
        # Missing values analysis
        with st.expander("Missing Values Analysis"):
            missing_values = df.isnull().sum()
            missing_percent = (missing_values / len(df)) * 100
            missing_data = pd.concat([missing_values, missing_percent], axis=1)
            missing_data.columns = ['Missing Values', 'Percentage']
            missing_data = missing_data[missing_data['Missing Values'] > 0]
            
            if not missing_data.empty:
                st.dataframe(missing_data)
                
                # Plot missing values
                fig = px.bar(
                    x=missing_data.index,
                    y=missing_data['Percentage'],
                    title='Missing Values Percentage by Column',
                    labels={'x': 'Columns', 'y': 'Missing Percentage (%)'},
                    color=missing_data['Percentage'],
                    color_continuous_scale=color_theme
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values in the dataset!")
        
        # Overview Visualizations
        st.markdown("<h2 class='sub-header'>Key Insights</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if {"BattingTeam", "total_run"}.issubset(df.columns):
                # Create total runs by team chart
                total_runs_per_team = df.groupby("BattingTeam")["total_run"].sum().reset_index()
                total_runs_per_team.columns = ["Team", "Total_Runs"]
                
                fig = px.bar(
                    total_runs_per_team,
                    x="Total_Runs",
                    y="Team",
                    orientation='h',
                    title="Total Runs by Team",
                    color="Total_Runs",
                    color_continuous_scale=color_theme
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if {"overs", "total_run"}.issubset(df.columns):
                # Create runs by over chart
                over_runs = df.groupby("overs")["total_run"].sum().reset_index()
                
                fig = px.line(
                    over_runs,
                    x="overs",
                    y="total_run",
                    title="Runs Scored by Over",
                    markers=True,
                    line_shape="spline",
                    color_discrete_sequence=[px.colors.qualitative.Set1[0]]
                )
                fig.update_layout(xaxis_title="Over Number", yaxis_title="Total Runs")
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Matrix
        if df.select_dtypes(include=['number']).shape[1] > 1:
            st.markdown("<h3>Correlation Analysis</h3>", unsafe_allow_html=True)
            
            numerical_df = df.select_dtypes(include=["number"])
            correlation_matrix = numerical_df.corr()
            
            # Filter columns with meaningful correlations
            if st.checkbox("Show only strong correlations"):
                threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.5, 0.05)
                mask = np.abs(correlation_matrix) < threshold
                correlation_matrix_filtered = correlation_matrix.copy()
                correlation_matrix_filtered[mask] = np.nan
                correlation_matrix = correlation_matrix_filtered
            
            # Plot correlation heatmap
            fig = px.imshow(
                correlation_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale=color_theme
            )
            fig.update_layout(
                title="Correlation Heatmap of Numerical Features",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Team Analysis
    with tab2:
        st.markdown("<h2 class='sub-header'>Team Performance Analysis</h2>", unsafe_allow_html=True)
        
        if {"BattingTeam", "total_run", "ID"}.issubset(df.columns):
            # Team selection for detailed analysis
            if 'selected_teams' in locals():
                team_for_analysis = st.selectbox("Select Team for Detailed Analysis", selected_teams)
                
                # Team metrics
                team_data = df[df["BattingTeam"] == team_for_analysis]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_runs = team_data["total_run"].sum()
                    st.metric("Total Runs", f"{total_runs:,}")
                
                with col2:
                    total_matches = team_data["ID"].nunique()
                    st.metric("Matches Played", total_matches)
                
                with col3:
                    if "isWicketDelivery" in df.columns:
                        wickets_lost = team_data["isWicketDelivery"].sum()
                        st.metric("Wickets Lost", wickets_lost)
                
                with col4:
                    if total_matches > 0:
                        avg_runs_per_match = total_runs / total_matches
                        st.metric("Avg. Runs/Match", f"{avg_runs_per_match:.2f}")
                
                # Team performance over overs
                if "overs" in df.columns:
                    st.markdown("<h3>Run Rate Analysis</h3>", unsafe_allow_html=True)
                    
                    team_over_runs = team_data.groupby("overs")["total_run"].sum().reset_index()
                    overall_over_runs = df.groupby("overs")["total_run"].sum().reset_index()
                    overall_over_runs.columns = ["overs", "overall_total_run"]
                    
                    comparison_df = team_over_runs.merge(overall_over_runs, on="overs")
                    comparison_df["overall_avg"] = comparison_df["overall_total_run"] / df["BattingTeam"].nunique()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=comparison_df["overs"],
                        y=comparison_df["total_run"],
                        mode="lines+markers",
                        name=f"{team_for_analysis} Runs",
                        line=dict(color=px.colors.qualitative.Set1[0], width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=comparison_df["overs"],
                        y=comparison_df["overall_avg"],
                        mode="lines+markers",
                        name="League Average",
                        line=dict(color=px.colors.qualitative.Set1[1], width=2, dash="dash")
                    ))
                    fig.update_layout(
                        title=f"{team_for_analysis} Run Rate Compared to League Average",
                        xaxis_title="Over Number",
                        yaxis_title="Runs Scored",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Opponent analysis
                if "BowlingTeam" in df.columns:
                    st.markdown("<h3>Performance Against Opponents</h3>", unsafe_allow_html=True)
                    
                    vs_opponents = df[df["BattingTeam"] == team_for_analysis].groupby("BowlingTeam")["total_run"].agg(
                        ["sum", "mean", "count"]
                    ).reset_index()
                    vs_opponents.columns = ["Opponent", "Total_Runs", "Avg_Runs", "Innings"]
                    
                    # Create a two-column layout for the charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(
                            vs_opponents,
                            x="Opponent",
                            y="Avg_Runs",
                            title=f"{team_for_analysis}'s Average Runs vs Each Opponent",
                            color="Avg_Runs",
                            color_continuous_scale=color_theme
                        )
                        fig.update_layout(xaxis_title="Opponent Team", yaxis_title="Average Runs")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            vs_opponents,
                            x="Opponent",
                            y="Innings",
                            title=f"Number of Innings Against Each Opponent",
                            color="Innings",
                            color_continuous_scale="Blues"
                        )
                        fig.update_layout(xaxis_title="Opponent Team", yaxis_title="Number of Innings")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Win-loss analysis if available
                if {"match_winner"}.issubset(df.columns):
                    st.markdown("<h3>Win-Loss Analysis</h3>", unsafe_allow_html=True)
                    # Add win-loss charts here
                else:
                    # Create a simulated win-loss analysis based on highest runs
                    st.markdown("<h3>Simulated Win-Loss Analysis (Based on Highest Run Total)</h3>", unsafe_allow_html=True)
                    
                    match_runs = df.groupby(["ID", "BattingTeam"])["total_run"].sum().reset_index()
                    match_winners = match_runs.loc[match_runs.groupby("ID")["total_run"].idxmax()]
                    
                    team_wins = match_winners["BattingTeam"].value_counts().reset_index()
                    team_wins.columns = ["Team", "Wins"]
                    
                    total_matches_per_team = df.groupby("BattingTeam")["ID"].nunique().reset_index()
                    total_matches_per_team.columns = ["Team", "Total_Matches"]
                    
                    win_loss = team_wins.merge(total_matches_per_team, on="Team")
                    win_loss["Losses"] = win_loss["Total_Matches"] - win_loss["Wins"]
                    win_loss["Win_Rate"] = (win_loss["Wins"] / win_loss["Total_Matches"]) * 100
                    
                    if team_for_analysis in win_loss["Team"].values:
                        team_win_loss = win_loss[win_loss["Team"] == team_for_analysis]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.pie(
                                names=["Wins", "Losses"],
                                values=[team_win_loss["Wins"].values[0], team_win_loss["Losses"].values[0]],
                                title=f"{team_for_analysis}'s Win-Loss Record",
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.bar(
                                win_loss,
                                x="Team",
                                y="Win_Rate",
                                title="Win Rate Comparison",
                                color="Win_Rate",
                                color_continuous_scale=color_theme
                            )
                            fig.add_hline(y=50, line_dash="dash", line_color="gray")
                            fig.update_layout(xaxis_title="Team", yaxis_title="Win Rate (%)")
                            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Player Analysis
    with tab3:
        st.markdown("<h2 class='sub-header'>Player Performance Analysis</h2>", unsafe_allow_html=True)
        
        if {"batter", "batsman_run"}.issubset(df.columns):
            # Top run scorers
            player_runs = df.groupby("batter")["batsman_run"].sum().reset_index()
            player_runs = player_runs.sort_values("batsman_run", ascending=False).head(20)
            
            fig = px.bar(
                player_runs,
                x="batter",
                y="batsman_run",
                title="Top 20 Run Scorers in IPL",
                color="batsman_run",
                color_continuous_scale=color_theme
            )
            fig.update_layout(xaxis_title="Player", yaxis_title="Total Runs")
            st.plotly_chart(fig, use_container_width=True)
            
            # Player comparison tool
            st.markdown("<h3>Player Comparison Tool</h3>", unsafe_allow_html=True)
            
            if 'selected_players' in locals() and len(selected_players) > 0:
                players_to_compare = st.multiselect(
                    "Select players to compare",
                    options=selected_players,
                    default=selected_players[:2] if len(selected_players) >= 2 else selected_players
                )
                
                if len(players_to_compare) > 0:
                    # Calculate metrics for selected players
                    player_metrics = []
                    
                    for player in players_to_compare:
                        player_data = df[df["batter"] == player]
                        
                        # Calculate basic metrics
                        runs = player_data["batsman_run"].sum()
                        balls_faced = player_data.shape[0]
                        
                        # Strike rate
                        strike_rate = (runs / balls_faced * 100) if balls_faced > 0 else 0
                        
                        # Boundaries
                        fours = player_data[player_data["batsman_run"] == 4].shape[0]
                        sixes = player_data[player_data["batsman_run"] == 6].shape[0]
                        
                        # Batting position (if available)
                        avg_position = player_data["batting_position"].mean() if "batting_position" in player_data.columns else None
                        
                        player_metrics.append({
                            "Player": player,
                            "Total_Runs": runs,
                            "Balls_Faced": balls_faced,
                            "Strike_Rate": strike_rate,
                            "Fours": fours,
                            "Sixes": sixes,
                            "Avg_Position": avg_position
                        })
                    
                    player_metrics_df = pd.DataFrame(player_metrics)
                    
                    # Display player metrics comparison
                    st.dataframe(player_metrics_df)
                    
                    # Create radar chart for player comparison
                    if len(players_to_compare) > 1:
                        # Get metrics for radar chart (normalized)
                        radar_metrics = ["Total_Runs", "Strike_Rate", "Fours", "Sixes"]
                        radar_df = player_metrics_df[["Player"] + radar_metrics].copy()
                        
                        # Normalize metrics
                        for metric in radar_metrics:
                            max_val = radar_df[metric].max()
                            if max_val > 0:  # Avoid division by zero
                                radar_df[f"{metric}_norm"] = radar_df[metric] / max_val
                        
                        fig = go.Figure()
                        
                        for i, player in enumerate(players_to_compare):
                            player_data = radar_df[radar_df["Player"] == player]
                            
                            fig.add_trace(go.Scatterpolar(
                                r=[
                                    player_data[f"{metric}_norm"].values[0] 
                                    for metric in radar_metrics
                                ],
                                theta=radar_metrics,
                                fill='toself',
                                name=player,
                                line_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="Player Comparison (Normalized Metrics)",
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Individual player detailed analysis
                    if len(players_to_compare) == 1:
                        st.markdown(f"<h3>Detailed Analysis: {players_to_compare[0]}</h3>", unsafe_allow_html=True)
                        player_data = df[df["batter"] == players_to_compare[0]]
                        
                        # Performance against different teams
                        if "BowlingTeam" in player_data.columns:
                            vs_teams = player_data.groupby("BowlingTeam")["batsman_run"].agg(
                                ["sum", "count", "mean"]
                            ).reset_index()
                            vs_teams.columns = ["Team", "Runs", "Balls", "Avg_Run_per_Ball"]
                            vs_teams["Strike_Rate"] = vs_teams["Avg_Run_per_Ball"] * 100
                            
                            fig = px.bar(
                                vs_teams,
                                x="Team",
                                y="Strike_Rate",
                                title=f"{players_to_compare[0]}'s Strike Rate Against Different Teams",
                                color="Runs",
                                color_continuous_scale=color_theme,
                                hover_data=["Runs", "Balls"]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance by over phase
                        if "overs" in player_data.columns:
                            # Define over phases
                            player_data["Phase"] = pd.cut(
                                player_data["overs"],
                                bins=[0, 6, 15, 20],
                                labels=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
                            )
                            
                            phase_performance = player_data.groupby("Phase")["batsman_run"].agg(
                                ["sum", "count", "mean"]
                            ).reset_index()
                            phase_performance.columns = ["Phase", "Runs", "Balls", "Avg_Run_per_Ball"]
                            phase_performance["Strike_Rate"] = phase_performance["Avg_Run_per_Ball"] * 100
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.pie(
                                    phase_performance,
                                    values="Runs",
                                    names="Phase",
                                    title=f"{players_to_compare[0]}'s Runs by Match Phase",
                                    color_discrete_sequence=px.colors.qualitative.Set2
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = px.bar(
                                    phase_performance,
                                    x="Phase",
                                    y="Strike_Rate",
                                    title=f"{players_to_compare[0]}'s Strike Rate by Match Phase",
                                    color="Phase",
                                    text="Balls",
                                    color_discrete_sequence=px.colors.qualitative.Set2
                                )
                                fig.update_layout(yaxis_title="Strike Rate")
                                st.plotly_chart(fig, use_container_width=True)
            
            # Bowler analysis (if data available)
            if {"bowler", "isWicketDelivery"}.issubset(df.columns):
                st.markdown("<h3>Bowling Analysis</h3>", unsafe_allow_html=True)
                
                bowler_data = df.groupby("bowler").agg({
                    "isWicketDelivery": "sum",
                    "total_run": "sum",
                    "ballnumber": "count"
                }).reset_index()
                
                bowler_data.columns = ["Bowler", "Wickets", "Runs_Conceded", "Balls_Bowled"]
                bowler_data["Economy"] = (bowler_data["Runs_Conceded"] / bowler_data["Balls_Bowled"]) * 6
                bowler_data = bowler_data.sort_values("Wickets", ascending=False)
                
                # Filter for bowlers with minimum deliveries
                min_deliveries = st.slider("Minimum Deliveries Bowled", 1, 1000, 100)
                filtered_bowlers = bowler_data[bowler_data["Balls_Bowled"] >= min_deliveries]
                
                # Top wicket-takers
                st.subheader("Top Wicket Takers")
                top_wicket_takers = filtered_bowlers.head(10)
                
                fig = px.bar(
                    top_wicket_takers,
                    x="Bowler",
                    y="Wickets",
                    color="Economy",
                    color_continuous_scale="RdYlGn_r",  # Reversed scale (lower economy is better)
                    hover_data=["Balls_Bowled", "Runs_Conceded"],
                    title="Top 10 Wicket Takers (with Economy Rate)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Best economy rates
                st.subheader("Best Economy Rates")
                best_economy = filtered_bowlers.sort_values("Economy").head(10)
                
                fig = px.bar(
                    best_economy,
                    x="Bowler",
                    y="Economy",
                    color="Wickets",
                    color_continuous_scale="Blues",
                    hover_data=["Balls_Bowled", "Runs_Conceded"],
                    title="Top 10 Bowlers by Economy Rate (with Wickets)"
                )
                fig.update_layout(yaxis_title="Economy Rate (Runs per Over)")
                st.plotly_chart(fig, use_container_width=True)

                # Tab 4: Advanced Metrics (continued)

                    # For the "Death Overs Performance (Batsmen)" section:
                # Filter for death overs
                death_overs_df = df[df["overs"] >= 16]

                # Calculate clutch batting stats
                clutch_batting = death_overs_df.groupby("batter").agg({
                    "batsman_run": "sum",
                    "ballnumber": "count"  # Total balls faced
                }).reset_index()
            clutch_batting.columns = ["batter", "clutch_runs", "balls_faced"]
            
            # Calculate strike rate
            clutch_batting["strike_rate"] = (clutch_batting["clutch_runs"] / clutch_batting["balls_faced"]) * 100
            
            # Normalize and calculate clutch score
            clutch_batting["Clutch_Score"] = (
                0.7 * (clutch_batting["clutch_runs"] / clutch_batting["clutch_runs"].max()) +
                0.3 * (clutch_batting["strike_rate"] / clutch_batting["strike_rate"].max())
            )
            
            # Filter for minimum balls faced
            min_balls_death = st.slider("Minimum Balls Faced in Death Overs", 10, 100, 20)
            filtered_clutch_batting = clutch_batting[clutch_batting["balls_faced"] >= min_balls_death]
            filtered_clutch_batting = filtered_clutch_batting.sort_values("Clutch_Score", ascending=False).head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    filtered_clutch_batting,
                    x="batter",
                    y="Clutch_Score",
                    color="strike_rate",
                    hover_data=["clutch_runs", "balls_faced"],
                    title="Top 10 Death Over Specialists (Batsmen)",
                    color_continuous_scale=color_theme
                )
                fig.update_layout(xaxis_title="Batsman", yaxis_title="Clutch Score")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    filtered_clutch_batting,
                    x="balls_faced",
                    y="clutch_runs",
                    size="strike_rate",
                    color="Clutch_Score",
                    hover_name="batter",
                    title="Death Overs: Runs vs Balls Faced",
                    color_continuous_scale=color_theme
                )
                fig.update_layout(xaxis_title="Balls Faced", yaxis_title="Total Runs")
                st.plotly_chart(fig, use_container_width=True)
        
        # Death overs performance (bowlers)
        if {"overs", "bowler", "isWicketDelivery", "total_run"}.issubset(df.columns):
            st.markdown("<h3>Death Overs Performance (Bowlers)</h3>", unsafe_allow_html=True)
            
            # Filter for death overs
            death_overs_df = df[df["overs"] >= 16]
            
            clutch_bowling = death_overs_df.groupby("bowler").agg({
                "total_run": "sum",
                "isWicketDelivery": "sum",
                "ballnumber": "count"  # Total balls bowled
            }).reset_index()
            
            clutch_bowling.columns = ["bowler", "runs_conceded", "wickets", "balls_bowled"]
            
            # Calculate economy rate
            clutch_bowling["economy"] = (clutch_bowling["runs_conceded"] / clutch_bowling["balls_bowled"]) * 6
            
            # Calculate wickets per over
            clutch_bowling["wickets_per_over"] = (clutch_bowling["wickets"] / clutch_bowling["balls_bowled"]) * 6
            
            # Normalize metrics (lower economy = better, higher wickets = better)
            clutch_bowling["Norm_Wickets"] = clutch_bowling["wickets"] / clutch_bowling["wickets"].max()
            clutch_bowling["Norm_Economy"] = 1 - (clutch_bowling["economy"] / clutch_bowling["economy"].max())  # Inverted
            
            # Clutch Score: Weighted average
            clutch_bowling["Clutch_Score"] = (
                0.6 * clutch_bowling["Norm_Wickets"] +
                0.4 * clutch_bowling["Norm_Economy"]
            )
            
            # Filter for minimum balls bowled
            min_balls_death_bowl = st.slider("Minimum Balls Bowled in Death Overs", 10, 100, 20)
            filtered_clutch_bowling = clutch_bowling[clutch_bowling["balls_bowled"] >= min_balls_death_bowl]
            filtered_clutch_bowling = filtered_clutch_bowling.sort_values("Clutch_Score", ascending=False).head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    filtered_clutch_bowling,
                    x="bowler",
                    y="Clutch_Score",
                    color="economy",
                    color_continuous_scale="RdYlGn_r",  # Reversed (lower economy is better)
                    hover_data=["wickets", "balls_bowled"],
                    title="Top 10 Death Over Specialists (Bowlers)"
                )
                fig.update_layout(xaxis_title="Bowler", yaxis_title="Clutch Score")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    filtered_clutch_bowling,
                    x="economy",
                    y="wickets_per_over",
                    size="balls_bowled",
                    color="Clutch_Score",
                    hover_name="bowler",
                    title="Death Overs: Economy vs Wicket Rate",
                    color_continuous_scale=color_theme
                )
                fig.update_layout(
                    xaxis_title="Economy Rate", 
                    yaxis_title="Wickets per Over",
                    xaxis=dict(autorange="reversed")  # Lower economy (left) is better
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Match Momentum Analysis
        st.markdown("<h3>Match Momentum Analysis</h3>", unsafe_allow_html=True)
        
        if {"ID", "overs", "BattingTeam", "total_run"}.issubset(df.columns):
            # Select a specific match for analysis
            match_ids = sorted(df["ID"].unique())
            
            if match_ids:
                selected_match = st.selectbox("Select Match for Momentum Analysis", match_ids)
                match_data = df[df["ID"] == selected_match]
                
                # Get teams in the match
                teams = match_data["BattingTeam"].unique()
                
                if len(teams) == 2:
                    # Calculate cumulative runs per over for each team
                    team1_data = match_data[match_data["BattingTeam"] == teams[0]]
                    team2_data = match_data[match_data["BattingTeam"] == teams[1]]
                    
                    team1_overs = team1_data.groupby("overs")["total_run"].sum().reset_index()
                    team2_overs = team2_data.groupby("overs")["total_run"].sum().reset_index()
                    
                    team1_overs["cumulative_runs"] = team1_overs["total_run"].cumsum()
                    team2_overs["cumulative_runs"] = team2_overs["total_run"].cumsum()
                    
                    # Create momentum plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=team1_overs["overs"],
                        y=team1_overs["cumulative_runs"],
                        mode="lines+markers",
                        name=teams[0],
                        line=dict(color=px.colors.qualitative.Set1[0], width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=team2_overs["overs"],
                        y=team2_overs["cumulative_runs"],
                        mode="lines+markers",
                        name=teams[1],
                        line=dict(color=px.colors.qualitative.Set1[1], width=3)
                    ))
                    
                    # Add run rate
                    team1_overs["run_rate"] = team1_overs["cumulative_runs"] / (team1_overs["overs"] + 1)  # +1 to avoid division by zero
                    team2_overs["run_rate"] = team2_overs["cumulative_runs"] / (team2_overs["overs"] + 1)
                    
                    # Create a secondary y-axis for run rate
                    fig.add_trace(go.Scatter(
                        x=team1_overs["overs"],
                        y=team1_overs["run_rate"],
                        mode="lines",
                        name=f"{teams[0]} Run Rate",
                        line=dict(color=px.colors.qualitative.Set1[0], width=1, dash="dot"),
                        yaxis="y2"
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=team2_overs["overs"],
                        y=team2_overs["run_rate"],
                        mode="lines",
                        name=f"{teams[1]} Run Rate",
                        line=dict(color=px.colors.qualitative.Set1[1], width=1, dash="dot"),
                        yaxis="y2"
                    ))
                    
                    fig.update_layout(
                        title=f"Match Momentum: {teams[0]} vs {teams[1]}",
                        xaxis_title="Overs",
                        yaxis_title="Cumulative Runs",
                        yaxis2=dict(
                            title="Run Rate",
                            overlaying="y",
                            side="right"
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show key moments in the match
                    if "isWicketDelivery" in df.columns:
                        st.subheader("Key Moments in the Match")
                        
                        # Wickets
                        wickets = match_data[match_data["isWicketDelivery"] == 1]
                        
                        # Boundaries
                        boundaries = match_data[(match_data["batsman_run"] == 4) | (match_data["batsman_run"] == 6)]
                        
                        # Create tabs for key moments
                        tab1, tab2 = st.tabs(["⚾ Wickets", "🏏 Boundaries"])
                        
                        with tab1:
                            if not wickets.empty:
                                st.dataframe(
                                    wickets[["overs", "BattingTeam", "batter", "bowler"]].sort_values("overs"),
                                    use_container_width=True
                                )
                            else:
                                st.info("No wicket data available for this match.")
                        
                        with tab2:
                            if not boundaries.empty:
                                boundaries_df = boundaries[["overs", "BattingTeam", "batter", "bowler", "batsman_run"]]
                                boundaries_df["Type"] = boundaries_df["batsman_run"].apply(lambda x: "Six" if x == 6 else "Four")
                                st.dataframe(
                                    boundaries_df[["overs", "BattingTeam", "batter", "bowler", "Type"]].sort_values("overs"),
                                    use_container_width=True
                                )
                            else:
                                st.info("No boundary data available for this match.")
                else:
                    st.error("This match doesn't have exactly two teams in the data.")
        
        # Partnership Analysis
        st.markdown("<h3>Partnership Analysis</h3>", unsafe_allow_html=True)
        
        if {"batter", "non_striker", "batsman_run"}.issubset(df.columns):
            # Calculate partnership stats
            partnerships = df.groupby(["batter", "non_striker"])["batsman_run"].agg(
                ["sum", "count"]
            ).reset_index()
            
            partnerships.columns = ["Batter1", "Batter2", "Runs", "Balls"]
            partnerships["Average"] = partnerships["Runs"] / partnerships["Balls"]
            partnerships["Strike_Rate"] = partnerships["Average"] * 100
            
            # Filter for minimum balls
            min_partnership_balls = st.slider("Minimum Partnership Balls", 10, 200, 30)
            filtered_partnerships = partnerships[partnerships["Balls"] >= min_partnership_balls]
            
            # Sort by runs
            top_partnerships = filtered_partnerships.sort_values("Runs", ascending=False).head(15)
            
            fig = px.scatter(
                top_partnerships,
                x="Balls",
                y="Runs",
                size="Strike_Rate",
                color="Strike_Rate",
                hover_name=top_partnerships.apply(lambda x: f"{x['Batter1']} & {x['Batter2']}", axis=1),
                title="Top Batting Partnerships",
                color_continuous_scale=color_theme
            )
            fig.update_layout(xaxis_title="Balls Faced", yaxis_title="Partnership Runs")
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to view detailed partnership table
            if st.checkbox("View Detailed Partnership Table"):
                st.dataframe(top_partnerships, use_container_width=True)

    # Tab 5: Predictive Insights
    with tab5:
        st.markdown("<h2 class='sub-header'>Predictive Analysis & Insights</h2>", unsafe_allow_html=True)
        
        st.info("This section provides predictive insights based on historical IPL data. These predictions are for educational purposes only.")
        
        # Win Probability Model
        st.markdown("<h3>Win Probability Predictor</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input parameters for win probability
            batting_team = st.selectbox("Batting Team", options=df["BattingTeam"].unique() if "BattingTeam" in df.columns else ["Team A", "Team B"])
            bowling_team = st.selectbox("Bowling Team", options=[team for team in df["BattingTeam"].unique() if team != batting_team] if "BattingTeam" in df.columns else ["Team B", "Team A"])
            
            current_score = st.number_input("Current Score", min_value=0, max_value=300, value=100)
            wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=3)
            overs_completed = st.slider("Overs Completed", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
            target = st.number_input("Target Score (0 if batting first)", min_value=0, max_value=300, value=180)
            
        with col2:
            # Display a simple win probability visualization
            if st.button("Calculate Win Probability"):
                with st.spinner("Calculating win probability..."):
                    # Simple win probability model (in a real app, this would be a trained ML model)
                    if target > 0:  # Batting second
                        required_runs = target - current_score
                        required_run_rate = required_runs / max(0.1, (20 - overs_completed))
                        wickets_remaining = 10 - wickets_lost
                        
                        # Base probability
                        if required_run_rate > 15:
                            prob = 0.1
                        elif required_run_rate > 12:
                            prob = 0.3
                        elif required_run_rate > 10:
                            prob = 0.5
                        elif required_run_rate > 8:
                            prob = 0.7
                        else:
                            prob = 0.9
                        
                        # Adjust for wickets
                        prob *= (0.5 + (wickets_remaining / 20))
                        
                        # Clamp probability
                        win_prob = max(0.05, min(0.95, prob))
                    else:  # Batting first
                        current_run_rate = current_score / max(0.1, overs_completed)
                        projected_score = current_score + (current_run_rate * (20 - overs_completed))
                        
                        # Simple model based on projected score
                        if projected_score > 220:
                            win_prob = 0.85
                        elif projected_score > 200:
                            win_prob = 0.75
                        elif projected_score > 180:
                            win_prob = 0.65
                        elif projected_score > 160:
                            win_prob = 0.55
                        elif projected_score > 140:
                            win_prob = 0.45
                        else:
                            win_prob = 0.35
                        
                        # Adjust for wickets lost
                        win_prob *= (0.7 + ((10 - wickets_lost) / 30))
                    
                    # Create a gauge chart for win probability
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=win_prob * 100,
                        title={'text': f"{batting_team} Win Probability"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "red"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': win_prob * 100
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    if win_prob > 0.7:
                        st.success(f"{batting_team} is in a strong position to win!")
                    elif win_prob > 0.5:
                        st.info(f"{batting_team} has a slight edge, but the game is still open.")
                    else:
                        st.warning(f"{bowling_team} is in a better position to win.")
            else:
                st.info("Enter match situation and click 'Calculate Win Probability' to see prediction.")
        
        # Player Performance Predictor
        st.markdown("<h3>Player Performance Predictor</h3>", unsafe_allow_html=True)
        
        if {"batter", "batsman_run", "BowlingTeam"}.issubset(df.columns):
            # Player selection
            all_batters = sorted(df["batter"].unique())
            selected_batter = st.selectbox("Select Batsman", options=all_batters if all_batters else ["Batter"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                opposition_team = st.selectbox("Opposition Team", options=sorted(df["BowlingTeam"].unique()) if "BowlingTeam" in df.columns else ["Team A"])
                match_phase = st.selectbox("Match Phase", options=["Powerplay (1-6)", "Middle Overs (7-15)", "Death Overs (16-20)"])
                
                # Map phase to over ranges
                phase_map = {
                    "Powerplay (1-6)": (1, 6),
                    "Middle Overs (7-15)": (7, 15),
                    "Death Overs (16-20)": (16, 20)
                }
                
            with col2:
                if st.button("Predict Performance"):
                    with st.spinner("Analyzing player data..."):
                        # Filter data for the selected player and conditions
                        start_over, end_over = phase_map[match_phase]
                        
                        player_data = df[
                            (df["batter"] == selected_batter) & 
                            (df["BowlingTeam"] == opposition_team) &
                            (df["overs"] >= start_over) &
                            (df["overs"] <= end_over)
                        ]
                        
                        if not player_data.empty:
                            # Calculate metrics
                            total_runs = player_data["batsman_run"].sum()
                            total_balls = player_data.shape[0]
                            strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
                            
                            # Predict performance with some randomness
                            import random
                            predicted_balls = max(10, min(30, total_balls + random.randint(-5, 5)))
                            predicted_sr = max(100, min(250, strike_rate * random.uniform(0.8, 1.2)))
                            predicted_runs = int((predicted_sr / 100) * predicted_balls)
                            
                            # Show prediction in a nice format
                            st.metric("Predicted Runs", predicted_runs, delta=None)
                            st.metric("Predicted Strike Rate", f"{predicted_sr:.2f}", delta=None)
                            
                            # Show historical stats for comparison
                            st.subheader("Historical Performance:")
                            st.metric("Average Runs", f"{total_runs/max(1, len(player_data['ID'].unique())):.1f}")
                            st.metric("Historical Strike Rate", f"{strike_rate:.2f}")
                        else:
                            st.warning(f"Not enough data for {selected_batter} against {opposition_team} during {match_phase}.")
                else:
                    st.info("Click 'Predict Performance' to analyze this player's expected stats.")
        
        # Match Simulator
        st.markdown("<h3>IPL Match Simulator</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Team 1", options=sorted(df["BattingTeam"].unique()) if "BattingTeam" in df.columns else ["Team A"])
            team2 = st.selectbox("Team 2", options=[team for team in sorted(df["BattingTeam"].unique()) if team != team1] if "BattingTeam" in df.columns else ["Team B"])
            
            venue = st.selectbox("Venue", options=sorted(df["venue"].unique()) if "venue" in df.columns else ["Neutral Venue"])
            
        with col2:
            if st.button("Simulate Match"):
                with st.spinner("Simulating match outcome..."):
                    # Simple match simulation (in a real app this would use a more sophisticated model)
                    import time
                    time.sleep(1)  # Simulate processing time
                    
                    # Create a simple simulation
                    import random
                    
                    # First innings
                    team1_score = random.randint(140, 220)
                    team1_wickets = random.randint(3, 10)
                    
                    # Second innings
                    team2_score = random.randint(team1_score - 30, team1_score + 30)
                    team2_wickets = random.randint(3, 10)
                    
                    # Determine winner
                    if team1_score > team2_score:
                        winner = team1
                        win_margin = f"{team1_score - team2_score} runs"
                    else:
                        winner = team2
                        win_margin = f"{10 - team2_wickets} wickets"
                    
                    # Display simulated scorecard
                    st.subheader("Simulated Match Result")
                    
                    # Create a styled box for the match summary
                    st.markdown(f"""
                    <div style='background-color:#f0f0f0; padding:20px; border-radius:10px; box-shadow:2px 2px 5px grey;'>
                        <h3 style='text-align:center; color:#1E88E5;'>{team1} vs {team2}</h3>
                        <h4 style='text-align:center; font-style:italic;'>at {venue}</h4>
                        <p style='text-align:center; font-size:18px;'>{team1}: {team1_score}/{team1_wickets}</p>
                        <p style='text-align:center; font-size:18px;'>{team2}: {team2_score}/{team2_wickets}</p>
                        <h4 style='text-align:center; margin-top:20px; color:green;'>{winner} wins by {win_margin}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show simulated commentary highlights
                    st.subheader("Match Highlights")
                    
                    highlights = [
                        f"Powerplay: {team1} scored {random.randint(40, 70)}/{random.randint(0, 3)} in the first 6 overs.",
                        f"Middle Overs: {team2} maintained a run rate of {random.uniform(7.5, 10.5):.1f} during overs 7-15.",
                        f"Death Overs: {random.choice(all_batters if 'all_batters' in locals() else ['A Player'])} hit {random.randint(3, 6)} sixes in the final overs.",
                        f"Bowling: {random.choice(df['bowler'].unique() if 'bowler' in df.columns else ['A Bowler'])} took {random.randint(2, 5)} wickets."
                    ]
                    
                    for highlight in highlights:
                        st.markdown(f"- {highlight}")
            else:
                st.info("Select teams and venue, then click 'Simulate Match' to see predicted outcome.")

    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>IPL Data Analysis Dashboard | Created with Streamlit</p>
        <p>Data source: IPL Dataset 2008-2022 | Last updated: April 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Help tooltip
    st.sidebar.markdown("---")
    with st.sidebar.expander("Need Help?"):
        st.markdown("""
        ### How to Use This Dashboard
        
        1. **Upload Data**: Start by uploading an IPL dataset CSV or use the sample data option.
        2. **Navigate Tabs**: Explore different aspects of IPL data through the various tabs.
        3. **Customize Views**: Use the filters in the sidebar to focus on specific teams, players, or seasons.
        4. **Download Insights**: Export processed data using the export options.
        
        ### Tips
        - Hover over charts for detailed information
        - Use the expanders to view additional details
        - Try different color themes for visualizations
        """)
        
    # Version info
    st.sidebar.info("v1.0.0 | Developed by IPL Analytics")
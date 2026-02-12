"""
Interactive Dashboard for Micro-Muscle Fatigue Prediction
Built with Streamlit for hackathon demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

from data_generation import WearableDataGenerator, create_sequences
from model import FatigueLSTMModel


# Page configuration
st.set_page_config(
    page_title="Fatigue Prediction System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .risk-green {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-yellow {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-red {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)


class FatigueDashboard:
    """Dashboard for real-time fatigue monitoring and prediction"""
    
    def __init__(self):
        self.model = FatigueLSTMModel(sequence_length=30, n_features=6)
        self.generator = WearableDataGenerator(seed=42)
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load pre-trained model or train new one"""
        if os.path.exists('fatigue_model.h5'):
            try:
                self.model.load_model('fatigue_model.h5')
            except Exception as e:
                st.warning("Could not load saved model. Using untrained model.")
                self.model.build_model()
        else:
            st.warning("No pre-trained model found. Please train the model first using train.py")
            self.model.build_model()
    
    def get_risk_level(self, fatigue_score):
        """Determine risk level from fatigue score"""
        if fatigue_score < 40:
            return "Green", "🟢", "Low Risk", "#28a745"
        elif fatigue_score < 70:
            return "Yellow", "🟡", "Medium Risk", "#ffc107"
        else:
            return "Red", "🔴", "High Risk", "#dc3545"
    
    def generate_recovery_recommendations(self, fatigue_score, session_data):
        """Generate personalized recovery recommendations"""
        recommendations = []
        
        if fatigue_score > 70:
            recommendations.append("🚨 **CRITICAL**: Immediate rest required")
            recommendations.append("⚠️ Reduce sprint load by 15-20%")
            recommendations.append("🧊 Apply ice therapy to hamstring region (15 min)")
            recommendations.append("💧 Increase hydration: 500ml water immediately")
        
        avg_sleep = session_data['sleep_hours'].mean()
        if avg_sleep < 6:
            recommendations.append("😴 **Sleep Deficit Detected**: Aim for 8+ hours tonight")
            recommendations.append("📵 Avoid screens 1 hour before bedtime")
        
        avg_strain = session_data['muscle_strain'].mean()
        if avg_strain > 7:
            recommendations.append("🧘 Targeted hamstring stretching protocol (20 min)")
            recommendations.append("💆 Consider massage therapy or foam rolling")
        
        avg_training_load = session_data['training_load'].mean()
        if avg_training_load > 75:
            recommendations.append("⏸️ Extend recovery window to 48 hours")
            recommendations.append("🏊 Light active recovery (swimming/cycling)")
        
        if fatigue_score < 40 and len(recommendations) == 0:
            recommendations.append("✅ Excellent recovery status")
            recommendations.append("💪 Maintain current training intensity")
            recommendations.append("🎯 Focus on technique and form")
        
        return recommendations
    
    def predict_fatigue(self, session_data):
        """Predict current and future fatigue"""
        # Prepare data for prediction
        feature_columns = ['heart_rate', 'acceleration', 'muscle_strain', 
                          'training_load', 'recovery_time', 'sleep_hours']
        
        # Get last 30 timesteps
        if len(session_data) >= 30:
            recent_data = session_data[feature_columns].iloc[-30:].values
            X = recent_data.reshape(1, 30, 6)
            
            # Current fatigue prediction
            current_fatigue = self.model.predict(X)[0]
            
            # Future projection (30 minutes ahead)
            future_sequence = session_data[feature_columns].values
            future_predictions = self.model.predict_future(future_sequence, steps_ahead=30)
            
            return current_fatigue, future_predictions
        else:
            st.warning("Need at least 30 minutes of data for prediction")
            return None, None
    
    def plot_fatigue_risk_gauge(self, fatigue_score):
        """Create gauge chart for fatigue risk"""
        risk_level, emoji, risk_text, color = self.get_risk_level(fatigue_score)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fatigue_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{emoji} Current Fatigue Risk", 'font': {'size': 24}},
            delta={'reference': 40, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#d4edda'},
                    {'range': [40, 70], 'color': '#fff3cd'},
                    {'range': [70, 100], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="white",
            font={'size': 16}
        )
        
        return fig
    
    def plot_future_projection(self, future_predictions):
        """Plot 30-minute fatigue forecast"""
        time_points = list(range(1, len(future_predictions) + 1))
        
        fig = go.Figure()
        
        # Predicted line
        fig.add_trace(go.Scatter(
            x=time_points,
            y=future_predictions,
            mode='lines+markers',
            name='Forecasted Fatigue',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=6)
        ))
        
        # Risk zones
        fig.add_hrect(y0=0, y1=40, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=40, y1=70, fillcolor="yellow", opacity=0.1, line_width=0)
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title="30-Minute Fatigue Forecast",
            xaxis_title="Minutes Ahead",
            yaxis_title="Predicted Fatigue Score (%)",
            height=350,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_yaxis(range=[0, 100])
        
        return fig
    
    def plot_sensor_data(self, session_data):
        """Visualize real-time sensor data"""
        fig = go.Figure()
        
        # Create time axis
        time_axis = list(range(len(session_data)))
        
        # Plot each sensor
        fig.add_trace(go.Scatter(
            x=time_axis, y=session_data['heart_rate'],
            name='Heart Rate', line=dict(color='#e74c3c', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_axis, y=session_data['muscle_strain'] * 10,  # Scale for visibility
            name='Muscle Strain (×10)', line=dict(color='#9b59b6', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_axis, y=session_data['acceleration'] * 10,  # Scale for visibility
            name='Acceleration (×10)', line=dict(color='#3498db', width=2)
        ))
        
        fig.update_layout(
            title="Real-Time Sensor Data",
            xaxis_title="Time (minutes)",
            yaxis_title="Sensor Values",
            height=300,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_muscle_stress_heatmap(self, session_data):
        """Visualize muscle stress distribution"""
        # Simulate muscle groups (for demo)
        muscle_groups = ['Hamstring', 'Quadriceps', 'Calves', 'Glutes', 'Hip Flexors']
        time_windows = ['0-10 min', '10-20 min', '20-30 min', '30-40 min', '40+ min']
        
        # Generate stress levels based on session data
        avg_strain = session_data['muscle_strain'].mean()
        stress_matrix = np.random.normal(avg_strain, 1.5, (len(muscle_groups), len(time_windows)))
        stress_matrix = np.clip(stress_matrix, 0, 10)
        
        fig = go.Figure(data=go.Heatmap(
            z=stress_matrix,
            x=time_windows,
            y=muscle_groups,
            colorscale='RdYlGn_r',
            text=np.round(stress_matrix, 1),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Stress Level")
        ))
        
        fig.update_layout(
            title="Muscle Stress Distribution",
            height=300,
            template='plotly_white'
        )
        
        return fig
    
    def plot_player_comparison(self):
        """Compare two simulated players"""
        # Generate data for two players
        player1_data = self.generator.generate_session_data(120, fatigue_trend='increasing')
        player2_data = self.generator.generate_session_data(120, fatigue_trend='stable')
        
        # Get predictions if possible
        if len(player1_data) >= 30:
            feature_cols = ['heart_rate', 'acceleration', 'muscle_strain', 
                           'training_load', 'recovery_time', 'sleep_hours']
            
            X1 = player1_data[feature_cols].iloc[-30:].values.reshape(1, 30, 6)
            X2 = player2_data[feature_cols].iloc[-30:].values.reshape(1, 30, 6)
            
            fatigue1 = self.model.predict(X1)[0] if self.model.model else player1_data['fatigue_score'].iloc[-1]
            fatigue2 = self.model.predict(X2)[0] if self.model.model else player2_data['fatigue_score'].iloc[-1]
        else:
            fatigue1 = player1_data['fatigue_score'].iloc[-1]
            fatigue2 = player2_data['fatigue_score'].iloc[-1]
        
        # Create comparison chart
        fig = go.Figure()
        
        players = ['Player A', 'Player B']
        fatigue_scores = [fatigue1, fatigue2]
        colors = ['#ff7f0e' if f > 70 else '#ffc107' if f > 40 else '#28a745' for f in fatigue_scores]
        
        fig.add_trace(go.Bar(
            x=players,
            y=fatigue_scores,
            marker_color=colors,
            text=[f'{f:.1f}%' for f in fatigue_scores],
            textposition='outside'
        ))
        
        # Add risk zones
        fig.add_hline(y=40, line_dash="dash", line_color="green", annotation_text="Low Risk")
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk")
        
        fig.update_layout(
            title="Player Fatigue Comparison",
            yaxis_title="Fatigue Score (%)",
            height=300,
            template='plotly_white',
            yaxis=dict(range=[0, 100])
        )
        
        return fig


def main():
    """Main dashboard application"""
    
    # Header
    st.title("⚡ LSTM-Based Micro-Muscle Fatigue Prediction System")
    st.markdown("*AI-Powered Early Warning System for Non-Contact Sports Injuries*")
    st.markdown("---")
    
    # Initialize dashboard
    dashboard = FatigueDashboard()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        demo_mode = st.selectbox(
            "Select Demo Mode",
            ["Live Simulation", "Upload CSV", "Player Comparison"]
        )
        
        if demo_mode == "Live Simulation":
            fatigue_trend = st.select_slider(
                "Fatigue Trend",
                options=['decreasing', 'stable', 'increasing'],
                value='increasing'
            )
            session_duration = st.slider(
                "Session Duration (minutes)",
                30, 180, 90
            )
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("**Model**: LSTM Neural Network  \n**Sequence Length**: 30 timesteps  \n**Features**: 6 sensors")
    
    # Main content
    if demo_mode == "Live Simulation":
        st.header("🔴 Live Training Session")
        
        # Generate button
        if st.button("▶️ Start New Session", type="primary"):
            with st.spinner("Generating session data..."):
                session_data = dashboard.generator.generate_session_data(
                    session_duration, 
                    fatigue_trend
                )
                st.session_state['session_data'] = session_data
        
        # Display if data exists
        if 'session_data' in st.session_state:
            session_data = st.session_state['session_data']
            
            # Predict fatigue
            current_fatigue, future_predictions = dashboard.predict_fatigue(session_data)
            
            if current_fatigue is not None:
                # Top metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    risk_level, emoji, risk_text, color = dashboard.get_risk_level(current_fatigue)
                    st.metric("Current Fatigue", f"{current_fatigue:.1f}%", 
                             delta=f"{risk_text}", delta_color="inverse")
                
                with col2:
                    final_prediction = future_predictions[-1]
                    delta_30min = final_prediction - current_fatigue
                    st.metric("30-Min Projection", f"{final_prediction:.1f}%",
                             delta=f"{delta_30min:+.1f}%", delta_color="inverse")
                
                with col3:
                    st.metric("Heart Rate Avg", f"{session_data['heart_rate'].mean():.0f} bpm")
                
                with col4:
                    st.metric("Muscle Strain Avg", f"{session_data['muscle_strain'].mean():.1f}/10")
                
                st.markdown("---")
                
                # Risk assessment and forecast
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(
                        dashboard.plot_fatigue_risk_gauge(current_fatigue),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        dashboard.plot_future_projection(future_predictions),
                        use_container_width=True
                    )
                
                # Alert banner
                if current_fatigue > 70:
                    st.error(f"🚨 **ALERT**: High micro-fatigue risk detected in hamstring region! Current score: {current_fatigue:.1f}%")
                elif current_fatigue > 40:
                    st.warning(f"⚠️ **CAUTION**: Moderate fatigue detected. Monitor closely.")
                else:
                    st.success(f"✅ **SAFE**: Fatigue levels within safe range.")
                
                st.markdown("---")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        dashboard.plot_sensor_data(session_data),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        dashboard.plot_muscle_stress_heatmap(session_data),
                        use_container_width=True
                    )
                
                # Recovery recommendations
                st.markdown("---")
                st.header("💊 Recovery Recommendations")
                
                recommendations = dashboard.generate_recovery_recommendations(
                    current_fatigue, session_data
                )
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
    
    elif demo_mode == "Upload CSV":
        st.header("📁 Upload Training Session Data")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with sensor data",
            type=['csv'],
            help="CSV should contain: heart_rate, acceleration, muscle_strain, training_load, recovery_time, sleep_hours"
        )
        
        if uploaded_file is not None:
            try:
                session_data = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(session_data)} data points")
                
                # Predict fatigue
                current_fatigue, future_predictions = dashboard.predict_fatigue(session_data)
                
                if current_fatigue is not None:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.plotly_chart(
                            dashboard.plot_fatigue_risk_gauge(current_fatigue),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.plotly_chart(
                            dashboard.plot_future_projection(future_predictions),
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("👆 Upload a CSV file to analyze session data")
            
            # Download sample CSV
            if st.button("Download Sample CSV"):
                sample_data = dashboard.generator.generate_session_data(60)
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    "Download Sample",
                    csv,
                    "sample_session.csv",
                    "text/csv"
                )
    
    elif demo_mode == "Player Comparison":
        st.header("👥 Multi-Player Fatigue Monitoring")
        
        if st.button("Compare Players", type="primary"):
            st.plotly_chart(
                dashboard.plot_player_comparison(),
                use_container_width=True
            )
            
            st.info("**Player A** shows progressive fatigue increase - recommend rest  \n**Player B** maintains stable fatigue levels - cleared for continued training")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>LSTM Micro-Muscle Fatigue Prediction System</strong></p>
            <p>Hackathon Prototype | AI-Powered Sports Injury Prevention</p>
            <p style='font-size: 0.8rem;'>⚠️ This is a demonstration system using simulated data. Not for clinical use.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

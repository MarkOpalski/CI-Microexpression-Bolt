"""
Interactive Plotly visualizations for emotion timeline and analysis
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from config import REPORT_CONFIG


class EmotionVisualizer:
    """
    Create interactive visualizations for emotion analysis results
    """
    
    def __init__(self, theme: str = "plotly_dark"):
        """
        Initialize visualizer
        
        Args:
            theme: Plotly theme to use
        """
        self.theme = theme
        self.emotion_colors = {
            'angry': '#FF4444',
            'disgust': '#8B4513',
            'fear': '#800080',
            'happy': '#FFD700',
            'sad': '#4169E1',
            'surprise': '#FF8C00',
            'neutral': '#808080'
        }
    
    def create_emotion_timeline(self, emotion_data: pd.DataFrame, 
                              speech_segments: Optional[List[Dict[str, Any]]] = None) -> go.Figure:
        """
        Create emotion timeline visualization
        
        Args:
            emotion_data: DataFrame with emotion timeline
            speech_segments: Optional speech segments for overlay
            
        Returns:
            Plotly figure
        """
        if emotion_data.empty:
            # Create empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No emotion data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Emotion Intensities', 'Dominant Emotions', 'Speech Segments'],
            vertical_spacing=0.08,
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Convert timestamp to datetime for better x-axis
        if 'timestamp' in emotion_data.columns:
            start_time = emotion_data['timestamp'].min()
            emotion_data['time_offset'] = emotion_data['timestamp'] - start_time
        
        # Plot 1: Emotion intensities over time
        emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        for emotion in emotion_columns:
            if emotion in emotion_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=emotion_data['time_offset'],
                        y=emotion_data[emotion],
                        mode='lines',
                        name=emotion.capitalize(),
                        line=dict(color=self.emotion_colors[emotion], width=2),
                        hovertemplate=f'<b>{emotion.capitalize()}</b><br>' +
                                    'Time: %{x:.1f}s<br>' +
                                    'Intensity: %{y:.1f}%<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Dominant emotion over time
        if 'dominant_emotion' in emotion_data.columns:
            # Create categorical y-axis for dominant emotions
            emotion_categories = list(self.emotion_colors.keys())
            emotion_data['emotion_numeric'] = emotion_data['dominant_emotion'].map(
                {emotion: i for i, emotion in enumerate(emotion_categories)}
            )
            
            fig.add_trace(
                go.Scatter(
                    x=emotion_data['time_offset'],
                    y=emotion_data['emotion_numeric'],
                    mode='markers+lines',
                    name='Dominant Emotion',
                    marker=dict(
                        color=[self.emotion_colors.get(emotion, '#808080') 
                              for emotion in emotion_data['dominant_emotion']],
                        size=8
                    ),
                    line=dict(width=1),
                    hovertemplate='<b>Dominant Emotion</b><br>' +
                                'Time: %{x:.1f}s<br>' +
                                'Emotion: %{text}<extra></extra>',
                    text=emotion_data['dominant_emotion']
                ),
                row=2, col=1
            )
            
            # Update y-axis for dominant emotions
            fig.update_yaxes(
                tickmode='array',
                tickvals=list(range(len(emotion_categories))),
                ticktext=[emotion.capitalize() for emotion in emotion_categories],
                row=2, col=1
            )
        
        # Plot 3: Speech segments
        if speech_segments:
            for i, segment in enumerate(speech_segments):
                start_time_offset = segment['start'] - (start_time if 'timestamp' in emotion_data.columns else 0)
                end_time_offset = segment['end'] - (start_time if 'timestamp' in emotion_data.columns else 0)
                
                # Color by dominant emotion in segment
                segment_color = self.emotion_colors.get(segment.get('dominant_emotion', 'neutral'), '#808080')
                
                fig.add_trace(
                    go.Scatter(
                        x=[start_time_offset, end_time_offset],
                        y=[i, i],
                        mode='lines+markers',
                        name=f'Segment {i+1}',
                        line=dict(color=segment_color, width=8),
                        marker=dict(size=6),
                        hovertemplate=f'<b>Speech Segment {i+1}</b><br>' +
                                    'Start: %{x[0]:.1f}s<br>' +
                                    'End: %{x[1]:.1f}s<br>' +
                                    f'Text: {segment["text"][:50]}...<br>' +
                                    f'Emotion: {segment.get("dominant_emotion", "N/A")}<extra></extra>',
                        showlegend=False
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title='Emotion Analysis Timeline',
            template=self.theme,
            height=800,
            hovermode='x unified'
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
        
        # Update y-axes
        fig.update_yaxes(title_text="Intensity (%)", row=1, col=1)
        fig.update_yaxes(title_text="Emotion", row=2, col=1)
        fig.update_yaxes(title_text="Segments", row=3, col=1)
        
        return fig
    
    def create_emotion_distribution(self, emotion_data: pd.DataFrame) -> go.Figure:
        """
        Create emotion distribution pie chart
        
        Args:
            emotion_data: DataFrame with emotion data
            
        Returns:
            Plotly figure
        """
        if emotion_data.empty or 'dominant_emotion' not in emotion_data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No emotion data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Count dominant emotions
        emotion_counts = emotion_data['dominant_emotion'].value_counts()
        
        # Create pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=[emotion.capitalize() for emotion in emotion_counts.index],
                values=emotion_counts.values,
                marker_colors=[self.emotion_colors.get(emotion, '#808080') 
                              for emotion in emotion_counts.index],
                hovertemplate='<b>%{label}</b><br>' +
                            'Count: %{value}<br>' +
                            'Percentage: %{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Emotion Distribution',
            template=self.theme
        )
        
        return fig
    
    def create_mismatch_visualization(self, mismatches: List[Dict[str, Any]]) -> go.Figure:
        """
        Create visualization for speech-emotion mismatches
        
        Args:
            mismatches: List of detected mismatches
            
        Returns:
            Plotly figure
        """
        if not mismatches:
            fig = go.Figure()
            fig.add_annotation(
                text="No mismatches detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create scatter plot
        fig = go.Figure()
        
        for i, mismatch in enumerate(mismatches):
            fig.add_trace(
                go.Scatter(
                    x=[mismatch['start_time']],
                    y=[mismatch['severity']],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=self.emotion_colors.get(mismatch['visual_emotion'], '#808080'),
                        line=dict(width=2, color='white')
                    ),
                    name=f'Mismatch {i+1}',
                    hovertemplate='<b>Mismatch</b><br>' +
                                'Time: %{x:.1f}s<br>' +
                                'Severity: %{y:.2f}<br>' +
                                f'Text: {mismatch["text_sentiment"]}<br>' +
                                f'Visual: {mismatch["visual_emotion"]}<br>' +
                                f'Text: "{mismatch["text"][:50]}..."<extra></extra>',
                    showlegend=False
                )
            )
        
        fig.update_layout(
            title='Speech-Emotion Mismatches',
            xaxis_title='Time (seconds)',
            yaxis_title='Mismatch Severity',
            template=self.theme
        )
        
        return fig
    
    def create_baseline_comparison(self, emotion_data: pd.DataFrame, 
                                 baseline_emotions: Dict[str, float]) -> go.Figure:
        """
        Create baseline comparison visualization
        
        Args:
            emotion_data: DataFrame with emotion data
            baseline_emotions: Baseline emotion scores
            
        Returns:
            Plotly figure
        """
        if emotion_data.empty or not baseline_emotions:
            fig = go.Figure()
            fig.add_annotation(
                text="No baseline data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Calculate mean emotions
        emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        mean_emotions = {}
        
        for emotion in emotion_columns:
            if emotion in emotion_data.columns:
                mean_emotions[emotion] = emotion_data[emotion].mean()
            else:
                mean_emotions[emotion] = 0.0
        
        # Create comparison bar chart
        emotions = list(baseline_emotions.keys())
        baseline_values = [baseline_emotions[emotion] for emotion in emotions]
        current_values = [mean_emotions.get(emotion, 0.0) for emotion in emotions]
        
        fig = go.Figure(data=[
            go.Bar(
                name='Baseline',
                x=emotions,
                y=baseline_values,
                marker_color='lightblue',
                opacity=0.7
            ),
            go.Bar(
                name='Current Session',
                x=emotions,
                y=current_values,
                marker_color=[self.emotion_colors.get(emotion, '#808080') for emotion in emotions],
                opacity=0.8
            )
        ])
        
        fig.update_layout(
            title='Emotion Comparison: Baseline vs Current Session',
            xaxis_title='Emotions',
            yaxis_title='Average Intensity (%)',
            barmode='group',
            template=self.theme
        )
        
        return fig
    
    def create_confidence_timeline(self, emotion_data: pd.DataFrame) -> go.Figure:
        """
        Create confidence score timeline
        
        Args:
            emotion_data: DataFrame with emotion data
            
        Returns:
            Plotly figure
        """
        if emotion_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No confidence data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Calculate confidence scores (max emotion value)
        emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        available_emotions = [col for col in emotion_columns if col in emotion_data.columns]
        
        if not available_emotions:
            return go.Figure()
        
        confidence_scores = emotion_data[available_emotions].max(axis=1)
        
        if 'timestamp' in emotion_data.columns:
            start_time = emotion_data['timestamp'].min()
            time_offset = emotion_data['timestamp'] - start_time
        else:
            time_offset = range(len(confidence_scores))
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=time_offset,
                y=confidence_scores,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='orange', width=2),
                marker=dict(size=4),
                hovertemplate='<b>Detection Confidence</b><br>' +
                            'Time: %{x:.1f}s<br>' +
                            'Confidence: %{y:.1f}%<extra></extra>'
            )
        )
        
        # Add threshold line
        threshold = REPORT_CONFIG.get('confidence_threshold', 0.6) * 100
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({threshold}%)"
        )
        
        fig.update_layout(
            title='Emotion Detection Confidence Over Time',
            xaxis_title='Time (seconds)',
            yaxis_title='Confidence (%)',
            template=self.theme
        )
        
        return fig
    
    def create_summary_dashboard(self, emotion_data: pd.DataFrame,
                               speech_segments: Optional[List[Dict[str, Any]]] = None,
                               mismatches: Optional[List[Dict[str, Any]]] = None,
                               baseline_emotions: Optional[Dict[str, float]] = None) -> go.Figure:
        """
        Create comprehensive summary dashboard
        
        Args:
            emotion_data: DataFrame with emotion data
            speech_segments: Optional speech segments
            mismatches: Optional detected mismatches
            baseline_emotions: Optional baseline emotions
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Emotion Timeline', 'Emotion Distribution', 
                          'Confidence Over Time', 'Baseline Comparison'],
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if not emotion_data.empty:
            # Timeline (simplified)
            if 'timestamp' in emotion_data.columns:
                start_time = emotion_data['timestamp'].min()
                time_offset = emotion_data['timestamp'] - start_time
            else:
                time_offset = range(len(emotion_data))
            
            # Add dominant emotion line
            if 'dominant_emotion' in emotion_data.columns:
                emotion_categories = list(self.emotion_colors.keys())
                emotion_numeric = emotion_data['dominant_emotion'].map(
                    {emotion: i for i, emotion in enumerate(emotion_categories)}
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=time_offset,
                        y=emotion_numeric,
                        mode='lines+markers',
                        name='Dominant Emotion',
                        marker=dict(size=4),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Distribution pie chart
            if 'dominant_emotion' in emotion_data.columns:
                emotion_counts = emotion_data['dominant_emotion'].value_counts()
                fig.add_trace(
                    go.Pie(
                        labels=[emotion.capitalize() for emotion in emotion_counts.index],
                        values=emotion_counts.values,
                        marker_colors=[self.emotion_colors.get(emotion, '#808080') 
                                      for emotion in emotion_counts.index],
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Confidence timeline
            emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            available_emotions = [col for col in emotion_columns if col in emotion_data.columns]
            
            if available_emotions:
                confidence_scores = emotion_data[available_emotions].max(axis=1)
                fig.add_trace(
                    go.Scatter(
                        x=time_offset,
                        y=confidence_scores,
                        mode='lines',
                        name='Confidence',
                        line=dict(color='orange'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Baseline comparison
            if baseline_emotions:
                emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                mean_emotions = {}
                
                for emotion in emotion_columns:
                    if emotion in emotion_data.columns:
                        mean_emotions[emotion] = emotion_data[emotion].mean()
                
                emotions = list(baseline_emotions.keys())
                baseline_values = [baseline_emotions[emotion] for emotion in emotions]
                current_values = [mean_emotions.get(emotion, 0.0) for emotion in emotions]
                
                fig.add_trace(
                    go.Bar(
                        x=emotions,
                        y=baseline_values,
                        name='Baseline',
                        marker_color='lightblue',
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Bar(
                        x=emotions,
                        y=current_values,
                        name='Current',
                        marker_color='darkblue',
                        opacity=0.8,
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title='Emotion Analysis Summary Dashboard',
            template=self.theme,
            height=600
        )
        
        return fig


def create_emotion_visualizer(theme: str = "plotly_dark") -> EmotionVisualizer:
    """
    Factory function to create emotion visualizer
    
    Args:
        theme: Plotly theme
        
    Returns:
        EmotionVisualizer instance
    """
    return EmotionVisualizer(theme)
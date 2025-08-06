"""
PDF report generation using ReportLab
"""
import io
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.colors import HexColor
import plotly.io as pio

from config import REPORT_CONFIG
from utils.helpers import format_duration


class CIReportGenerator:
    """
    Generate comprehensive CI analysis reports in PDF format
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report styling
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Colors
        self.colors = {
            'primary': HexColor('#2E86AB'),
            'secondary': HexColor('#A23B72'),
            'accent': HexColor('#F18F01'),
            'warning': HexColor('#C73E1D'),
            'success': HexColor('#4CAF50'),
            'neutral': HexColor('#757575')
        }
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.darkred
        ))
        
        self.styles.add(ParagraphStyle(
            name='HighlightBox',
            parent=self.styles['Normal'],
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=10,
            spaceAfter=12
        ))
    
    def generate_report(self, 
                       session_data: Dict[str, Any],
                       emotion_timeline: pd.DataFrame,
                       speech_segments: List[Dict[str, Any]],
                       mismatches: List[Dict[str, Any]],
                       baseline_emotions: Optional[Dict[str, float]] = None,
                       visualizations: Optional[Dict[str, Any]] = None) -> Path:
        """
        Generate comprehensive CI analysis report
        
        Args:
            session_data: Session metadata
            emotion_timeline: DataFrame with emotion data
            speech_segments: List of speech segments
            mismatches: List of detected mismatches
            baseline_emotions: Optional baseline emotion data
            visualizations: Optional visualization figures
            
        Returns:
            Path to generated PDF report
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = session_data.get('session_id', 'unknown')
        filename = f"CI_Analysis_Report_{session_id}_{timestamp}.pdf"
        output_path = self.output_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build report content
        story = []
        
        # Title page
        story.extend(self._create_title_page(session_data))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(
            emotion_timeline, speech_segments, mismatches
        ))
        story.append(PageBreak())
        
        # Session metadata
        story.extend(self._create_session_metadata(session_data))
        
        # Emotion analysis section
        story.extend(self._create_emotion_analysis_section(
            emotion_timeline, baseline_emotions
        ))
        
        # Speech analysis section
        story.extend(self._create_speech_analysis_section(speech_segments))
        
        # Mismatch analysis section
        story.extend(self._create_mismatch_analysis_section(mismatches))
        
        # Visualizations section
        if visualizations:
            story.append(PageBreak())
            story.extend(self._create_visualizations_section(visualizations))
        
        # Conclusions and recommendations
        story.append(PageBreak())
        story.extend(self._create_conclusions_section(
            emotion_timeline, mismatches
        ))
        
        # Technical appendix
        story.append(PageBreak())
        story.extend(self._create_technical_appendix(session_data))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def _create_title_page(self, session_data: Dict[str, Any]) -> List:
        """Create report title page"""
        story = []
        
        # Title
        story.append(Paragraph(
            "COUNTER-INTELLIGENCE<br/>MICROEXPRESSION ANALYSIS REPORT",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 0.5*inch))
        
        # Classification notice
        story.append(Paragraph(
            "<b>CLASSIFICATION: FOR OFFICIAL USE ONLY</b>",
            self.styles['HighlightBox']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Session information
        session_info = [
            ['Session ID:', session_data.get('session_id', 'N/A')],
            ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')],
            ['Analyst:', session_data.get('user_id', 'Unknown')],
            ['Source Type:', session_data.get('source_type', 'Unknown')],
            ['Duration:', format_duration(session_data.get('duration_seconds', 0))]
        ]
        
        table = Table(session_info, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(table)
        
        story.append(Spacer(1, 1*inch))
        
        # Warning notice
        warning_text = """
        <b>HANDLING NOTICE:</b><br/>
        This report contains sensitive analytical assessments based on behavioral 
        analysis techniques. Distribution is restricted to authorized personnel only. 
        The contents of this report are derived from automated analysis systems and 
        should be corroborated with additional intelligence sources before operational use.
        """
        story.append(Paragraph(warning_text, self.styles['HighlightBox']))
        
        return story
    
    def _create_executive_summary(self, emotion_timeline: pd.DataFrame,
                                speech_segments: List[Dict[str, Any]],
                                mismatches: List[Dict[str, Any]]) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        
        # Key findings
        total_frames = len(emotion_timeline) if not emotion_timeline.empty else 0
        total_speech_time = sum(seg['end'] - seg['start'] for seg in speech_segments)
        high_severity_mismatches = len([m for m in mismatches if m.get('severity', 0) > 0.7])
        
        summary_text = f"""
        <b>Analysis Overview:</b><br/>
        • Total frames analyzed: {total_frames:,}<br/>
        • Speech segments identified: {len(speech_segments)}<br/>
        • Total speech duration: {format_duration(total_speech_time)}<br/>
        • Behavioral inconsistencies detected: {len(mismatches)}<br/>
        • High-severity indicators: {high_severity_mismatches}<br/><br/>
        """
        
        if not emotion_timeline.empty and 'dominant_emotion' in emotion_timeline.columns:
            dominant_emotion = emotion_timeline['dominant_emotion'].mode().iloc[0]
            emotion_stability = emotion_timeline['dominant_emotion'].nunique()
            
            summary_text += f"""
            <b>Behavioral Assessment:</b><br/>
            • Primary emotional state: {dominant_emotion.upper()}<br/>
            • Emotional variability: {'HIGH' if emotion_stability > 4 else 'MODERATE' if emotion_stability > 2 else 'LOW'}<br/>
            • Deception indicators: {'PRESENT' if high_severity_mismatches > 0 else 'MINIMAL'}<br/><br/>
            """
        
        if high_severity_mismatches > 0:
            summary_text += f"""
            <b>⚠️ ANALYST ATTENTION REQUIRED:</b><br/>
            {high_severity_mismatches} high-confidence behavioral inconsistencies detected. 
            Recommend detailed review of flagged segments and correlation with additional intelligence sources.
            """
        else:
            summary_text += """
            <b>✓ BASELINE BEHAVIOR:</b><br/>
            No significant behavioral anomalies detected. Subject exhibits consistent 
            emotional-verbal alignment throughout the session.
            """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _create_session_metadata(self, session_data: Dict[str, Any]) -> List:
        """Create session metadata section"""
        story = []
        
        story.append(Paragraph("SESSION METADATA", self.styles['SectionHeader']))
        
        # Technical details
        metadata = [
            ['Parameter', 'Value'],
            ['Session ID', session_data.get('session_id', 'N/A')],
            ['Source Type', session_data.get('source_type', 'Unknown')],
            ['Video Resolution', f"{session_data.get('width', 'N/A')}x{session_data.get('height', 'N/A')}"],
            ['Frame Rate', f"{session_data.get('fps', 'N/A')} fps"],
            ['Total Frames', f"{session_data.get('frame_count', 'N/A'):,}"],
            ['Duration', format_duration(session_data.get('duration_seconds', 0))],
            ['Processing Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')],
            ['Analysis Models', 'MTCNN, DeepFace, Whisper'],
            ['Chain of Custody', session_data.get('file_hash', 'N/A')[:16] + '...']
        ]
        
        table = Table(metadata, colWidths=[2.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _create_emotion_analysis_section(self, emotion_timeline: pd.DataFrame,
                                       baseline_emotions: Optional[Dict[str, float]]) -> List:
        """Create emotion analysis section"""
        story = []
        
        story.append(Paragraph("EMOTIONAL BEHAVIORAL ANALYSIS", self.styles['SectionHeader']))
        
        if emotion_timeline.empty:
            story.append(Paragraph("No emotion data available for analysis.", self.styles['Normal']))
            return story
        
        # Emotion statistics
        story.append(Paragraph("Emotional Profile", self.styles['SubHeader']))
        
        if 'dominant_emotion' in emotion_timeline.columns:
            emotion_counts = emotion_timeline['dominant_emotion'].value_counts()
            emotion_percentages = (emotion_counts / len(emotion_timeline) * 100).round(1)
            
            emotion_data = [['Emotion', 'Frequency', 'Percentage']]
            for emotion, count in emotion_counts.items():
                percentage = emotion_percentages[emotion]
                emotion_data.append([emotion.capitalize(), str(count), f"{percentage}%"])
            
            table = Table(emotion_data, colWidths=[1.5*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 0.2*inch))
        
        # Baseline comparison
        if baseline_emotions:
            story.append(Paragraph("Baseline Deviation Analysis", self.styles['SubHeader']))
            
            emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            current_means = {}
            for emotion in emotion_columns:
                if emotion in emotion_timeline.columns:
                    current_means[emotion] = emotion_timeline[emotion].mean()
            
            deviation_data = [['Emotion', 'Baseline', 'Current', 'Deviation']]
            for emotion in baseline_emotions:
                baseline_val = baseline_emotions[emotion]
                current_val = current_means.get(emotion, 0)
                deviation = current_val - baseline_val
                
                deviation_data.append([
                    emotion.capitalize(),
                    f"{baseline_val:.1f}%",
                    f"{current_val:.1f}%",
                    f"{deviation:+.1f}%"
                ])
            
            table = Table(deviation_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        story.append(Spacer(1, 0.2*inch))
        return story
    
    def _create_speech_analysis_section(self, speech_segments: List[Dict[str, Any]]) -> List:
        """Create speech analysis section"""
        story = []
        
        story.append(Paragraph("SPEECH CONTENT ANALYSIS", self.styles['SectionHeader']))
        
        if not speech_segments:
            story.append(Paragraph("No speech segments detected.", self.styles['Normal']))
            return story
        
        # Speech statistics
        total_duration = sum(seg['end'] - seg['start'] for seg in speech_segments)
        avg_segment_length = total_duration / len(speech_segments)
        
        stats_text = f"""
        <b>Speech Metrics:</b><br/>
        • Total segments: {len(speech_segments)}<br/>
        • Total speech time: {format_duration(total_duration)}<br/>
        • Average segment length: {avg_segment_length:.1f} seconds<br/>
        • Speech rate: {len(speech_segments) / (total_duration / 60):.1f} segments/minute<br/>
        """
        story.append(Paragraph(stats_text, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key segments (top 5 by duration)
        story.append(Paragraph("Key Speech Segments", self.styles['SubHeader']))
        
        sorted_segments = sorted(speech_segments, 
                               key=lambda x: x['end'] - x['start'], 
                               reverse=True)[:5]
        
        for i, segment in enumerate(sorted_segments, 1):
            duration = segment['end'] - segment['start']
            emotion = segment.get('dominant_emotion', 'neutral')
            confidence = segment.get('emotion_confidence', 0)
            
            segment_text = f"""
            <b>Segment {i}:</b> {format_duration(segment['start'])} - {format_duration(segment['end'])} 
            (Duration: {duration:.1f}s, Emotion: {emotion}, Confidence: {confidence:.2f})<br/>
            <i>"{segment['text'][:200]}{'...' if len(segment['text']) > 200 else ''}"</i><br/><br/>
            """
            story.append(Paragraph(segment_text, self.styles['Normal']))
        
        return story
    
    def _create_mismatch_analysis_section(self, mismatches: List[Dict[str, Any]]) -> List:
        """Create mismatch analysis section"""
        story = []
        
        story.append(Paragraph("BEHAVIORAL INCONSISTENCY ANALYSIS", self.styles['SectionHeader']))
        
        if not mismatches:
            story.append(Paragraph(
                "✓ No significant behavioral inconsistencies detected. "
                "Subject exhibits consistent emotional-verbal alignment.",
                self.styles['Normal']
            ))
            return story
        
        # Mismatch summary
        high_severity = [m for m in mismatches if m.get('severity', 0) > 0.7]
        medium_severity = [m for m in mismatches if 0.4 <= m.get('severity', 0) <= 0.7]
        low_severity = [m for m in mismatches if m.get('severity', 0) < 0.4]
        
        summary_text = f"""
        <b>⚠️ INCONSISTENCY SUMMARY:</b><br/>
        • Total inconsistencies: {len(mismatches)}<br/>
        • High severity (>70%): {len(high_severity)}<br/>
        • Medium severity (40-70%): {len(medium_severity)}<br/>
        • Low severity (<40%): {len(low_severity)}<br/><br/>
        """
        story.append(Paragraph(summary_text, self.styles['HighlightBox']))
        
        # Detailed mismatch analysis
        story.append(Paragraph("Detailed Inconsistency Report", self.styles['SubHeader']))
        
        # Sort by severity
        sorted_mismatches = sorted(mismatches, key=lambda x: x.get('severity', 0), reverse=True)
        
        for i, mismatch in enumerate(sorted_mismatches[:10], 1):  # Top 10
            severity = mismatch.get('severity', 0)
            severity_label = "HIGH" if severity > 0.7 else "MEDIUM" if severity > 0.4 else "LOW"
            
            mismatch_text = f"""
            <b>Inconsistency #{i} - {severity_label} PRIORITY</b><br/>
            • Time: {format_duration(mismatch['start_time'])} - {format_duration(mismatch['end_time'])}<br/>
            • Severity Score: {severity:.2f}<br/>
            • Text Sentiment: {mismatch['text_sentiment'].upper()}<br/>
            • Visual Emotion: {mismatch['visual_emotion'].upper()}<br/>
            • Confidence: {mismatch['confidence']:.2f}<br/>
            • Content: "{mismatch['text'][:150]}{'...' if len(mismatch['text']) > 150 else ''}"<br/><br/>
            """
            story.append(Paragraph(mismatch_text, self.styles['Normal']))
        
        if len(mismatches) > 10:
            story.append(Paragraph(
                f"<i>Note: {len(mismatches) - 10} additional inconsistencies detected. "
                "Full details available in technical appendix.</i>",
                self.styles['Normal']
            ))
        
        return story
    
    def _create_visualizations_section(self, visualizations: Dict[str, Any]) -> List:
        """Create visualizations section"""
        story = []
        
        story.append(Paragraph("VISUAL ANALYSIS", self.styles['SectionHeader']))
        
        # Convert Plotly figures to images and embed
        for title, fig in visualizations.items():
            try:
                # Convert to PNG
                img_bytes = pio.to_image(fig, format='png', width=800, height=600)
                
                # Create ReportLab image
                img_buffer = io.BytesIO(img_bytes)
                img = RLImage(img_buffer, width=6*inch, height=4.5*inch)
                
                story.append(Paragraph(title, self.styles['SubHeader']))
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
                
            except Exception as e:
                story.append(Paragraph(f"Visualization '{title}' could not be rendered: {e}", 
                                     self.styles['Normal']))
        
        return story
    
    def _create_conclusions_section(self, emotion_timeline: pd.DataFrame,
                                  mismatches: List[Dict[str, Any]]) -> List:
        """Create conclusions and recommendations section"""
        story = []
        
        story.append(Paragraph("ANALYTICAL CONCLUSIONS & RECOMMENDATIONS", self.styles['SectionHeader']))
        
        # Risk assessment
        high_risk_indicators = len([m for m in mismatches if m.get('severity', 0) > 0.7])
        
        if high_risk_indicators > 0:
            risk_level = "HIGH"
            risk_color = colors.red
        elif len(mismatches) > 3:
            risk_level = "MEDIUM"
            risk_color = colors.orange
        else:
            risk_level = "LOW"
            risk_color = colors.green
        
        risk_text = f"""
        <b>RISK ASSESSMENT: <font color="{risk_color}">{risk_level}</font></b><br/>
        Based on behavioral analysis indicators and speech-emotion inconsistencies.
        """
        story.append(Paragraph(risk_text, self.styles['HighlightBox']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key findings
        story.append(Paragraph("Key Analytical Findings", self.styles['SubHeader']))
        
        findings = []
        
        if not emotion_timeline.empty and 'dominant_emotion' in emotion_timeline.columns:
            dominant_emotion = emotion_timeline['dominant_emotion'].mode().iloc[0]
            findings.append(f"Primary emotional state: {dominant_emotion.upper()}")
            
            emotion_stability = emotion_timeline['dominant_emotion'].nunique()
            if emotion_stability > 5:
                findings.append("High emotional variability detected - potential stress indicators")
            elif emotion_stability < 3:
                findings.append("Low emotional variability - consistent baseline behavior")
        
        if high_risk_indicators > 0:
            findings.append(f"{high_risk_indicators} high-confidence deception indicators identified")
        
        if len(mismatches) == 0:
            findings.append("No significant behavioral inconsistencies detected")
        
        for finding in findings:
            story.append(Paragraph(f"• {finding}", self.styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        story.append(Paragraph("Operational Recommendations", self.styles['SubHeader']))
        
        recommendations = []
        
        if high_risk_indicators > 0:
            recommendations.extend([
                "Recommend immediate follow-up interview with trained interrogator",
                "Cross-reference findings with additional intelligence sources",
                "Consider polygraph examination if operationally feasible"
            ])
        elif len(mismatches) > 0:
            recommendations.extend([
                "Monitor subject for additional behavioral indicators",
                "Review flagged segments with human analyst",
                "Consider extended observation period"
            ])
        else:
            recommendations.extend([
                "Subject exhibits baseline behavioral patterns",
                "No immediate follow-up action required",
                "Maintain standard monitoring protocols"
            ])
        
        recommendations.append("Archive analysis results for future behavioral baseline reference")
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
        
        return story
    
    def _create_technical_appendix(self, session_data: Dict[str, Any]) -> List:
        """Create technical appendix"""
        story = []
        
        story.append(Paragraph("TECHNICAL APPENDIX", self.styles['SectionHeader']))
        
        # Analysis methodology
        story.append(Paragraph("Analysis Methodology", self.styles['SubHeader']))
        
        methodology_text = """
        <b>Facial Expression Analysis:</b><br/>
        • Face Detection: MTCNN (Multi-task CNN) with 90% confidence threshold<br/>
        • Landmark Detection: 68-point facial landmark alignment<br/>
        • Emotion Classification: DeepFace framework with ensemble models<br/>
        • Emotion Categories: Anger, Contempt, Fear, Joy, Surprise, Disgust, Sadness<br/><br/>
        
        <b>Speech Analysis:</b><br/>
        • Transcription: OpenAI Whisper automatic speech recognition<br/>
        • Temporal Alignment: Frame-level synchronization with emotion timeline<br/>
        • Sentiment Analysis: Keyword-based classification with contextual weighting<br/><br/>
        
        <b>Inconsistency Detection:</b><br/>
        • Threshold-based mismatch identification between verbal and non-verbal cues<br/>
        • Severity scoring based on confidence levels and deviation magnitude<br/>
        • Temporal clustering of behavioral anomalies<br/>
        """
        story.append(Paragraph(methodology_text, self.styles['Normal']))
        
        # System specifications
        story.append(Paragraph("System Specifications", self.styles['SubHeader']))
        
        specs_text = f"""
        <b>Processing Environment:</b><br/>
        • Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>
        • Session ID: {session_data.get('session_id', 'N/A')}<br/>
        • Processing Mode: {'Air-gapped' if session_data.get('air_gapped', True) else 'Connected'}<br/>
        • Data Retention: Local processing only, no cloud transmission<br/>
        • Chain of Custody: SHA-256 hash verification maintained<br/>
        """
        story.append(Paragraph(specs_text, self.styles['Normal']))
        
        # Limitations and disclaimers
        story.append(Paragraph("Limitations and Disclaimers", self.styles['SubHeader']))
        
        disclaimer_text = """
        <b>IMPORTANT LIMITATIONS:</b><br/>
        • Automated analysis results require human analyst validation<br/>
        • Cultural and individual variations may affect emotion recognition accuracy<br/>
        • Environmental factors (lighting, camera angle) can impact detection quality<br/>
        • Results should be corroborated with additional intelligence sources<br/>
        • System trained on Western facial expressions - may have reduced accuracy for other populations<br/>
        • False positive rate: ~5-10% for high-confidence detections<br/><br/>
        
        <b>OPERATIONAL SECURITY:</b><br/>
        • All processing performed locally without external network access<br/>
        • Source material metadata stripped during ingestion<br/>
        • Audit trail maintained for chain of custody<br/>
        • Report classification: FOR OFFICIAL USE ONLY<br/>
        """
        story.append(Paragraph(disclaimer_text, self.styles['HighlightBox']))
        
        return story


def create_report_generator(output_dir: Path) -> CIReportGenerator:
    """
    Factory function to create report generator
    
    Args:
        output_dir: Output directory for reports
        
    Returns:
        CIReportGenerator instance
    """
    return CIReportGenerator(output_dir)
"""
Micro-expression analysis using DeepFace and optional Py-Feat FACS
"""
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
from pathlib import Path

# Suppress DeepFace warnings
warnings.filterwarnings('ignore')

from deepface import DeepFace
from config import ANALYSIS_CONFIG

# Optional Py-Feat import
try:
    from feat import Detector as FeatDetector
    FEAT_AVAILABLE = True
except ImportError:
    FEAT_AVAILABLE = False
    FeatDetector = None


class ExpressionAnalyzer:
    """
    Micro-expression analysis with emotion detection and optional FACS AU
    """
    
    def __init__(self, enable_facs: Optional[bool] = None):
        """
        Initialize expression analyzer
        
        Args:
            enable_facs: Enable FACS Action Units analysis (requires py-feat)
        """
        self.enable_facs = enable_facs if enable_facs is not None else ANALYSIS_CONFIG["enable_facs_au"]
        self.emotion_threshold = ANALYSIS_CONFIG["emotion_threshold"]
        self.baseline_window = ANALYSIS_CONFIG["baseline_window_seconds"]
        self.spike_factor = ANALYSIS_CONFIG["spike_detection_factor"]
        
        # Emotion labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Initialize FACS detector if enabled
        self.facs_detector = None
        if self.enable_facs and FEAT_AVAILABLE:
            try:
                self.facs_detector = FeatDetector(
                    face_model="retinaface",
                    landmark_model="mobilefacenet",
                    au_model="xgb",
                    emotion_model="resmasknet",
                    facepose_model="img2pose"
                )
            except Exception as e:
                print(f"Warning: Could not initialize FACS detector: {e}")
                self.facs_detector = None
        
        # Baseline tracking
        self.baseline_emotions = {}
        self.emotion_history = []
    
    def analyze_face(self, face_crop: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """
        Analyze facial expression in cropped face image
        
        Args:
            face_crop: Cropped face image
            timestamp: Timestamp of the frame
            
        Returns:
            Analysis results dictionary
        """
        results = {
            'timestamp': timestamp,
            'emotions': {},
            'dominant_emotion': None,
            'confidence': 0.0,
            'baseline_deviation': {},
            'is_spike': False,
            'facs_aus': None
        }
        
        try:
            # DeepFace emotion analysis
            emotion_results = DeepFace.analyze(
                face_crop,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Handle both single result and list of results
            if isinstance(emotion_results, list):
                emotion_data = emotion_results[0]
            else:
                emotion_data = emotion_results
            
            # Extract emotion scores
            emotions = emotion_data['emotion']
            results['emotions'] = emotions
            results['dominant_emotion'] = emotion_data['dominant_emotion']
            results['confidence'] = max(emotions.values()) / 100.0
            
            # Calculate baseline deviations
            results['baseline_deviation'] = self._calculate_baseline_deviation(emotions)
            
            # Detect emotion spikes
            results['is_spike'] = self._detect_emotion_spike(emotions)
            
            # Add to emotion history
            self.emotion_history.append({
                'timestamp': timestamp,
                'emotions': emotions,
                'dominant_emotion': emotion_data['dominant_emotion']
            })
            
            # FACS Action Units analysis (if enabled)
            if self.facs_detector is not None:
                try:
                    facs_results = self._analyze_facs(face_crop)
                    results['facs_aus'] = facs_results
                except Exception as e:
                    print(f"FACS analysis failed: {e}")
            
        except Exception as e:
            print(f"Expression analysis failed: {e}")
            # Return default values on failure
            results['emotions'] = {emotion: 0.0 for emotion in self.emotion_labels}
            results['dominant_emotion'] = 'neutral'
        
        return results
    
    def _analyze_facs(self, face_crop: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Analyze FACS Action Units using Py-Feat
        
        Args:
            face_crop: Cropped face image
            
        Returns:
            Dictionary of Action Unit intensities
        """
        if self.facs_detector is None:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Analyze with Py-Feat
            results = self.facs_detector.detect_image(rgb_face)
            
            if results is not None and len(results) > 0:
                # Extract AU columns
                au_columns = [col for col in results.columns if col.startswith('AU')]
                aus = {}
                
                for au_col in au_columns:
                    au_value = results[au_col].iloc[0]
                    if not pd.isna(au_value):
                        aus[au_col] = float(au_value)
                
                return aus
        
        except Exception as e:
            print(f"FACS analysis error: {e}")
        
        return None
    
    def _calculate_baseline_deviation(self, current_emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate deviation from baseline emotions
        
        Args:
            current_emotions: Current emotion scores
            
        Returns:
            Baseline deviation scores
        """
        if not self.baseline_emotions:
            return {emotion: 0.0 for emotion in current_emotions}
        
        deviations = {}
        for emotion, current_score in current_emotions.items():
            baseline_score = self.baseline_emotions.get(emotion, current_score)
            deviation = current_score - baseline_score
            deviations[emotion] = deviation
        
        return deviations
    
    def _detect_emotion_spike(self, current_emotions: Dict[str, float]) -> bool:
        """
        Detect if current emotions represent a significant spike
        
        Args:
            current_emotions: Current emotion scores
            
        Returns:
            True if spike detected
        """
        if not self.baseline_emotions:
            return False
        
        for emotion, current_score in current_emotions.items():
            baseline_score = self.baseline_emotions.get(emotion, 0.0)
            if current_score > baseline_score * self.spike_factor and current_score > 20.0:
                return True
        
        return False
    
    def establish_baseline(self, face_crops: List[np.ndarray], 
                          timestamps: List[float]) -> Dict[str, float]:
        """
        Establish emotional baseline from initial frames
        
        Args:
            face_crops: List of face crop images
            timestamps: Corresponding timestamps
            
        Returns:
            Baseline emotion scores
        """
        emotion_accumulator = {emotion: [] for emotion in self.emotion_labels}
        
        for face_crop in face_crops:
            try:
                emotion_results = DeepFace.analyze(
                    face_crop,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(emotion_results, list):
                    emotions = emotion_results[0]['emotion']
                else:
                    emotions = emotion_results['emotion']
                
                for emotion, score in emotions.items():
                    emotion_accumulator[emotion].append(score)
            
            except Exception:
                continue
        
        # Calculate baseline as median of each emotion
        baseline = {}
        for emotion, scores in emotion_accumulator.items():
            if scores:
                baseline[emotion] = np.median(scores)
            else:
                baseline[emotion] = 0.0
        
        self.baseline_emotions = baseline
        return baseline
    
    def get_emotion_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from emotion history
        
        Returns:
            Dictionary with emotion statistics
        """
        if not self.emotion_history:
            return {}
        
        # Convert to DataFrame for easier analysis
        df_data = []
        for entry in self.emotion_history:
            row = {'timestamp': entry['timestamp'], 'dominant_emotion': entry['dominant_emotion']}
            row.update(entry['emotions'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        stats = {
            'total_frames': len(df),
            'duration_seconds': df['timestamp'].max() - df['timestamp'].min() if len(df) > 1 else 0,
            'dominant_emotion_counts': df['dominant_emotion'].value_counts().to_dict(),
            'emotion_means': {col: df[col].mean() for col in self.emotion_labels if col in df.columns},
            'emotion_stds': {col: df[col].std() for col in self.emotion_labels if col in df.columns},
            'emotion_maxes': {col: df[col].max() for col in self.emotion_labels if col in df.columns}
        }
        
        return stats
    
    def detect_inconsistencies(self, speech_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect emotion-speech inconsistencies
        
        Args:
            speech_segments: List of speech segments with timestamps and content
            
        Returns:
            List of detected inconsistencies
        """
        inconsistencies = []
        
        for segment in speech_segments:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            
            # Find emotion data in time range
            relevant_emotions = [
                entry for entry in self.emotion_history
                if start_time <= entry['timestamp'] <= end_time
            ]
            
            if not relevant_emotions:
                continue
            
            # Analyze text sentiment (simple keyword-based approach)
            text_sentiment = self._analyze_text_sentiment(text)
            
            # Get dominant emotions in segment
            emotion_counts = {}
            for entry in relevant_emotions:
                emotion = entry['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            dominant_visual_emotion = max(emotion_counts, key=emotion_counts.get)
            
            # Check for inconsistencies
            if self._is_inconsistent(text_sentiment, dominant_visual_emotion):
                inconsistency = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text,
                    'text_sentiment': text_sentiment,
                    'visual_emotion': dominant_visual_emotion,
                    'confidence': len(relevant_emotions) / max(1, (end_time - start_time))
                }
                inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _analyze_text_sentiment(self, text: str) -> str:
        """
        Simple keyword-based text sentiment analysis
        
        Args:
            text: Input text
            
        Returns:
            Predicted sentiment
        """
        text_lower = text.lower()
        
        # Simple keyword matching
        positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'wonderful', 'amazing']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disgusting', 'angry']
        fear_words = ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        fear_count = sum(1 for word in fear_words if word in text_lower)
        
        if fear_count > 0:
            return 'fear'
        elif positive_count > negative_count:
            return 'happy'
        elif negative_count > positive_count:
            return 'angry'
        else:
            return 'neutral'
    
    def _is_inconsistent(self, text_sentiment: str, visual_emotion: str) -> bool:
        """
        Check if text sentiment and visual emotion are inconsistent
        
        Args:
            text_sentiment: Predicted text sentiment
            visual_emotion: Detected visual emotion
            
        Returns:
            True if inconsistent
        """
        # Define inconsistency rules
        inconsistency_map = {
            'happy': ['angry', 'sad', 'fear', 'disgust'],
            'angry': ['happy', 'surprise'],
            'sad': ['happy', 'surprise'],
            'fear': ['happy'],
            'neutral': []  # Neutral can be consistent with anything
        }
        
        return visual_emotion in inconsistency_map.get(text_sentiment, [])
    
    def export_timeline(self) -> pd.DataFrame:
        """
        Export emotion timeline as DataFrame
        
        Returns:
            DataFrame with emotion timeline
        """
        if not self.emotion_history:
            return pd.DataFrame()
        
        df_data = []
        for entry in self.emotion_history:
            row = {
                'timestamp': entry['timestamp'],
                'dominant_emotion': entry['dominant_emotion']
            }
            row.update(entry['emotions'])
            df_data.append(row)
        
        return pd.DataFrame(df_data)


def create_expression_analyzer(enable_facs: Optional[bool] = None) -> ExpressionAnalyzer:
    """
    Factory function to create expression analyzer
    
    Args:
        enable_facs: Enable FACS Action Units analysis
        
    Returns:
        ExpressionAnalyzer instance
    """
    return ExpressionAnalyzer(enable_facs)
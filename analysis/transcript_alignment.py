"""
Speech transcription and alignment with emotion timeline using OpenAI Whisper
"""
import whisper
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import cv2
import subprocess
import os

from config import ANALYSIS_CONFIG


class TranscriptAligner:
    """
    Align Whisper speech transcription with emotion timeline
    """
    
    def __init__(self, model_size: Optional[str] = None):
        """
        Initialize transcript aligner
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size or ANALYSIS_CONFIG["whisper_model"]
        self.chunk_duration = ANALYSIS_CONFIG["chunk_duration"]
        
        # Load Whisper model
        print(f"Loading Whisper model: {self.model_size}")
        self.model = whisper.load_model(self.model_size)
        
        self.transcription_cache = {}
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        # Create temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio.close()
        
        try:
            # Use OpenCV to extract audio
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Use ffmpeg if available, otherwise use basic extraction
            if self._has_ffmpeg():
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1',
                    '-y', temp_audio.name
                ]
                subprocess.run(cmd, capture_output=True, check=True)
            else:
                # Fallback: create silent audio file
                duration = frame_count / fps if fps > 0 else 10
                self._create_silent_audio(temp_audio.name, duration)
        
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            # Create silent audio as fallback
            self._create_silent_audio(temp_audio.name, 10)
        
        return temp_audio.name
    
    def _has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _create_silent_audio(self, output_path: str, duration: float):
        """Create a silent audio file"""
        import wave
        
        sample_rate = 16000
        samples = int(sample_rate * duration)
        
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Write silent samples
            silent_data = b'\x00\x00' * samples
            wav_file.writeframes(silent_data)
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription results with timestamps
        """
        try:
            # Check cache
            if audio_path in self.transcription_cache:
                return self.transcription_cache[audio_path]
            
            print("Transcribing audio...")
            result = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )
            
            # Cache result
            self.transcription_cache[audio_path] = result
            
            return result
        
        except Exception as e:
            print(f"Transcription failed: {e}")
            # Return empty result
            return {
                'text': '',
                'segments': [],
                'language': 'en'
            }
    
    def align_with_emotions(self, transcription: Dict[str, Any], 
                           emotion_timeline: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Align transcription segments with emotion timeline
        
        Args:
            transcription: Whisper transcription results
            emotion_timeline: DataFrame with emotion data and timestamps
            
        Returns:
            List of aligned segments with emotion data
        """
        aligned_segments = []
        
        if emotion_timeline.empty:
            # No emotion data available
            for segment in transcription.get('segments', []):
                aligned_segment = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'emotions': {},
                    'dominant_emotion': 'neutral',
                    'emotion_confidence': 0.0,
                    'emotion_variance': 0.0
                }
                aligned_segments.append(aligned_segment)
            return aligned_segments
        
        for segment in transcription.get('segments', []):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            
            # Find emotion data within segment timeframe
            mask = (emotion_timeline['timestamp'] >= start_time) & \
                   (emotion_timeline['timestamp'] <= end_time)
            segment_emotions = emotion_timeline[mask]
            
            if len(segment_emotions) == 0:
                # No emotion data for this segment
                aligned_segment = {
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'emotions': {},
                    'dominant_emotion': 'neutral',
                    'emotion_confidence': 0.0,
                    'emotion_variance': 0.0
                }
            else:
                # Calculate emotion statistics for segment
                emotion_stats = self._calculate_segment_emotions(segment_emotions)
                
                aligned_segment = {
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'emotions': emotion_stats['mean_emotions'],
                    'dominant_emotion': emotion_stats['dominant_emotion'],
                    'emotion_confidence': emotion_stats['confidence'],
                    'emotion_variance': emotion_stats['variance'],
                    'emotion_spikes': emotion_stats['spikes']
                }
            
            aligned_segments.append(aligned_segment)
        
        return aligned_segments
    
    def _calculate_segment_emotions(self, segment_emotions: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate emotion statistics for a segment
        
        Args:
            segment_emotions: DataFrame with emotion data for segment
            
        Returns:
            Dictionary with emotion statistics
        """
        emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Calculate mean emotions
        mean_emotions = {}
        for emotion in emotion_columns:
            if emotion in segment_emotions.columns:
                mean_emotions[emotion] = segment_emotions[emotion].mean()
            else:
                mean_emotions[emotion] = 0.0
        
        # Find dominant emotion
        dominant_emotion = max(mean_emotions, key=mean_emotions.get)
        confidence = mean_emotions[dominant_emotion] / 100.0
        
        # Calculate emotion variance (measure of emotional stability)
        variance = 0.0
        if len(segment_emotions) > 1:
            emotion_values = []
            for emotion in emotion_columns:
                if emotion in segment_emotions.columns:
                    emotion_values.extend(segment_emotions[emotion].values)
            if emotion_values:
                variance = np.var(emotion_values)
        
        # Detect emotion spikes within segment
        spikes = []
        if 'dominant_emotion' in segment_emotions.columns:
            emotion_changes = segment_emotions['dominant_emotion'].value_counts()
            if len(emotion_changes) > 1:
                # Multiple emotions detected - potential spikes
                for emotion, count in emotion_changes.items():
                    if emotion != dominant_emotion and count > 1:
                        spikes.append(emotion)
        
        return {
            'mean_emotions': mean_emotions,
            'dominant_emotion': dominant_emotion,
            'confidence': confidence,
            'variance': variance,
            'spikes': spikes
        }
    
    def detect_speech_emotion_mismatches(self, aligned_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect mismatches between speech content and facial emotions
        
        Args:
            aligned_segments: List of aligned segments
            
        Returns:
            List of detected mismatches
        """
        mismatches = []
        
        for segment in aligned_segments:
            text = segment['text'].lower()
            visual_emotion = segment['dominant_emotion']
            confidence = segment['emotion_confidence']
            
            # Skip low-confidence detections
            if confidence < 0.3:
                continue
            
            # Analyze text sentiment
            text_sentiment = self._analyze_text_sentiment(text)
            
            # Check for mismatch
            if self._is_mismatch(text_sentiment, visual_emotion):
                mismatch = {
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'text': segment['text'],
                    'text_sentiment': text_sentiment,
                    'visual_emotion': visual_emotion,
                    'confidence': confidence,
                    'severity': self._calculate_mismatch_severity(text_sentiment, visual_emotion, confidence)
                }
                mismatches.append(mismatch)
        
        # Sort by severity
        mismatches.sort(key=lambda x: x['severity'], reverse=True)
        
        return mismatches
    
    def _analyze_text_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of text content
        
        Args:
            text: Input text
            
        Returns:
            Predicted sentiment category
        """
        # Keyword-based sentiment analysis
        positive_keywords = [
            'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'perfect',
            'awesome', 'brilliant', 'outstanding', 'superb'
        ]
        
        negative_keywords = [
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'dislike', 'angry', 'mad', 'furious', 'upset', 'disappointed',
            'frustrated', 'annoyed', 'irritated', 'outraged'
        ]
        
        fear_keywords = [
            'scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous',
            'concerned', 'frightened', 'panic', 'fear', 'stress', 'tension'
        ]
        
        sad_keywords = [
            'sad', 'depressed', 'unhappy', 'miserable', 'devastated',
            'heartbroken', 'grief', 'sorrow', 'melancholy', 'down'
        ]
        
        surprise_keywords = [
            'surprised', 'shocked', 'amazed', 'astonished', 'stunned',
            'unexpected', 'sudden', 'wow', 'incredible', 'unbelievable'
        ]
        
        # Count keyword matches
        word_counts = {
            'happy': sum(1 for word in positive_keywords if word in text),
            'angry': sum(1 for word in negative_keywords if word in text),
            'fear': sum(1 for word in fear_keywords if word in text),
            'sad': sum(1 for word in sad_keywords if word in text),
            'surprise': sum(1 for word in surprise_keywords if word in text)
        }
        
        # Find dominant sentiment
        max_count = max(word_counts.values())
        if max_count == 0:
            return 'neutral'
        
        return max(word_counts, key=word_counts.get)
    
    def _is_mismatch(self, text_sentiment: str, visual_emotion: str) -> bool:
        """
        Check if text sentiment and visual emotion represent a mismatch
        
        Args:
            text_sentiment: Predicted text sentiment
            visual_emotion: Detected visual emotion
            
        Returns:
            True if mismatch detected
        """
        # Define mismatch rules
        mismatch_pairs = [
            ('happy', 'angry'),
            ('happy', 'sad'),
            ('happy', 'fear'),
            ('angry', 'happy'),
            ('angry', 'surprise'),
            ('sad', 'happy'),
            ('sad', 'surprise'),
            ('fear', 'happy'),
            ('surprise', 'angry'),
            ('surprise', 'sad')
        ]
        
        return (text_sentiment, visual_emotion) in mismatch_pairs
    
    def _calculate_mismatch_severity(self, text_sentiment: str, 
                                   visual_emotion: str, confidence: float) -> float:
        """
        Calculate severity score for a mismatch
        
        Args:
            text_sentiment: Text sentiment
            visual_emotion: Visual emotion
            confidence: Emotion detection confidence
            
        Returns:
            Severity score (0-1)
        """
        # Base severity based on emotion pair
        severity_map = {
            ('happy', 'angry'): 0.9,
            ('happy', 'sad'): 0.8,
            ('angry', 'happy'): 0.9,
            ('sad', 'happy'): 0.7,
            ('fear', 'happy'): 0.6,
            ('surprise', 'angry'): 0.5
        }
        
        base_severity = severity_map.get((text_sentiment, visual_emotion), 0.3)
        
        # Adjust by confidence
        adjusted_severity = base_severity * confidence
        
        return min(1.0, adjusted_severity)
    
    def export_transcript(self, aligned_segments: List[Dict[str, Any]]) -> str:
        """
        Export transcript with emotion annotations
        
        Args:
            aligned_segments: List of aligned segments
            
        Returns:
            Formatted transcript string
        """
        transcript_lines = []
        
        for segment in aligned_segments:
            start_time = self._format_timestamp(segment['start'])
            end_time = self._format_timestamp(segment['end'])
            text = segment['text']
            emotion = segment['dominant_emotion']
            confidence = segment['emotion_confidence']
            
            line = f"[{start_time} - {end_time}] ({emotion}, {confidence:.2f}) {text}"
            transcript_lines.append(line)
        
        return '\n'.join(transcript_lines)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def cleanup(self):
        """Clean up temporary files"""
        for audio_path in self.transcription_cache.keys():
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception:
                pass


def create_transcript_aligner(model_size: Optional[str] = None) -> TranscriptAligner:
    """
    Factory function to create transcript aligner
    
    Args:
        model_size: Whisper model size
        
    Returns:
        TranscriptAligner instance
    """
    return TranscriptAligner(model_size)
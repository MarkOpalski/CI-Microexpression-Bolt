"""
Main application for CI Microexpression Tracking System
"""
import argparse
import sys
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any
import uuid

# Core modules
from config import OUTPUT_DIR, ANALYSIS_CONFIG, SECURITY_CONFIG
from utils.helpers import generate_session_id, create_output_directory
from video.ingestion import create_video_source
from video.face_detection import create_face_detector
from analysis.expression_analysis import create_expression_analyzer
from analysis.transcript_alignment import create_transcript_aligner
from reporting.visualizations import create_emotion_visualizer
from reporting.report_generator import create_report_generator
from security.audit_log import (
    audit_logger, log_video_ingest, log_analysis_start, 
    log_report_generation, log_security_event
)


class CIMicroexpressionTracker:
    """
    Main application class for CI Microexpression Tracking
    """
    
    def __init__(self, user_id: str = "ANALYST"):
        """
        Initialize the tracking system
        
        Args:
            user_id: User identifier for audit logging
        """
        self.user_id = user_id
        self.session_id = generate_session_id()
        self.session_dir = create_output_directory(self.session_id)
        
        # Initialize components
        self.face_detector = None
        self.expression_analyzer = None
        self.transcript_aligner = None
        self.visualizer = None
        self.report_generator = None
        
        # Session data
        self.session_data = {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'start_time': time.time(),
            'air_gapped': SECURITY_CONFIG['air_gapped_mode']
        }
        
        print(f"🔒 CI Microexpression Tracking System Initialized")
        print(f"📋 Session ID: {self.session_id}")
        print(f"👤 Analyst: {self.user_id}")
        print(f"🛡️  Security Mode: {'Air-gapped' if SECURITY_CONFIG['air_gapped_mode'] else 'Connected'}")
        print(f"📁 Output Directory: {self.session_dir}")
        print("-" * 60)
    
    def initialize_components(self):
        """Initialize analysis components"""
        print("🔧 Initializing analysis components...")
        
        try:
            # Face detection
            print("  • Loading face detection models...")
            self.face_detector = create_face_detector()
            
            # Expression analysis
            print("  • Loading emotion recognition models...")
            self.expression_analyzer = create_expression_analyzer(
                enable_facs=ANALYSIS_CONFIG['enable_facs_au']
            )
            
            # Speech transcription
            print("  • Loading speech recognition models...")
            self.transcript_aligner = create_transcript_aligner(
                model_size=ANALYSIS_CONFIG['whisper_model']
            )
            
            # Visualization
            self.visualizer = create_emotion_visualizer()
            
            # Report generation
            self.report_generator = create_report_generator(self.session_dir)
            
            print("✅ All components initialized successfully")
            
            # Log initialization
            log_security_event(
                "System components initialized",
                self.user_id, self.session_id, "INFO"
            )
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            log_security_event(
                f"Component initialization failed: {str(e)}",
                self.user_id, self.session_id, "ERROR"
            )
            raise
    
    def analyze_video(self, source_path: str) -> Dict[str, Any]:
        """
        Perform complete video analysis
        
        Args:
            source_path: Path to video file or camera index
            
        Returns:
            Analysis results dictionary
        """
        print(f"🎥 Starting video analysis: {source_path}")
        
        # Initialize components if not already done
        if self.face_detector is None:
            self.initialize_components()
        
        results = {
            'session_data': self.session_data,
            'emotion_timeline': None,
            'speech_segments': [],
            'mismatches': [],
            'baseline_emotions': None,
            'visualizations': {},
            'report_path': None
        }
        
        try:
            # Create video source
            print("📹 Initializing video source...")
            with create_video_source(source_path, self.user_id, self.session_id) as video_source:
                
                # Get video info
                video_info = video_source.get_info()
                self.session_data.update(video_info)
                
                print(f"  • Resolution: {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}")
                print(f"  • FPS: {video_info.get('fps', 'N/A')}")
                if 'duration_seconds' in video_info:
                    print(f"  • Duration: {video_info['duration_seconds']:.1f} seconds")
                
                # Log analysis start
                log_analysis_start("video_analysis", self.user_id, self.session_id)
                
                # Phase 1: Face detection and emotion analysis
                print("\n🔍 Phase 1: Facial expression analysis...")
                emotion_data = self._analyze_facial_expressions(video_source)
                results['emotion_timeline'] = emotion_data
                
                # Phase 2: Speech transcription (for video files)
                if video_source.is_file:
                    print("\n🎤 Phase 2: Speech transcription...")
                    speech_data = self._analyze_speech(source_path)
                    results['speech_segments'] = speech_data
                    
                    # Phase 3: Alignment and mismatch detection
                    print("\n🔗 Phase 3: Speech-emotion alignment...")
                    mismatches = self._detect_mismatches(emotion_data, speech_data)
                    results['mismatches'] = mismatches
                else:
                    print("ℹ️  Skipping speech analysis for live sources")
                
                # Phase 4: Visualization generation
                print("\n📊 Phase 4: Generating visualizations...")
                visualizations = self._generate_visualizations(results)
                results['visualizations'] = visualizations
                
                # Phase 5: Report generation
                print("\n📄 Phase 5: Generating analysis report...")
                report_path = self._generate_report(results)
                results['report_path'] = report_path
                
                print(f"\n✅ Analysis complete!")
                print(f"📊 Emotions analyzed: {len(emotion_data) if emotion_data is not None else 0} frames")
                print(f"🎤 Speech segments: {len(results['speech_segments'])}")
                print(f"⚠️  Inconsistencies: {len(results['mismatches'])}")
                print(f"📄 Report saved: {report_path.name if report_path else 'N/A'}")
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            log_security_event(
                f"Video analysis failed: {str(e)}",
                self.user_id, self.session_id, "ERROR"
            )
            raise
        
        return results
    
    def _analyze_facial_expressions(self, video_source) -> Optional[Any]:
        """Analyze facial expressions in video"""
        emotion_timeline = []
        baseline_frames = []
        frame_count = 0
        
        print("  • Processing frames for facial expressions...")
        
        try:
            for frame, timestamp in video_source.get_frames():
                frame_count += 1
                
                # Progress indicator
                if frame_count % 30 == 0:
                    print(f"    Processed {frame_count} frames...")
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                if faces:
                    # Use largest face
                    primary_face = faces[0]
                    face_crop = self.face_detector.crop_face(frame, primary_face['bbox'])
                    
                    # Analyze expression
                    expression_result = self.expression_analyzer.analyze_face(face_crop, timestamp)
                    emotion_timeline.append(expression_result)
                    
                    # Collect baseline frames (first 30 seconds)
                    if len(baseline_frames) < 100 and timestamp < 30:
                        baseline_frames.append(face_crop)
                
                # Break for live sources after reasonable duration
                if video_source.is_live and frame_count > 1800:  # ~1 minute at 30fps
                    break
            
            print(f"  • Processed {frame_count} total frames")
            print(f"  • Detected faces in {len(emotion_timeline)} frames")
            
            # Establish baseline if we have enough data
            if len(baseline_frames) >= 10:
                print("  • Establishing emotional baseline...")
                baseline_timestamps = [i for i in range(len(baseline_frames))]
                baseline_emotions = self.expression_analyzer.establish_baseline(
                    baseline_frames, baseline_timestamps
                )
                self.session_data['baseline_emotions'] = baseline_emotions
            
            # Convert to DataFrame
            if emotion_timeline:
                import pandas as pd
                return pd.DataFrame(emotion_timeline)
            else:
                return None
                
        except Exception as e:
            print(f"  ❌ Facial expression analysis failed: {e}")
            return None
    
    def _analyze_speech(self, video_path: str) -> list:
        """Analyze speech content"""
        try:
            # Extract audio
            print("  • Extracting audio from video...")
            audio_path = self.transcript_aligner.extract_audio_from_video(video_path)
            
            # Transcribe
            print("  • Transcribing speech...")
            transcription = self.transcript_aligner.transcribe_audio(audio_path)
            
            # Align with emotions
            emotion_timeline = self.session_data.get('emotion_timeline')
            if emotion_timeline is not None:
                print("  • Aligning speech with emotion timeline...")
                aligned_segments = self.transcript_aligner.align_with_emotions(
                    transcription, emotion_timeline
                )
            else:
                # Create basic segments without emotion alignment
                aligned_segments = []
                for segment in transcription.get('segments', []):
                    aligned_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip(),
                        'emotions': {},
                        'dominant_emotion': 'neutral',
                        'emotion_confidence': 0.0
                    })
            
            print(f"  • Identified {len(aligned_segments)} speech segments")
            return aligned_segments
            
        except Exception as e:
            print(f"  ❌ Speech analysis failed: {e}")
            return []
    
    def _detect_mismatches(self, emotion_timeline, speech_segments: list) -> list:
        """Detect speech-emotion mismatches"""
        try:
            if emotion_timeline is None or not speech_segments:
                return []
            
            print("  • Detecting behavioral inconsistencies...")
            mismatches = self.transcript_aligner.detect_speech_emotion_mismatches(speech_segments)
            
            if mismatches:
                high_severity = len([m for m in mismatches if m.get('severity', 0) > 0.7])
                print(f"  • Found {len(mismatches)} inconsistencies ({high_severity} high-severity)")
            else:
                print("  • No significant inconsistencies detected")
            
            return mismatches
            
        except Exception as e:
            print(f"  ❌ Mismatch detection failed: {e}")
            return []
    
    def _generate_visualizations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis visualizations"""
        visualizations = {}
        
        try:
            emotion_timeline = results.get('emotion_timeline')
            speech_segments = results.get('speech_segments', [])
            mismatches = results.get('mismatches', [])
            baseline_emotions = self.session_data.get('baseline_emotions')
            
            if emotion_timeline is not None and not emotion_timeline.empty:
                # Main timeline
                print("  • Creating emotion timeline...")
                timeline_fig = self.visualizer.create_emotion_timeline(
                    emotion_timeline, speech_segments
                )
                visualizations['Emotion Timeline'] = timeline_fig
                
                # Distribution chart
                print("  • Creating emotion distribution...")
                dist_fig = self.visualizer.create_emotion_distribution(emotion_timeline)
                visualizations['Emotion Distribution'] = dist_fig
                
                # Confidence timeline
                print("  • Creating confidence timeline...")
                conf_fig = self.visualizer.create_confidence_timeline(emotion_timeline)
                visualizations['Detection Confidence'] = conf_fig
                
                # Baseline comparison
                if baseline_emotions:
                    print("  • Creating baseline comparison...")
                    baseline_fig = self.visualizer.create_baseline_comparison(
                        emotion_timeline, baseline_emotions
                    )
                    visualizations['Baseline Comparison'] = baseline_fig
            
            # Mismatch visualization
            if mismatches:
                print("  • Creating mismatch visualization...")
                mismatch_fig = self.visualizer.create_mismatch_visualization(mismatches)
                visualizations['Speech-Emotion Mismatches'] = mismatch_fig
            
            print(f"  • Generated {len(visualizations)} visualizations")
            
        except Exception as e:
            print(f"  ❌ Visualization generation failed: {e}")
        
        return visualizations
    
    def _generate_report(self, results: Dict[str, Any]) -> Optional[Path]:
        """Generate comprehensive analysis report"""
        try:
            report_path = self.report_generator.generate_report(
                session_data=self.session_data,
                emotion_timeline=results.get('emotion_timeline'),
                speech_segments=results.get('speech_segments', []),
                mismatches=results.get('mismatches', []),
                baseline_emotions=self.session_data.get('baseline_emotions'),
                visualizations=results.get('visualizations', {})
            )
            
            # Log report generation
            log_report_generation(str(report_path), self.user_id, self.session_id)
            
            return report_path
            
        except Exception as e:
            print(f"  ❌ Report generation failed: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.transcript_aligner:
            self.transcript_aligner.cleanup()
        
        # Log session end
        self.session_data['end_time'] = time.time()
        self.session_data['duration_seconds'] = (
            self.session_data['end_time'] - self.session_data['start_time']
        )
        
        log_security_event(
            f"Analysis session completed (duration: {self.session_data['duration_seconds']:.1f}s)",
            self.user_id, self.session_id, "INFO"
        )


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="CI Microexpression Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --source 0                    # Use webcam
  python app.py --source video.mp4            # Analyze video file
  python app.py --source rtsp://camera/stream # Analyze RTSP stream
  python app.py --source video.mp4 --user analyst1 # Specify analyst ID
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        required=True,
        help='Video source: camera index (0), file path, or stream URL'
    )
    
    parser.add_argument(
        '--user', '-u',
        default='ANALYST',
        help='User/Analyst identifier for audit logging'
    )
    
    parser.add_argument(
        '--verify-audit',
        action='store_true',
        help='Verify audit log integrity and exit'
    )
    
    parser.add_argument(
        '--setup-git-hooks',
        action='store_true',
        help='Setup git hooks for local development'
    )
    
    args = parser.parse_args()
    
    # Handle git hooks setup
    if args.setup_git_hooks:
        print("🔧 Setting up git hooks for local development...")
        try:
            # Check if os.chmod is available in this environment
            if not hasattr(os, 'chmod'):
                print("⚠️  Warning: os.chmod not available in this Python environment")
                print("   This indicates a corrupted or limited Python installation")
                print("   Git hooks will be configured but permissions must be set manually")
            
            import subprocess
            subprocess.run(['git', 'config', 'core.hooksPath', '.githooks'], check=True)
            
            # Only run chmod on Unix-like systems
            if sys.platform != 'win32' and hasattr(os, 'chmod'):
                subprocess.run(['chmod', '+x', '.githooks/pre-push'], check=True)
                print("✅ Git hooks configured successfully")
            else:
                if sys.platform == 'win32':
                    print("✅ Git hooks configured (chmod skipped on Windows)")
                else:
                    print("✅ Git hooks configured (chmod skipped - environment limitation)")
                print("   - Manually set executable permissions if needed")
            
            print("   - Pre-push hook will prevent accidental GitHub pushes")
            print("   - Use 'git config --unset core.hooksPath' to disable")
        except subprocess.CalledProcessError as e:
            print(f"❌ Git hooks setup failed: {e}")
            print("   This is normal in environments without Git")
        except FileNotFoundError:
            print("❌ Git not found - hooks setup skipped")
            print("   Run this command manually in your local environment:")
            print("   git config core.hooksPath .githooks && chmod +x .githooks/pre-push")
        return 0
    
    # Handle audit verification
    if args.verify_audit:
        print("🔍 Verifying audit log integrity...")
        verification_result = audit_logger.verify_integrity()
        
        if verification_result['valid']:
            print(f"✅ Audit log verified: {verification_result['total_entries']} entries")
        else:
            print(f"❌ Audit log integrity compromised:")
            print(f"  • Corrupted entries: {len(verification_result.get('corrupted_entries', []))}")
            print(f"  • Chain breaks: {len(verification_result.get('chain_breaks', []))}")
        
        return 0 if verification_result['valid'] else 1
    
    # Initialize tracker
    tracker = CIMicroexpressionTracker(user_id=args.user)
    
    try:
        # Convert source to appropriate type
        source = args.source
        if source.isdigit():
            source = int(source)  # Camera index
        
        # Run analysis
        results = tracker.analyze_video(source)
        
        print("\n" + "="*60)
        print("📋 ANALYSIS SUMMARY")
        print("="*60)
        
        if results['report_path']:
            print(f"📄 Report: {results['report_path']}")
        
        if results['mismatches']:
            high_severity = len([m for m in results['mismatches'] if m.get('severity', 0) > 0.7])
            if high_severity > 0:
                print(f"⚠️  HIGH PRIORITY: {high_severity} high-confidence behavioral inconsistencies detected")
            else:
                print(f"ℹ️  {len(results['mismatches'])} behavioral inconsistencies detected (low-medium confidence)")
        else:
            print("✅ No significant behavioral inconsistencies detected")
        
        print(f"📁 Session files: {tracker.session_dir}")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return 1
        
    finally:
        tracker.cleanup()


if __name__ == "__main__":
    sys.exit(main())
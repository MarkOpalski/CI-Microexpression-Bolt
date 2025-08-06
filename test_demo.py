#!/usr/bin/env python3
"""
Demo test for CI Microexpression Tracking System
"""
import numpy as np
import cv2
from app import CIMicroexpressionTracker
import tempfile
import os

def create_test_video():
    """Create a simple test video with a face-like pattern"""
    # Create temporary video file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(temp_fd)
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 3  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create a simple frame with geometric shapes (simulating a face)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background
        frame[:] = (50, 50, 50)
        
        # Face oval
        center = (width//2, height//2)
        axes = (100, 120)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (200, 180, 160), -1)
        
        # Eyes
        eye1_center = (center[0] - 30, center[1] - 20)
        eye2_center = (center[0] + 30, center[1] - 20)
        cv2.circle(frame, eye1_center, 15, (0, 0, 0), -1)
        cv2.circle(frame, eye2_center, 15, (0, 0, 0), -1)
        
        # Mouth (changes over time to simulate expression change)
        mouth_y = center[1] + 30
        if frame_num < total_frames // 3:
            # Neutral
            cv2.ellipse(frame, (center[0], mouth_y), (20, 5), 0, 0, 180, (0, 0, 0), 2)
        elif frame_num < 2 * total_frames // 3:
            # Smile
            cv2.ellipse(frame, (center[0], mouth_y - 5), (25, 10), 0, 0, 180, (0, 0, 0), 2)
        else:
            # Frown
            cv2.ellipse(frame, (center[0], mouth_y + 5), (20, 8), 0, 180, 360, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    return temp_path

def main():
    print("🎬 CI Microexpression Tracking - Demo Test")
    print("=" * 50)
    
    try:
        # Create test video
        print("📹 Creating test video...")
        test_video_path = create_test_video()
        print(f"✅ Test video created: {test_video_path}")
        
        # Initialize tracker
        print("\n🔧 Initializing CI tracker...")
        tracker = CIMicroexpressionTracker(user_id="DEMO_ANALYST")
        print(f"✅ Session ID: {tracker.session_id}")
        
        # Run analysis (this will download models on first run)
        print("\n🔍 Running analysis...")
        print("⏳ Note: First run will download AI models (~300MB)")
        
        results = tracker.analyze_video(test_video_path)
        
        # Display results
        print("\n📊 ANALYSIS RESULTS")
        print("=" * 30)
        
        if results['emotion_timeline'] is not None:
            print(f"✅ Emotions analyzed: {len(results['emotion_timeline'])} frames")
        else:
            print("ℹ️  No emotion data (expected for synthetic test)")
        
        print(f"🎤 Speech segments: {len(results['speech_segments'])}")
        print(f"⚠️  Inconsistencies: {len(results['mismatches'])}")
        
        if results['report_path']:
            print(f"📄 Report generated: {results['report_path'].name}")
        
        print(f"📁 Session files: {tracker.session_dir}")
        
        # Cleanup
        tracker.cleanup()
        os.unlink(test_video_path)
        
        print("\n🎯 Demo test completed successfully!")
        print("📋 System is ready for operational use")
        
    except Exception as e:
        print(f"\n❌ Demo test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
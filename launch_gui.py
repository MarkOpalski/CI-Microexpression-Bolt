#!/usr/bin/env python3
"""
Launcher script for CI Microexpression Tracking GUI
"""
import sys
import os
from pathlib import Path

def main():
    """Launch the CI Tracking GUI application"""
    try:
        print("🔒 Starting CI Microexpression Tracking System GUI...")
        print("=" * 60)
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher required")
            return 1
        
        # Check if we're in the right directory
        if not Path("gui_app.py").exists():
            print("❌ Please run from the project root directory")
            return 1
        
        # Import and run GUI
        from gui_app import main as gui_main
        gui_main()
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Run: pip install -r requirements.txt")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  Application terminated by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
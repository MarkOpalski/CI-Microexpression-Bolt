"""
GUI Application for CI Microexpression Tracking System
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import cv2
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import webbrowser

from app import CIMicroexpressionTracker
from config import OUTPUT_DIR


class CITrackingGUI:
    """
    Professional GUI for CI Microexpression Tracking System
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CI Microexpression Tracking System")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Configure style
        self.setup_styles()
        
        # Application state
        self.tracker = None
        self.current_session = None
        self.video_source = None
        self.analysis_thread = None
        self.is_analyzing = False
        self.frame_queue = queue.Queue()
        
        # Create GUI components
        self.create_layout()
        self.setup_accessibility()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_styles(self):
        """Configure professional styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2E86AB')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#333333')
        style.configure('Status.TLabel', font=('Arial', 10), foreground='#666666')
        style.configure('Success.TLabel', font=('Arial', 10), foreground='#4CAF50')
        style.configure('Warning.TLabel', font=('Arial', 10), foreground='#FF9800')
        style.configure('Error.TLabel', font=('Arial', 10), foreground='#F44336')
        
        # Configure buttons
        style.configure('Primary.TButton', font=('Arial', 10, 'bold'))
        style.configure('Secondary.TButton', font=('Arial', 10))
        style.configure('Danger.TButton', font=('Arial', 10), foreground='#F44336')
        
    def create_layout(self):
        """Create the main application layout with ARIA-like structure"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Banner (Header) region
        self.create_banner(main_container)
        
        # Main content area
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Navigation panel (left sidebar)
        self.create_navigation(content_frame)
        
        # Main content area
        self.create_main_content(content_frame)
        
        # Complementary panel (right sidebar)
        self.create_complementary_panel(content_frame)
        
        # Content info (footer)
        self.create_contentinfo(main_container)
        
    def create_banner(self, parent):
        """Create banner/header region"""
        banner_frame = ttk.Frame(parent, relief=tk.RAISED, borderwidth=1)
        banner_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(
            banner_frame, 
            text="CI MICROEXPRESSION TRACKING SYSTEM",
            style='Title.TLabel'
        )
        title_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Classification notice
        classification_label = ttk.Label(
            banner_frame,
            text="CLASSIFICATION: FOR OFFICIAL USE ONLY",
            style='Warning.TLabel',
            background='#FFF3CD',
            relief=tk.RAISED,
            borderwidth=1
        )
        classification_label.pack(side=tk.RIGHT, padx=10, pady=10)
        
    def create_navigation(self, parent):
        """Create navigation panel"""
        nav_frame = ttk.LabelFrame(parent, text="Navigation", padding=10)
        nav_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        nav_frame.configure(width=200)
        nav_frame.pack_propagate(False)
        
        # Session info
        session_frame = ttk.LabelFrame(nav_frame, text="Session Info", padding=5)
        session_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.session_id_label = ttk.Label(session_frame, text="Session: Not Started", style='Status.TLabel')
        self.session_id_label.pack(anchor=tk.W)
        
        self.analyst_label = ttk.Label(session_frame, text="Analyst: Not Set", style='Status.TLabel')
        self.analyst_label.pack(anchor=tk.W)
        
        # Quick actions
        actions_frame = ttk.LabelFrame(nav_frame, text="Quick Actions", padding=5)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            actions_frame,
            text="New Analysis",
            command=self.new_analysis,
            style='Primary.TButton'
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            actions_frame,
            text="Load Video File",
            command=self.load_video_file,
            style='Secondary.TButton'
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            actions_frame,
            text="Use Webcam",
            command=self.use_webcam,
            style='Secondary.TButton'
        ).pack(fill=tk.X, pady=2)
        
        ttk.Separator(actions_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            actions_frame,
            text="View Reports",
            command=self.view_reports,
            style='Secondary.TButton'
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            actions_frame,
            text="Verify Audit Log",
            command=self.verify_audit_log,
            style='Secondary.TButton'
        ).pack(fill=tk.X, pady=2)
        
        # System status
        status_frame = ttk.LabelFrame(nav_frame, text="System Status", padding=5)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.system_status_label = ttk.Label(status_frame, text="● Ready", style='Success.TLabel')
        self.system_status_label.pack(anchor=tk.W)
        
        self.security_mode_label = ttk.Label(status_frame, text="🛡️ Air-gapped", style='Status.TLabel')
        self.security_mode_label.pack(anchor=tk.W)
        
    def create_main_content(self, parent):
        """Create main content area"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Analysis tab
        self.create_analysis_tab()
        
        # Results tab
        self.create_results_tab()
        
        # Settings tab
        self.create_settings_tab()
        
    def create_analysis_tab(self):
        """Create analysis tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Analysis")
        
        # Video preview area
        preview_frame = ttk.LabelFrame(analysis_frame, text="Video Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video display
        self.video_canvas = tk.Canvas(preview_frame, bg='black', width=640, height=480)
        self.video_canvas.pack(side=tk.LEFT, padx=(0, 10))
        
        # Analysis controls
        controls_frame = ttk.Frame(preview_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Source selection
        source_frame = ttk.LabelFrame(controls_frame, text="Video Source", padding=5)
        source_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.source_var = tk.StringVar(value="none")
        ttk.Radiobutton(source_frame, text="No Source", variable=self.source_var, value="none").pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="Webcam", variable=self.source_var, value="webcam").pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="Video File", variable=self.source_var, value="file").pack(anchor=tk.W)
        
        self.file_path_var = tk.StringVar()
        self.file_path_label = ttk.Label(source_frame, textvariable=self.file_path_var, style='Status.TLabel')
        self.file_path_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Analyst ID
        analyst_frame = ttk.LabelFrame(controls_frame, text="Analyst Information", padding=5)
        analyst_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(analyst_frame, text="Analyst ID:").pack(anchor=tk.W)
        self.analyst_entry = ttk.Entry(analyst_frame, width=20)
        self.analyst_entry.pack(fill=tk.X, pady=(2, 0))
        self.analyst_entry.insert(0, "ANALYST")
        
        # Analysis controls
        control_frame = ttk.LabelFrame(controls_frame, text="Analysis Control", padding=5)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(
            control_frame,
            text="Start Analysis",
            command=self.start_analysis,
            style='Primary.TButton'
        )
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Analysis",
            command=self.stop_analysis,
            style='Danger.TButton',
            state=tk.DISABLED
        )
        self.stop_button.pack(fill=tk.X, pady=2)
        
        # Progress
        progress_frame = ttk.LabelFrame(controls_frame, text="Progress", padding=5)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var, style='Status.TLabel')
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
    def create_results_tab(self):
        """Create results tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Results display
        results_paned = ttk.PanedWindow(results_frame, orient=tk.HORIZONTAL)
        results_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary panel
        summary_frame = ttk.LabelFrame(results_paned, text="Analysis Summary", padding=10)
        results_paned.add(summary_frame, weight=1)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=15, width=40)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Details panel
        details_frame = ttk.LabelFrame(results_paned, text="Detailed Results", padding=10)
        results_paned.add(details_frame, weight=2)
        
        # Create treeview for detailed results
        columns = ('Time', 'Type', 'Description', 'Confidence')
        self.results_tree = ttk.Treeview(details_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)
        
        # Scrollbars for treeview
        tree_scroll_y = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        tree_scroll_x = ttk.Scrollbar(details_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_settings_tab(self):
        """Create settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Settings content
        settings_scroll = scrolledtext.ScrolledText(settings_frame, height=20)
        settings_scroll.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        settings_content = """
SYSTEM CONFIGURATION

Analysis Settings:
• Emotion Threshold: 70%
• Baseline Window: 30 seconds
• Spike Detection Factor: 2.0x
• FACS Action Units: Disabled
• Whisper Model: Base

Security Settings:
• Air-gapped Mode: Enabled
• Metadata Stripping: Enabled
• Audit Logging: Enabled
• Max File Size: 500MB

Video Processing:
• Target FPS: 30
• Max Resolution: 1920x1080
• Face Detection Confidence: 90%
• Minimum Face Size: 40px

Report Generation:
• PDF Template: Standard
• Include Timeline: Yes
• Include Statistics: Yes
• Confidence Threshold: 60%

Note: Settings are configured in config.py
Restart application after making changes.
        """
        settings_scroll.insert(tk.END, settings_content)
        settings_scroll.configure(state=tk.DISABLED)
        
    def create_complementary_panel(self, parent):
        """Create complementary information panel"""
        comp_frame = ttk.LabelFrame(parent, text="System Information", padding=10)
        comp_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        comp_frame.configure(width=250)
        comp_frame.pack_propagate(False)
        
        # Real-time stats
        stats_frame = ttk.LabelFrame(comp_frame, text="Live Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.frames_processed_var = tk.StringVar(value="Frames: 0")
        ttk.Label(stats_frame, textvariable=self.frames_processed_var, style='Status.TLabel').pack(anchor=tk.W)
        
        self.faces_detected_var = tk.StringVar(value="Faces: 0")
        ttk.Label(stats_frame, textvariable=self.faces_detected_var, style='Status.TLabel').pack(anchor=tk.W)
        
        self.dominant_emotion_var = tk.StringVar(value="Emotion: None")
        ttk.Label(stats_frame, textvariable=self.dominant_emotion_var, style='Status.TLabel').pack(anchor=tk.W)
        
        self.confidence_var = tk.StringVar(value="Confidence: 0%")
        ttk.Label(stats_frame, textvariable=self.confidence_var, style='Status.TLabel').pack(anchor=tk.W)
        
        # Alerts
        alerts_frame = ttk.LabelFrame(comp_frame, text="Alerts", padding=5)
        alerts_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.alerts_text = tk.Text(alerts_frame, height=8, width=30, wrap=tk.WORD)
        alerts_scroll = ttk.Scrollbar(alerts_frame, orient=tk.VERTICAL, command=self.alerts_text.yview)
        self.alerts_text.configure(yscrollcommand=alerts_scroll.set)
        
        self.alerts_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alerts_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Recent sessions
        sessions_frame = ttk.LabelFrame(comp_frame, text="Recent Sessions", padding=5)
        sessions_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.sessions_listbox = tk.Listbox(sessions_frame, height=6)
        self.sessions_listbox.pack(fill=tk.X)
        self.load_recent_sessions()
        
    def create_contentinfo(self, parent):
        """Create footer/contentinfo region"""
        footer_frame = ttk.Frame(parent, relief=tk.RAISED, borderwidth=1)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - CI Microexpression Tracking System")
        status_label = ttk.Label(footer_frame, textvariable=self.status_var, style='Status.TLabel')
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Version info
        version_label = ttk.Label(footer_frame, text="v1.0 | Secure Mode", style='Status.TLabel')
        version_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
    def setup_accessibility(self):
        """Setup accessibility features"""
        # Keyboard shortcuts
        self.root.bind('<Control-n>', lambda e: self.new_analysis())
        self.root.bind('<Control-o>', lambda e: self.load_video_file())
        self.root.bind('<Control-s>', lambda e: self.start_analysis())
        self.root.bind('<Control-q>', lambda e: self.on_closing())
        self.root.bind('<F1>', lambda e: self.show_help())
        
        # Focus management
        self.root.bind('<Tab>', self.handle_tab_navigation)
        
    def handle_tab_navigation(self, event):
        """Handle tab navigation for accessibility"""
        # Let tkinter handle default tab navigation
        return None
        
    def show_help(self):
        """Show help dialog"""
        help_text = """
CI MICROEXPRESSION TRACKING SYSTEM - HELP

Keyboard Shortcuts:
• Ctrl+N: New Analysis
• Ctrl+O: Load Video File
• Ctrl+S: Start Analysis
• Ctrl+Q: Quit Application
• F1: Show Help

Navigation:
• Use Tab to navigate between controls
• Use arrow keys in lists and trees
• Use Space/Enter to activate buttons

Analysis Workflow:
1. Set Analyst ID
2. Select video source (webcam or file)
3. Click "Start Analysis"
4. Monitor progress in real-time
5. Review results in Results tab
6. Generate reports as needed

Security Features:
• All processing is local (air-gapped)
• Audit logging for chain of custody
• Metadata stripping for security
• Session isolation and tracking

For technical support, consult the system documentation.
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - CI Tracking System")
        help_window.geometry("600x500")
        help_window.transient(self.root)
        help_window.grab_set()
        
        help_text_widget = scrolledtext.ScrolledText(help_window, wrap=tk.WORD)
        help_text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        help_text_widget.insert(tk.END, help_text)
        help_text_widget.configure(state=tk.DISABLED)
        
        ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)
        
    def new_analysis(self):
        """Start a new analysis session"""
        if self.is_analyzing:
            messagebox.showwarning("Analysis in Progress", "Please stop the current analysis before starting a new one.")
            return
            
        # Reset UI
        self.clear_results()
        self.update_status("Ready for new analysis")
        self.notebook.select(0)  # Switch to analysis tab
        
    def load_video_file(self):
        """Load a video file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(f"File: {Path(file_path).name}")
            self.source_var.set("file")
            self.video_source = file_path
            self.update_status(f"Video file loaded: {Path(file_path).name}")
            
    def use_webcam(self):
        """Configure webcam as video source"""
        self.source_var.set("webcam")
        self.video_source = 0
        self.file_path_var.set("Source: Webcam (Device 0)")
        self.update_status("Webcam selected as video source")
        
    def start_analysis(self):
        """Start the analysis process"""
        if self.is_analyzing:
            messagebox.showwarning("Analysis in Progress", "Analysis is already running.")
            return
            
        # Validate inputs
        analyst_id = self.analyst_entry.get().strip()
        if not analyst_id:
            messagebox.showerror("Missing Information", "Please enter an Analyst ID.")
            return
            
        if self.source_var.get() == "none":
            messagebox.showerror("No Source Selected", "Please select a video source (webcam or file).")
            return
            
        if self.source_var.get() == "file" and not self.video_source:
            messagebox.showerror("No File Selected", "Please select a video file.")
            return
            
        # Start analysis in separate thread
        self.is_analyzing = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.progress_bar.start()
        
        self.analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=(self.video_source, analyst_id),
            daemon=True
        )
        self.analysis_thread.start()
        
        self.update_status("Analysis started...")
        
    def run_analysis(self, source, analyst_id):
        """Run analysis in background thread"""
        try:
            # Initialize tracker
            self.tracker = CIMicroexpressionTracker(user_id=analyst_id)
            self.current_session = self.tracker.session_id
            
            # Update UI
            self.root.after(0, self.update_session_info)
            self.root.after(0, lambda: self.update_status("Initializing analysis components..."))
            
            # Run analysis
            results = self.tracker.analyze_video(source)
            
            # Update results
            self.root.after(0, lambda: self.display_results(results))
            self.root.after(0, lambda: self.update_status("Analysis completed successfully"))
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.root.after(0, lambda: self.update_status(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", error_msg))
            
        finally:
            self.root.after(0, self.analysis_finished)
            
    def stop_analysis(self):
        """Stop the current analysis"""
        if self.is_analyzing:
            self.is_analyzing = False
            self.update_status("Stopping analysis...")
            # Note: In a real implementation, you'd need to add proper cancellation support
            
    def analysis_finished(self):
        """Clean up after analysis completion"""
        self.is_analyzing = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.progress_bar.stop()
        
        if self.tracker:
            self.tracker.cleanup()
            
    def display_results(self, results):
        """Display analysis results"""
        # Switch to results tab
        self.notebook.select(1)
        
        # Update summary
        summary = self.generate_summary(results)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary)
        
        # Update detailed results
        self.populate_results_tree(results)
        
        # Add alert if high-severity mismatches found
        mismatches = results.get('mismatches', [])
        high_severity = [m for m in mismatches if m.get('severity', 0) > 0.7]
        
        if high_severity:
            alert_msg = f"⚠️ HIGH PRIORITY: {len(high_severity)} high-confidence behavioral inconsistencies detected\n"
            self.add_alert(alert_msg)
            
    def generate_summary(self, results):
        """Generate analysis summary text"""
        emotion_timeline = results.get('emotion_timeline')
        speech_segments = results.get('speech_segments', [])
        mismatches = results.get('mismatches', [])
        
        total_frames = len(emotion_timeline) if emotion_timeline is not None else 0
        high_severity_mismatches = len([m for m in mismatches if m.get('severity', 0) > 0.7])
        
        summary = f"""ANALYSIS SUMMARY
Session ID: {self.current_session}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROCESSING RESULTS:
• Total frames analyzed: {total_frames:,}
• Speech segments: {len(speech_segments)}
• Behavioral inconsistencies: {len(mismatches)}
• High-severity indicators: {high_severity_mismatches}

"""
        
        if emotion_timeline is not None and not emotion_timeline.empty and 'dominant_emotion' in emotion_timeline.columns:
            dominant_emotion = emotion_timeline['dominant_emotion'].mode().iloc[0]
            emotion_stability = emotion_timeline['dominant_emotion'].nunique()
            
            summary += f"""BEHAVIORAL ASSESSMENT:
• Primary emotional state: {dominant_emotion.upper()}
• Emotional variability: {'HIGH' if emotion_stability > 4 else 'MODERATE' if emotion_stability > 2 else 'LOW'}
• Deception indicators: {'PRESENT' if high_severity_mismatches > 0 else 'MINIMAL'}

"""
        
        if high_severity_mismatches > 0:
            summary += f"""⚠️ ANALYST ATTENTION REQUIRED:
{high_severity_mismatches} high-confidence behavioral inconsistencies detected.
Recommend detailed review of flagged segments.

"""
        else:
            summary += """✓ BASELINE BEHAVIOR:
No significant behavioral anomalies detected.
Subject exhibits consistent emotional-verbal alignment.

"""
            
        if results.get('report_path'):
            summary += f"📄 Report generated: {results['report_path'].name}\n"
            
        return summary
        
    def populate_results_tree(self, results):
        """Populate the results tree with detailed data"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        # Add emotion data
        emotion_timeline = results.get('emotion_timeline')
        if emotion_timeline is not None and not emotion_timeline.empty:
            for _, row in emotion_timeline.head(50).iterrows():  # Show first 50 entries
                timestamp = f"{row.get('timestamp', 0):.1f}s"
                emotion_type = "Emotion"
                description = f"Dominant: {row.get('dominant_emotion', 'Unknown')}"
                confidence = f"{max([row.get(col, 0) for col in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] if col in row]):.1f}%"
                
                self.results_tree.insert('', tk.END, values=(timestamp, emotion_type, description, confidence))
        
        # Add speech segments
        speech_segments = results.get('speech_segments', [])
        for segment in speech_segments[:20]:  # Show first 20 segments
            timestamp = f"{segment['start']:.1f}s"
            segment_type = "Speech"
            description = segment['text'][:50] + "..." if len(segment['text']) > 50 else segment['text']
            confidence = f"{segment.get('emotion_confidence', 0):.1f}%"
            
            self.results_tree.insert('', tk.END, values=(timestamp, segment_type, description, confidence))
        
        # Add mismatches
        mismatches = results.get('mismatches', [])
        for mismatch in mismatches:
            timestamp = f"{mismatch['start_time']:.1f}s"
            mismatch_type = "Mismatch"
            description = f"Text: {mismatch['text_sentiment']} | Visual: {mismatch['visual_emotion']}"
            confidence = f"{mismatch.get('severity', 0):.1f}"
            
            self.results_tree.insert('', tk.END, values=(timestamp, mismatch_type, description, confidence))
            
    def add_alert(self, message):
        """Add an alert message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        alert_text = f"[{timestamp}] {message}\n"
        self.alerts_text.insert(tk.END, alert_text)
        self.alerts_text.see(tk.END)
        
    def clear_results(self):
        """Clear all results displays"""
        self.summary_text.delete(1.0, tk.END)
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.alerts_text.delete(1.0, tk.END)
        
    def update_session_info(self):
        """Update session information display"""
        if self.current_session:
            self.session_id_label.configure(text=f"Session: {self.current_session[:8]}...")
        
        analyst_id = self.analyst_entry.get().strip()
        if analyst_id:
            self.analyst_label.configure(text=f"Analyst: {analyst_id}")
            
    def load_recent_sessions(self):
        """Load recent session list"""
        try:
            if OUTPUT_DIR.exists():
                sessions = []
                for session_dir in OUTPUT_DIR.glob("session_*"):
                    if session_dir.is_dir():
                        sessions.append(session_dir.name)
                
                sessions.sort(reverse=True)  # Most recent first
                
                for session in sessions[:10]:  # Show last 10 sessions
                    self.sessions_listbox.insert(tk.END, session)
        except Exception:
            pass  # Ignore errors loading sessions
            
    def view_reports(self):
        """Open reports directory"""
        try:
            if OUTPUT_DIR.exists():
                # Open file explorer to output directory
                import subprocess
                import sys
                
                if sys.platform == "win32":
                    subprocess.Popen(f'explorer "{OUTPUT_DIR}"')
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(OUTPUT_DIR)])
                else:
                    subprocess.Popen(["xdg-open", str(OUTPUT_DIR)])
            else:
                messagebox.showinfo("No Reports", "No reports directory found.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open reports directory: {e}")
            
    def verify_audit_log(self):
        """Verify audit log integrity"""
        try:
            from security.audit_log import audit_logger
            
            self.update_status("Verifying audit log integrity...")
            verification_result = audit_logger.verify_integrity()
            
            if verification_result['valid']:
                message = f"✅ Audit log verified: {verification_result['total_entries']} entries"
                messagebox.showinfo("Audit Verification", message)
            else:
                corrupted = len(verification_result.get('corrupted_entries', []))
                chain_breaks = len(verification_result.get('chain_breaks', []))
                message = f"❌ Audit log integrity compromised:\n• Corrupted entries: {corrupted}\n• Chain breaks: {chain_breaks}"
                messagebox.showerror("Audit Verification Failed", message)
                
            self.update_status("Audit verification completed")
            
        except Exception as e:
            messagebox.showerror("Verification Error", f"Could not verify audit log: {e}")
            
    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        
    def on_closing(self):
        """Handle application closing"""
        if self.is_analyzing:
            if messagebox.askokcancel("Analysis in Progress", "Analysis is running. Stop and quit?"):
                self.stop_analysis()
                self.root.after(1000, self.root.destroy)  # Give time for cleanup
        else:
            self.root.destroy()
            
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point for GUI application"""
    try:
        app = CITrackingGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start GUI application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
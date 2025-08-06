"""
Tamper-evident audit logging for chain of custody
"""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading

from config import SECURITY_CONFIG


class AuditLogger:
    """
    Append-only audit log with cryptographic integrity checks
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or SECURITY_CONFIG["audit_log_path"]
        self.hash_algorithm = SECURITY_CONFIG["hash_algorithm"]
        self._lock = threading.Lock()
        self._ensure_log_exists()
    
    def _ensure_log_exists(self):
        """Create audit log file if it doesn't exist"""
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Initialize with genesis entry
            genesis_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "SYSTEM_INIT",
                "description": "Audit log initialized",
                "user_id": "SYSTEM",
                "session_id": None,
                "data_hash": None,
                "previous_hash": "0" * 64
            }
            self._write_entry(genesis_entry)
    
    def log_event(self, event_type: str, description: str, 
                  user_id: str = "UNKNOWN", session_id: Optional[str] = None,
                  data: Optional[Dict[str, Any]] = None):
        """
        Log a security event with integrity protection
        
        Args:
            event_type: Type of event (e.g., VIDEO_INGEST, ANALYSIS_START)
            description: Human-readable description
            user_id: User identifier
            session_id: Session identifier
            data: Additional structured data
        """
        with self._lock:
            # Calculate hash of data if provided
            data_hash = None
            if data:
                data_str = json.dumps(data, sort_keys=True)
                data_hash = hashlib.new(self.hash_algorithm, 
                                      data_str.encode()).hexdigest()
            
            # Get previous entry hash for chaining
            previous_hash = self._get_last_hash()
            
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "description": description,
                "user_id": user_id,
                "session_id": session_id,
                "data_hash": data_hash,
                "previous_hash": previous_hash
            }
            
            self._write_entry(entry)
    
    def _write_entry(self, entry: Dict[str, Any]):
        """Write entry to log file with integrity hash"""
        # Calculate entry hash
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.new(self.hash_algorithm, 
                                entry_str.encode()).hexdigest()
        
        # Create final log line
        log_line = {
            "entry": entry,
            "entry_hash": entry_hash
        }
        
        # Append to file
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_line) + '\n')
    
    def _get_last_hash(self) -> str:
        """Get hash of the last entry for chaining"""
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = json.loads(lines[-1].strip())
                    return last_line["entry_hash"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        return "0" * 64
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire audit log
        
        Returns:
            Dict with verification results
        """
        results = {
            "valid": True,
            "total_entries": 0,
            "corrupted_entries": [],
            "chain_breaks": []
        }
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            previous_hash = "0" * 64
            
            for i, line in enumerate(lines):
                try:
                    log_entry = json.loads(line.strip())
                    entry = log_entry["entry"]
                    stored_hash = log_entry["entry_hash"]
                    
                    # Verify entry hash
                    entry_str = json.dumps(entry, sort_keys=True)
                    calculated_hash = hashlib.new(self.hash_algorithm,
                                                entry_str.encode()).hexdigest()
                    
                    if calculated_hash != stored_hash:
                        results["corrupted_entries"].append(i)
                        results["valid"] = False
                    
                    # Verify chain
                    if entry["previous_hash"] != previous_hash:
                        results["chain_breaks"].append(i)
                        results["valid"] = False
                    
                    previous_hash = stored_hash
                    results["total_entries"] += 1
                    
                except json.JSONDecodeError:
                    results["corrupted_entries"].append(i)
                    results["valid"] = False
        
        except FileNotFoundError:
            results["valid"] = False
            results["error"] = "Audit log file not found"
        
        return results
    
    def get_session_events(self, session_id: str) -> list:
        """Get all events for a specific session"""
        events = []
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    log_entry = json.loads(line.strip())
                    entry = log_entry["entry"]
                    if entry.get("session_id") == session_id:
                        events.append(entry)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return events


# Global audit logger instance
audit_logger = AuditLogger()


def log_video_ingest(file_path: str, file_hash: str, user_id: str, session_id: str):
    """Log video file ingestion"""
    audit_logger.log_event(
        "VIDEO_INGEST",
        f"Video file ingested: {Path(file_path).name}",
        user_id=user_id,
        session_id=session_id,
        data={"file_path": str(file_path), "file_hash": file_hash}
    )


def log_analysis_start(analysis_type: str, user_id: str, session_id: str):
    """Log start of analysis process"""
    audit_logger.log_event(
        "ANALYSIS_START",
        f"Started {analysis_type} analysis",
        user_id=user_id,
        session_id=session_id,
        data={"analysis_type": analysis_type}
    )


def log_report_generation(report_path: str, user_id: str, session_id: str):
    """Log report generation"""
    audit_logger.log_event(
        "REPORT_GENERATED",
        f"Report generated: {Path(report_path).name}",
        user_id=user_id,
        session_id=session_id,
        data={"report_path": str(report_path)}
    )


def log_security_event(event_description: str, user_id: str, session_id: str, 
                      severity: str = "INFO"):
    """Log security-related events"""
    audit_logger.log_event(
        f"SECURITY_{severity}",
        event_description,
        user_id=user_id,
        session_id=session_id,
        data={"severity": severity}
    )
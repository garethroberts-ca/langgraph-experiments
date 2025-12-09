#!/usr/bin/env python3
"""
LangGraph Advanced Example 36: Intelligent File Operations
===========================================================

This example demonstrates LangGraph patterns for file system operations:

- Batch file processing and transformation
- Content analysis and classification
- Duplicate detection and deduplication
- Intelligent file organisation
- Directory synchronisation
- File validation pipelines
- Backup and archive workflows

These patterns showcase how LangGraph can orchestrate complex file system
operations with proper error handling and state management.

Author: LangGraph Examples
"""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# =============================================================================
# Helper Utilities
# =============================================================================

def print_banner(title: str) -> None:
    """Print a formatted section banner."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width + "\n")


def log(message: str, indent: int = 0) -> None:
    """Print a log message with optional indentation."""
    prefix = "  " * indent
    print(f"{prefix}> {message}")


def reduce_list(left: list | None, right: list | None) -> list:
    """Merge two lists, handling None values."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right


def get_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# =============================================================================
# Demo 1: Batch File Processing Pipeline
# =============================================================================

def demo_batch_file_processing():
    """
    Process multiple files through a transformation pipeline.
    
    Demonstrates:
    - File discovery and filtering
    - Batch transformation
    - Progress tracking
    - Result aggregation
    """
    print_banner("Demo 1: Batch File Processing Pipeline")
    
    class BatchProcessState(TypedDict):
        source_directory: str
        file_pattern: str
        discovered_files: list[dict]
        processed_files: list[dict]
        failed_files: list[dict]
        summary: dict
    
    def discover_files(state: BatchProcessState) -> dict:
        """Discover files matching the pattern."""
        log("Discovering files...")
        
        # For demo, we simulate file discovery
        # In production, this would scan the actual filesystem
        simulated_files = [
            {"path": "/data/report_2024_01.csv", "size": 15420, "modified": "2024-01-15"},
            {"path": "/data/report_2024_02.csv", "size": 18932, "modified": "2024-02-15"},
            {"path": "/data/report_2024_03.csv", "size": 12847, "modified": "2024-03-15"},
            {"path": "/data/notes.txt", "size": 1024, "modified": "2024-03-20"},
            {"path": "/data/report_2024_04.csv", "size": 21503, "modified": "2024-04-15"},
        ]
        
        # Filter by pattern
        pattern = state["file_pattern"]
        matching = [f for f in simulated_files if pattern in f["path"]]
        
        log(f"Found {len(matching)} files matching '{pattern}'", indent=1)
        return {"discovered_files": matching}
    
    def validate_files(state: BatchProcessState) -> dict:
        """Validate files before processing."""
        log("Validating files...")
        
        valid_files = []
        invalid_files = []
        
        for file_info in state["discovered_files"]:
            # Simulate validation checks
            if file_info["size"] > 0:
                valid_files.append(file_info)
            else:
                invalid_files.append({
                    **file_info,
                    "error": "File is empty"
                })
        
        log(f"Valid: {len(valid_files)}, Invalid: {len(invalid_files)}", indent=1)
        return {
            "discovered_files": valid_files,
            "failed_files": invalid_files
        }
    
    def process_files(state: BatchProcessState) -> dict:
        """Process each file in the batch."""
        log("Processing files...")
        
        processed = []
        failed = list(state.get("failed_files", []))
        
        for file_info in state["discovered_files"]:
            try:
                # Simulate file processing
                log(f"Processing: {file_info['path']}", indent=1)
                
                # Simulated transformation result
                processed.append({
                    **file_info,
                    "status": "processed",
                    "output_path": file_info["path"].replace(".csv", "_processed.csv"),
                    "rows_processed": file_info["size"] // 50  # Simulated row count
                })
                
            except Exception as e:
                failed.append({
                    **file_info,
                    "error": str(e)
                })
        
        return {
            "processed_files": processed,
            "failed_files": failed
        }
    
    def generate_summary(state: BatchProcessState) -> dict:
        """Generate processing summary."""
        log("Generating summary...")
        
        total_size = sum(f["size"] for f in state["processed_files"])
        total_rows = sum(f.get("rows_processed", 0) for f in state["processed_files"])
        
        summary = {
            "total_files_discovered": len(state["discovered_files"]) + len(state["failed_files"]),
            "files_processed": len(state["processed_files"]),
            "files_failed": len(state["failed_files"]),
            "total_bytes_processed": total_size,
            "total_rows_processed": total_rows,
            "timestamp": datetime.now().isoformat()
        }
        
        return {"summary": summary}
    
    # Build graph
    builder = StateGraph(BatchProcessState)
    builder.add_node("discover", discover_files)
    builder.add_node("validate", validate_files)
    builder.add_node("process", process_files)
    builder.add_node("summarise", generate_summary)
    
    builder.add_edge(START, "discover")
    builder.add_edge("discover", "validate")
    builder.add_edge("validate", "process")
    builder.add_edge("process", "summarise")
    builder.add_edge("summarise", END)
    
    graph = builder.compile()
    
    result = graph.invoke({
        "source_directory": "/data",
        "file_pattern": ".csv",
        "discovered_files": [],
        "processed_files": [],
        "failed_files": [],
        "summary": {}
    })
    
    log("\nPROCESSING SUMMARY")
    log("=" * 40)
    for key, value in result["summary"].items():
        log(f"{key}: {value}", indent=1)


# =============================================================================
# Demo 2: Content Analysis and Classification
# =============================================================================

def demo_content_classification():
    """
    Analyse and classify files based on their content.
    
    Demonstrates:
    - Content extraction
    - AI-powered classification
    - Metadata enrichment
    """
    print_banner("Demo 2: Content Analysis and Classification")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    class ClassificationState(TypedDict):
        files_to_classify: list[dict]
        classified_files: list[dict]
        classification_stats: dict
    
    def extract_content(state: ClassificationState) -> dict:
        """Extract content from files for analysis."""
        log("Extracting file content...")
        
        # Simulate content extraction
        files_with_content = []
        for file_info in state["files_to_classify"]:
            files_with_content.append({
                **file_info,
                "content_preview": file_info.get("simulated_content", "")[:500]
            })
        
        log(f"Extracted content from {len(files_with_content)} files", indent=1)
        return {"files_to_classify": files_with_content}
    
    def classify_files(state: ClassificationState) -> dict:
        """Classify files using AI."""
        log("Classifying files...")
        
        classified = []
        
        for file_info in state["files_to_classify"]:
            content = file_info.get("content_preview", "")
            filename = file_info.get("filename", "unknown")
            
            # Use LLM for classification
            response = llm.invoke(
                f"""Classify this file based on its name and content preview.
                
Filename: {filename}
Content preview: {content[:200]}

Respond with a JSON object containing:
- category: one of [document, code, data, image, config, log, other]
- confidence: low/medium/high
- suggested_folder: appropriate folder name

JSON:"""
            )
            
            try:
                # Parse classification result
                classification = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
            except json.JSONDecodeError:
                classification = {
                    "category": "other",
                    "confidence": "low",
                    "suggested_folder": "unsorted"
                }
            
            classified.append({
                **file_info,
                "classification": classification
            })
            
            log(f"{filename} -> {classification['category']} ({classification['confidence']})", indent=1)
        
        return {"classified_files": classified}
    
    def calculate_stats(state: ClassificationState) -> dict:
        """Calculate classification statistics."""
        log("Calculating statistics...")
        
        category_counts = {}
        confidence_counts = {"low": 0, "medium": 0, "high": 0}
        
        for file_info in state["classified_files"]:
            cat = file_info["classification"]["category"]
            conf = file_info["classification"]["confidence"]
            
            category_counts[cat] = category_counts.get(cat, 0) + 1
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        return {
            "classification_stats": {
                "by_category": category_counts,
                "by_confidence": confidence_counts,
                "total_classified": len(state["classified_files"])
            }
        }
    
    # Build graph
    builder = StateGraph(ClassificationState)
    builder.add_node("extract", extract_content)
    builder.add_node("classify", classify_files)
    builder.add_node("stats", calculate_stats)
    
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "classify")
    builder.add_edge("classify", "stats")
    builder.add_edge("stats", END)
    
    graph = builder.compile()
    
    # Sample files to classify
    files = [
        {"filename": "quarterly_report.docx", "simulated_content": "Q3 Financial Report. Revenue increased by 15%..."},
        {"filename": "app.py", "simulated_content": "import flask\nfrom flask import Flask\napp = Flask(__name__)"},
        {"filename": "users.csv", "simulated_content": "id,name,email\n1,John,john@example.com\n2,Jane,jane@example.com"},
        {"filename": "config.yaml", "simulated_content": "database:\n  host: localhost\n  port: 5432"},
        {"filename": "error.log", "simulated_content": "[ERROR] 2024-01-15 10:32:15 Connection timeout..."},
    ]
    
    result = graph.invoke({
        "files_to_classify": files,
        "classified_files": [],
        "classification_stats": {}
    })
    
    log("\nCLASSIFICATION RESULTS")
    log("=" * 40)
    log("By Category:")
    for cat, count in result["classification_stats"]["by_category"].items():
        log(f"  {cat}: {count}", indent=1)
    
    log("By Confidence:")
    for conf, count in result["classification_stats"]["by_confidence"].items():
        log(f"  {conf}: {count}", indent=1)


# =============================================================================
# Demo 3: Duplicate Detection and Deduplication
# =============================================================================

def demo_duplicate_detection():
    """
    Detect and handle duplicate files.
    
    Demonstrates:
    - Hash-based duplicate detection
    - Similarity analysis
    - Deduplication strategies
    """
    print_banner("Demo 3: Duplicate Detection and Deduplication")
    
    class DedupeState(TypedDict):
        files_to_scan: list[dict]
        file_hashes: dict
        duplicate_groups: list[dict]
        dedup_actions: list[dict]
        space_saved: int
    
    def calculate_hashes(state: DedupeState) -> dict:
        """Calculate hashes for all files."""
        log("Calculating file hashes...")
        
        hashes = {}
        
        for file_info in state["files_to_scan"]:
            # Simulate hash calculation
            # In production, use actual file content
            simulated_hash = hashlib.md5(
                f"{file_info['filename']}{file_info['size']}".encode()
            ).hexdigest()
            
            file_info["hash"] = simulated_hash
            
            if simulated_hash not in hashes:
                hashes[simulated_hash] = []
            hashes[simulated_hash].append(file_info)
        
        log(f"Calculated {len(hashes)} unique hashes for {len(state['files_to_scan'])} files", indent=1)
        return {"file_hashes": hashes}
    
    def identify_duplicates(state: DedupeState) -> dict:
        """Identify groups of duplicate files."""
        log("Identifying duplicates...")
        
        duplicate_groups = []
        
        for hash_value, files in state["file_hashes"].items():
            if len(files) > 1:
                # Sort by modification date to identify original
                sorted_files = sorted(files, key=lambda x: x.get("modified", ""))
                
                duplicate_groups.append({
                    "hash": hash_value,
                    "original": sorted_files[0],
                    "duplicates": sorted_files[1:],
                    "total_wasted_space": sum(f["size"] for f in sorted_files[1:])
                })
        
        log(f"Found {len(duplicate_groups)} groups of duplicates", indent=1)
        return {"duplicate_groups": duplicate_groups}
    
    def plan_deduplication(state: DedupeState) -> dict:
        """Plan deduplication actions."""
        log("Planning deduplication...")
        
        actions = []
        total_space = 0
        
        for group in state["duplicate_groups"]:
            original = group["original"]
            
            for dup in group["duplicates"]:
                actions.append({
                    "action": "delete",
                    "file": dup["path"],
                    "reason": f"Duplicate of {original['path']}",
                    "space_freed": dup["size"]
                })
                total_space += dup["size"]
        
        log(f"Planned {len(actions)} deletion actions", indent=1)
        return {
            "dedup_actions": actions,
            "space_saved": total_space
        }
    
    # Build graph
    builder = StateGraph(DedupeState)
    builder.add_node("hash", calculate_hashes)
    builder.add_node("identify", identify_duplicates)
    builder.add_node("plan", plan_deduplication)
    
    builder.add_edge(START, "hash")
    builder.add_edge("hash", "identify")
    builder.add_edge("identify", "plan")
    builder.add_edge("plan", END)
    
    graph = builder.compile()
    
    # Sample files with duplicates
    files = [
        {"filename": "photo.jpg", "path": "/photos/photo.jpg", "size": 2048000, "modified": "2024-01-01"},
        {"filename": "photo_copy.jpg", "path": "/photos/backup/photo_copy.jpg", "size": 2048000, "modified": "2024-01-15"},
        {"filename": "document.pdf", "path": "/docs/document.pdf", "size": 512000, "modified": "2024-02-01"},
        {"filename": "document.pdf", "path": "/docs/old/document.pdf", "size": 512000, "modified": "2024-02-10"},
        {"filename": "unique.txt", "path": "/docs/unique.txt", "size": 1024, "modified": "2024-03-01"},
    ]
    
    result = graph.invoke({
        "files_to_scan": files,
        "file_hashes": {},
        "duplicate_groups": [],
        "dedup_actions": [],
        "space_saved": 0
    })
    
    log("\nDEDUPLICATION REPORT")
    log("=" * 40)
    log(f"Duplicate groups found: {len(result['duplicate_groups'])}")
    log(f"Files to remove: {len(result['dedup_actions'])}")
    log(f"Space to be freed: {result['space_saved'] / 1024:.1f} KB")
    
    log("\nPlanned Actions:")
    for action in result["dedup_actions"]:
        log(f"DELETE: {action['file']} ({action['space_freed']} bytes)", indent=1)


# =============================================================================
# Demo 4: Intelligent File Organisation
# =============================================================================

def demo_file_organisation():
    """
    Organise files into a structured directory hierarchy.
    
    Demonstrates:
    - Rule-based organisation
    - Date-based folder structure
    - Type-based categorisation
    """
    print_banner("Demo 4: Intelligent File Organisation")
    
    class OrganiseState(TypedDict):
        source_files: list[dict]
        organisation_rules: dict
        move_operations: list[dict]
        organised_structure: dict
    
    def analyse_files(state: OrganiseState) -> dict:
        """Analyse files to determine organisation."""
        log("Analysing files for organisation...")
        
        for file_info in state["source_files"]:
            # Extract metadata
            filename = file_info["filename"]
            extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            
            file_info["extension"] = extension
            file_info["year"] = file_info.get("modified", "2024-01-01")[:4]
            file_info["month"] = file_info.get("modified", "2024-01-01")[5:7]
        
        log(f"Analysed {len(state['source_files'])} files", indent=1)
        return {}
    
    def apply_rules(state: OrganiseState) -> dict:
        """Apply organisation rules to determine destinations."""
        log("Applying organisation rules...")
        
        rules = state["organisation_rules"]
        operations = []
        
        for file_info in state["source_files"]:
            ext = file_info["extension"]
            year = file_info["year"]
            month = file_info["month"]
            
            # Determine destination based on rules
            if ext in rules.get("images", []):
                dest_folder = f"Photos/{year}/{month}"
            elif ext in rules.get("documents", []):
                dest_folder = f"Documents/{year}"
            elif ext in rules.get("code", []):
                dest_folder = f"Code/{year}"
            elif ext in rules.get("data", []):
                dest_folder = f"Data/{year}/{month}"
            else:
                dest_folder = f"Other/{year}"
            
            operations.append({
                "source": file_info["path"],
                "destination": f"{dest_folder}/{file_info['filename']}",
                "category": dest_folder.split("/")[0]
            })
        
        log(f"Generated {len(operations)} move operations", indent=1)
        return {"move_operations": operations}
    
    def preview_structure(state: OrganiseState) -> dict:
        """Generate preview of organised structure."""
        log("Generating structure preview...")
        
        structure = {}
        
        for op in state["move_operations"]:
            folder = "/".join(op["destination"].split("/")[:-1])
            if folder not in structure:
                structure[folder] = []
            structure[folder].append(op["destination"].split("/")[-1])
        
        return {"organised_structure": structure}
    
    # Build graph
    builder = StateGraph(OrganiseState)
    builder.add_node("analyse", analyse_files)
    builder.add_node("rules", apply_rules)
    builder.add_node("preview", preview_structure)
    
    builder.add_edge(START, "analyse")
    builder.add_edge("analyse", "rules")
    builder.add_edge("rules", "preview")
    builder.add_edge("preview", END)
    
    graph = builder.compile()
    
    # Sample files
    files = [
        {"filename": "vacation.jpg", "path": "/downloads/vacation.jpg", "modified": "2024-07-15"},
        {"filename": "report.pdf", "path": "/downloads/report.pdf", "modified": "2024-03-20"},
        {"filename": "script.py", "path": "/downloads/script.py", "modified": "2024-05-10"},
        {"filename": "sales.csv", "path": "/downloads/sales.csv", "modified": "2024-08-01"},
        {"filename": "sunset.png", "path": "/downloads/sunset.png", "modified": "2024-07-20"},
        {"filename": "notes.docx", "path": "/downloads/notes.docx", "modified": "2024-06-15"},
    ]
    
    rules = {
        "images": ["jpg", "jpeg", "png", "gif", "bmp"],
        "documents": ["pdf", "doc", "docx", "txt", "rtf"],
        "code": ["py", "js", "ts", "java", "go", "rs"],
        "data": ["csv", "json", "xml", "xlsx"]
    }
    
    result = graph.invoke({
        "source_files": files,
        "organisation_rules": rules,
        "move_operations": [],
        "organised_structure": {}
    })
    
    log("\nORGANISED STRUCTURE")
    log("=" * 40)
    for folder, files in sorted(result["organised_structure"].items()):
        log(f"{folder}/")
        for f in files:
            log(f"  {f}", indent=1)


# =============================================================================
# Demo 5: Directory Synchronisation
# =============================================================================

def demo_directory_sync():
    """
    Synchronise two directories with conflict resolution.
    
    Demonstrates:
    - Directory comparison
    - Conflict detection
    - Sync strategy application
    """
    print_banner("Demo 5: Directory Synchronisation")
    
    class SyncState(TypedDict):
        source_dir: str
        target_dir: str
        source_files: list[dict]
        target_files: list[dict]
        sync_plan: dict
        conflicts: list[dict]
    
    def scan_directories(state: SyncState) -> dict:
        """Scan both directories."""
        log(f"Scanning directories...")
        log(f"Source: {state['source_dir']}", indent=1)
        log(f"Target: {state['target_dir']}", indent=1)
        
        # Simulated directory contents
        source_files = [
            {"name": "file1.txt", "size": 1024, "modified": "2024-03-15 10:00:00", "hash": "abc123"},
            {"name": "file2.txt", "size": 2048, "modified": "2024-03-16 14:30:00", "hash": "def456"},
            {"name": "file3.txt", "size": 512, "modified": "2024-03-17 09:00:00", "hash": "ghi789"},
            {"name": "newfile.txt", "size": 256, "modified": "2024-03-18 11:00:00", "hash": "jkl012"},
        ]
        
        target_files = [
            {"name": "file1.txt", "size": 1024, "modified": "2024-03-15 10:00:00", "hash": "abc123"},
            {"name": "file2.txt", "size": 2500, "modified": "2024-03-17 16:00:00", "hash": "modified"},
            {"name": "file3.txt", "size": 512, "modified": "2024-03-17 09:00:00", "hash": "ghi789"},
            {"name": "targetonly.txt", "size": 128, "modified": "2024-03-14 08:00:00", "hash": "mno345"},
        ]
        
        return {
            "source_files": source_files,
            "target_files": target_files
        }
    
    def compare_directories(state: SyncState) -> dict:
        """Compare directories and identify differences."""
        log("Comparing directories...")
        
        source_by_name = {f["name"]: f for f in state["source_files"]}
        target_by_name = {f["name"]: f for f in state["target_files"]}
        
        sync_plan = {
            "copy_to_target": [],
            "copy_to_source": [],
            "update_target": [],
            "update_source": [],
            "conflicts": [],
            "identical": []
        }
        
        conflicts = []
        
        # Check source files
        for name, src_file in source_by_name.items():
            if name not in target_by_name:
                sync_plan["copy_to_target"].append(src_file)
            elif src_file["hash"] != target_by_name[name]["hash"]:
                # Files differ - check which is newer
                src_time = src_file["modified"]
                tgt_time = target_by_name[name]["modified"]
                
                if src_time > tgt_time:
                    sync_plan["update_target"].append({
                        "file": name,
                        "source_modified": src_time,
                        "target_modified": tgt_time
                    })
                elif tgt_time > src_time:
                    sync_plan["update_source"].append({
                        "file": name,
                        "source_modified": src_time,
                        "target_modified": tgt_time
                    })
                else:
                    conflicts.append({
                        "file": name,
                        "reason": "Same timestamp but different content",
                        "source_hash": src_file["hash"],
                        "target_hash": target_by_name[name]["hash"]
                    })
            else:
                sync_plan["identical"].append(name)
        
        # Check for files only in target
        for name, tgt_file in target_by_name.items():
            if name not in source_by_name:
                sync_plan["copy_to_source"].append(tgt_file)
        
        log(f"Identical: {len(sync_plan['identical'])}", indent=1)
        log(f"Copy to target: {len(sync_plan['copy_to_target'])}", indent=1)
        log(f"Copy to source: {len(sync_plan['copy_to_source'])}", indent=1)
        log(f"Update target: {len(sync_plan['update_target'])}", indent=1)
        log(f"Conflicts: {len(conflicts)}", indent=1)
        
        return {
            "sync_plan": sync_plan,
            "conflicts": conflicts
        }
    
    def resolve_conflicts(state: SyncState) -> dict:
        """Apply conflict resolution strategy."""
        log("Resolving conflicts...")
        
        for conflict in state["conflicts"]:
            # Default strategy: keep both with renamed copy
            conflict["resolution"] = "keep_both"
            conflict["action"] = f"Rename target to {conflict['file']}.conflict"
            log(f"Conflict '{conflict['file']}': {conflict['action']}", indent=1)
        
        return {}
    
    # Build graph
    builder = StateGraph(SyncState)
    builder.add_node("scan", scan_directories)
    builder.add_node("compare", compare_directories)
    builder.add_node("resolve", resolve_conflicts)
    
    builder.add_edge(START, "scan")
    builder.add_edge("scan", "compare")
    builder.add_edge("compare", "resolve")
    builder.add_edge("resolve", END)
    
    graph = builder.compile()
    
    result = graph.invoke({
        "source_dir": "/home/user/documents",
        "target_dir": "/backup/documents",
        "source_files": [],
        "target_files": [],
        "sync_plan": {},
        "conflicts": []
    })
    
    log("\nSYNC PLAN SUMMARY")
    log("=" * 40)
    plan = result["sync_plan"]
    
    if plan["copy_to_target"]:
        log("Files to copy to target:")
        for f in plan["copy_to_target"]:
            log(f"  + {f['name']}", indent=1)
    
    if plan["copy_to_source"]:
        log("Files to copy to source:")
        for f in plan["copy_to_source"]:
            log(f"  + {f['name']}", indent=1)
    
    if plan["update_target"]:
        log("Files to update in target:")
        for f in plan["update_target"]:
            log(f"  ~ {f['file']}", indent=1)


# =============================================================================
# Demo 6: File Validation Pipeline
# =============================================================================

def demo_file_validation():
    """
    Validate files against defined rules and schemas.
    
    Demonstrates:
    - Multi-rule validation
    - Schema validation for structured files
    - Validation reporting
    """
    print_banner("Demo 6: File Validation Pipeline")
    
    class ValidationState(TypedDict):
        files_to_validate: list[dict]
        validation_rules: dict
        validation_results: list[dict]
        summary: dict
    
    def check_file_properties(state: ValidationState) -> dict:
        """Validate basic file properties."""
        log("Checking file properties...")
        
        results = []
        
        for file_info in state["files_to_validate"]:
            file_result = {
                "file": file_info["filename"],
                "checks": [],
                "passed": True
            }
            
            rules = state["validation_rules"]
            
            # Check file size
            max_size = rules.get("max_size_bytes", float("inf"))
            if file_info.get("size", 0) > max_size:
                file_result["checks"].append({
                    "rule": "max_size",
                    "passed": False,
                    "message": f"File exceeds maximum size of {max_size} bytes"
                })
                file_result["passed"] = False
            else:
                file_result["checks"].append({
                    "rule": "max_size",
                    "passed": True
                })
            
            # Check extension
            allowed_extensions = rules.get("allowed_extensions", [])
            ext = file_info["filename"].rsplit(".", 1)[-1].lower() if "." in file_info["filename"] else ""
            if allowed_extensions and ext not in allowed_extensions:
                file_result["checks"].append({
                    "rule": "extension",
                    "passed": False,
                    "message": f"Extension .{ext} not in allowed list"
                })
                file_result["passed"] = False
            else:
                file_result["checks"].append({
                    "rule": "extension",
                    "passed": True
                })
            
            # Check naming convention
            naming_pattern = rules.get("naming_pattern")
            if naming_pattern:
                import re
                if not re.match(naming_pattern, file_info["filename"]):
                    file_result["checks"].append({
                        "rule": "naming",
                        "passed": False,
                        "message": f"Filename doesn't match pattern {naming_pattern}"
                    })
                    file_result["passed"] = False
                else:
                    file_result["checks"].append({
                        "rule": "naming",
                        "passed": True
                    })
            
            results.append(file_result)
        
        log(f"Validated {len(results)} files", indent=1)
        return {"validation_results": results}
    
    def validate_content(state: ValidationState) -> dict:
        """Validate file content where applicable."""
        log("Validating content...")
        
        for result in state["validation_results"]:
            # Find original file info
            file_info = next(
                (f for f in state["files_to_validate"] if f["filename"] == result["file"]),
                {}
            )
            
            # Simulate content validation for CSV files
            if file_info.get("filename", "").endswith(".csv"):
                simulated_content = file_info.get("simulated_content", "")
                
                # Check for header row
                if simulated_content and not simulated_content.startswith("id,"):
                    result["checks"].append({
                        "rule": "csv_header",
                        "passed": False,
                        "message": "CSV missing expected header row"
                    })
                    result["passed"] = False
                else:
                    result["checks"].append({
                        "rule": "csv_header",
                        "passed": True
                    })
        
        return {}
    
    def generate_report(state: ValidationState) -> dict:
        """Generate validation summary report."""
        log("Generating report...")
        
        passed_count = sum(1 for r in state["validation_results"] if r["passed"])
        failed_count = len(state["validation_results"]) - passed_count
        
        failed_checks = {}
        for result in state["validation_results"]:
            for check in result["checks"]:
                if not check["passed"]:
                    rule = check["rule"]
                    failed_checks[rule] = failed_checks.get(rule, 0) + 1
        
        summary = {
            "total_files": len(state["validation_results"]),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": round(passed_count / len(state["validation_results"]) * 100, 1),
            "failed_checks_by_rule": failed_checks
        }
        
        return {"summary": summary}
    
    # Build graph
    builder = StateGraph(ValidationState)
    builder.add_node("properties", check_file_properties)
    builder.add_node("content", validate_content)
    builder.add_node("report", generate_report)
    
    builder.add_edge(START, "properties")
    builder.add_edge("properties", "content")
    builder.add_edge("content", "report")
    builder.add_edge("report", END)
    
    graph = builder.compile()
    
    # Sample files and rules
    files = [
        {"filename": "data_2024.csv", "size": 1024, "simulated_content": "id,name,value\n1,test,100"},
        {"filename": "report.pdf", "size": 2048000},
        {"filename": "invalid file.txt", "size": 512},
        {"filename": "data_2023.csv", "size": 15000000, "simulated_content": "name,id\ntest,1"},
    ]
    
    rules = {
        "max_size_bytes": 10000000,  # 10MB
        "allowed_extensions": ["csv", "pdf", "txt"],
        "naming_pattern": r"^[a-z0-9_]+\.[a-z]+$"  # lowercase with underscores only
    }
    
    result = graph.invoke({
        "files_to_validate": files,
        "validation_rules": rules,
        "validation_results": [],
        "summary": {}
    })
    
    log("\nVALIDATION REPORT")
    log("=" * 40)
    log(f"Total files: {result['summary']['total_files']}")
    log(f"Passed: {result['summary']['passed']}")
    log(f"Failed: {result['summary']['failed']}")
    log(f"Pass rate: {result['summary']['pass_rate']}%")
    
    if result['summary']['failed_checks_by_rule']:
        log("\nFailed checks by rule:")
        for rule, count in result['summary']['failed_checks_by_rule'].items():
            log(f"  {rule}: {count}", indent=1)
    
    log("\nDetailed Results:")
    for r in result["validation_results"]:
        status = "PASS" if r["passed"] else "FAIL"
        log(f"  {r['file']}: {status}", indent=1)


# =============================================================================
# Demo 7: Backup and Archive Workflow
# =============================================================================

def demo_backup_workflow():
    """
    Orchestrate backup and archival operations.
    
    Demonstrates:
    - Incremental backup detection
    - Compression and archival
    - Retention policy enforcement
    """
    print_banner("Demo 7: Backup and Archive Workflow")
    
    class BackupState(TypedDict):
        source_directory: str
        backup_directory: str
        retention_days: int
        files_to_backup: list[dict]
        last_backup_manifest: dict
        backup_plan: dict
        archive_candidates: list[dict]
    
    def load_manifest(state: BackupState) -> dict:
        """Load last backup manifest."""
        log("Loading backup manifest...")
        
        # Simulated manifest from previous backup
        manifest = {
            "timestamp": "2024-03-15T10:00:00",
            "files": {
                "report.docx": {"hash": "abc123", "backed_up": "2024-03-15"},
                "data.csv": {"hash": "def456", "backed_up": "2024-03-10"},
                "image.png": {"hash": "ghi789", "backed_up": "2024-03-01"}
            }
        }
        
        log(f"Loaded manifest with {len(manifest['files'])} entries", indent=1)
        return {"last_backup_manifest": manifest}
    
    def scan_for_changes(state: BackupState) -> dict:
        """Identify files that need backing up."""
        log("Scanning for changes...")
        
        # Simulated current files
        current_files = [
            {"name": "report.docx", "hash": "abc123", "modified": "2024-03-15", "size": 25600},
            {"name": "data.csv", "hash": "xyz999", "modified": "2024-03-18", "size": 51200},  # Modified
            {"name": "image.png", "hash": "ghi789", "modified": "2024-03-01", "size": 102400},
            {"name": "newfile.txt", "hash": "new123", "modified": "2024-03-17", "size": 1024},  # New
        ]
        
        manifest = state["last_backup_manifest"]["files"]
        
        to_backup = []
        for file_info in current_files:
            name = file_info["name"]
            
            if name not in manifest:
                file_info["backup_reason"] = "new"
                to_backup.append(file_info)
            elif file_info["hash"] != manifest[name]["hash"]:
                file_info["backup_reason"] = "modified"
                to_backup.append(file_info)
        
        log(f"Files to backup: {len(to_backup)} (new: {sum(1 for f in to_backup if f['backup_reason'] == 'new')}, modified: {sum(1 for f in to_backup if f['backup_reason'] == 'modified')})", indent=1)
        return {"files_to_backup": to_backup}
    
    def create_backup_plan(state: BackupState) -> dict:
        """Create detailed backup plan."""
        log("Creating backup plan...")
        
        total_size = sum(f["size"] for f in state["files_to_backup"])
        
        plan = {
            "type": "incremental",
            "files": len(state["files_to_backup"]),
            "total_bytes": total_size,
            "estimated_compressed": int(total_size * 0.6),  # Estimate 40% compression
            "destination": f"{state['backup_directory']}/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz",
            "operations": [
                {"file": f["name"], "action": "backup", "reason": f["backup_reason"]}
                for f in state["files_to_backup"]
            ]
        }
        
        log(f"Backup type: {plan['type']}", indent=1)
        log(f"Estimated size: {plan['estimated_compressed'] / 1024:.1f} KB compressed", indent=1)
        
        return {"backup_plan": plan}
    
    def identify_archives(state: BackupState) -> dict:
        """Identify old backups for archival based on retention policy."""
        log("Identifying archive candidates...")
        
        # Simulated existing backups
        existing_backups = [
            {"name": "backup_20240301_100000.tar.gz", "date": "2024-03-01", "size": 51200},
            {"name": "backup_20240308_100000.tar.gz", "date": "2024-03-08", "size": 52100},
            {"name": "backup_20240315_100000.tar.gz", "date": "2024-03-15", "size": 53400},
        ]
        
        retention_days = state["retention_days"]
        cutoff_date = datetime.now().strftime("%Y-%m-%d")  # Simplified for demo
        
        # For demo, mark older backups for archival
        to_archive = [
            b for b in existing_backups
            if b["date"] < "2024-03-10"  # Simplified date comparison
        ]
        
        log(f"Backups eligible for archival: {len(to_archive)}", indent=1)
        return {"archive_candidates": to_archive}
    
    # Build graph
    builder = StateGraph(BackupState)
    builder.add_node("manifest", load_manifest)
    builder.add_node("scan", scan_for_changes)
    builder.add_node("plan", create_backup_plan)
    builder.add_node("archive", identify_archives)
    
    builder.add_edge(START, "manifest")
    builder.add_edge("manifest", "scan")
    builder.add_edge("scan", "plan")
    builder.add_edge("plan", "archive")
    builder.add_edge("archive", END)
    
    graph = builder.compile()
    
    result = graph.invoke({
        "source_directory": "/home/user/documents",
        "backup_directory": "/backups",
        "retention_days": 30,
        "files_to_backup": [],
        "last_backup_manifest": {},
        "backup_plan": {},
        "archive_candidates": []
    })
    
    log("\nBACKUP SUMMARY")
    log("=" * 40)
    plan = result["backup_plan"]
    log(f"Backup type: {plan['type']}")
    log(f"Files to backup: {plan['files']}")
    log(f"Total size: {plan['total_bytes'] / 1024:.1f} KB")
    log(f"Estimated compressed: {plan['estimated_compressed'] / 1024:.1f} KB")
    log(f"Destination: {plan['destination']}")
    
    log("\nBackup Operations:")
    for op in plan["operations"]:
        log(f"  {op['action'].upper()}: {op['file']} ({op['reason']})", indent=1)
    
    if result["archive_candidates"]:
        log("\nArchive Candidates:")
        for archive in result["archive_candidates"]:
            log(f"  {archive['name']} ({archive['date']})", indent=1)


# =============================================================================
# Demo 8: Real File Operations (with temp directory)
# =============================================================================

def demo_real_file_operations():
    """
    Demonstrate actual file operations using a temporary directory.
    
    Demonstrates:
    - Real file creation and manipulation
    - Actual directory operations
    - File content processing
    """
    print_banner("Demo 8: Real File Operations")
    
    class RealFileState(TypedDict):
        working_dir: str
        created_files: list[str]
        processed_files: list[dict]
        cleanup_complete: bool
    
    def setup_workspace(state: RealFileState) -> dict:
        """Create temporary workspace with sample files."""
        log("Setting up workspace...")
        
        work_dir = state["working_dir"]
        os.makedirs(work_dir, exist_ok=True)
        
        # Create sample files
        created = []
        
        sample_files = {
            "sample1.txt": "This is the first sample file.\nIt contains multiple lines.\n",
            "sample2.txt": "Second sample file with different content.\n",
            "data.csv": "id,name,value\n1,alpha,100\n2,beta,200\n3,gamma,300\n",
            "config.json": '{"setting1": "value1", "setting2": 42}\n'
        }
        
        for filename, content in sample_files.items():
            filepath = os.path.join(work_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
            created.append(filepath)
            log(f"Created: {filename}", indent=1)
        
        return {"created_files": created}
    
    def process_files(state: RealFileState) -> dict:
        """Process the created files."""
        log("Processing files...")
        
        processed = []
        
        for filepath in state["created_files"]:
            filename = os.path.basename(filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            file_info = {
                "path": filepath,
                "filename": filename,
                "size": os.path.getsize(filepath),
                "lines": len(content.splitlines()),
                "characters": len(content),
                "extension": filename.rsplit('.', 1)[-1] if '.' in filename else ''
            }
            
            # Calculate hash
            file_info["md5"] = hashlib.md5(content.encode()).hexdigest()[:8]
            
            processed.append(file_info)
            log(f"Processed: {filename} ({file_info['size']} bytes, {file_info['lines']} lines)", indent=1)
        
        return {"processed_files": processed}
    
    def transform_files(state: RealFileState) -> dict:
        """Apply transformations to files."""
        log("Transforming files...")
        
        work_dir = state["working_dir"]
        
        for file_info in state["processed_files"]:
            if file_info["extension"] == "txt":
                # Create uppercase version
                with open(file_info["path"], 'r') as f:
                    content = f.read()
                
                upper_path = os.path.join(work_dir, f"upper_{file_info['filename']}")
                with open(upper_path, 'w') as f:
                    f.write(content.upper())
                
                log(f"Created uppercase version: upper_{file_info['filename']}", indent=1)
        
        return {}
    
    def cleanup_workspace(state: RealFileState) -> dict:
        """Clean up the temporary workspace."""
        log("Cleaning up workspace...")
        
        work_dir = state["working_dir"]
        
        # List all files before cleanup
        all_files = os.listdir(work_dir)
        log(f"Files in workspace: {len(all_files)}", indent=1)
        
        # Remove all files and directory
        shutil.rmtree(work_dir)
        log("Workspace cleaned up", indent=1)
        
        return {"cleanup_complete": True}
    
    # Build graph
    builder = StateGraph(RealFileState)
    builder.add_node("setup", setup_workspace)
    builder.add_node("process", process_files)
    builder.add_node("transform", transform_files)
    builder.add_node("cleanup", cleanup_workspace)
    
    builder.add_edge(START, "setup")
    builder.add_edge("setup", "process")
    builder.add_edge("process", "transform")
    builder.add_edge("transform", "cleanup")
    builder.add_edge("cleanup", END)
    
    graph = builder.compile()
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp(prefix="langgraph_demo_")
    
    result = graph.invoke({
        "working_dir": temp_dir,
        "created_files": [],
        "processed_files": [],
        "cleanup_complete": False
    })
    
    log("\nFILE PROCESSING RESULTS")
    log("=" * 40)
    for file_info in result["processed_files"]:
        log(f"{file_info['filename']}:")
        log(f"  Size: {file_info['size']} bytes", indent=1)
        log(f"  Lines: {file_info['lines']}", indent=1)
        log(f"  MD5: {file_info['md5']}", indent=1)
    
    log(f"\nCleanup complete: {result['cleanup_complete']}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all file operations demonstrations."""
    print("\n" + "=" * 70)
    print(" LangGraph: Intelligent File Operations ".center(70))
    print("=" * 70)
    
    demo_batch_file_processing()
    demo_content_classification()
    demo_duplicate_detection()
    demo_file_organisation()
    demo_directory_sync()
    demo_file_validation()
    demo_backup_workflow()
    demo_real_file_operations()
    
    print("\n" + "=" * 70)
    print(" All File Operations Demonstrations Complete ".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

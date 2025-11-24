#!/usr/bin/env python3
"""
Data Backup Script for Reddit Sentiment Analyzer
Creates backups of database and important files
"""

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
import json
import logging

def setup_logging():
    """Setup logging for backup operations"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/backup.log'),
            logging.StreamHandler()
        ]
    )

def create_backup():
    """Create a comprehensive backup of application data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backups/backup_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"🔄 Starting backup to {backup_dir}")
    
    try:
        # Backup database
        backup_database(backup_dir)
        
        # Backup configuration files
        backup_configurations(backup_dir)
        
        # Backup analysis results
        backup_analysis_data(backup_dir)
        
        # Create backup manifest
        create_manifest(backup_dir)
        
        logging.info(f"✅ Backup completed successfully: {backup_dir}")
        return backup_dir
        
    except Exception as e:
        logging.error(f"❌ Backup failed: {e}")
        raise

def backup_database(backup_dir: Path):
    """Backup SQLite database"""
    db_source = Path("data/database/reddit_data.db")
    if db_source.exists():
        db_backup = backup_dir / "database"
        db_backup.mkdir(exist_ok=True)
        
        # Copy database file
        shutil.copy2(db_source, db_backup / "reddit_data.db")
        
        # Also create SQL dump
        create_sql_dump(db_source, db_backup / "database_dump.sql")
        
        logging.info("💾 Database backed up successfully")

def create_sql_dump(db_path: Path, dump_path: Path):
    """Create SQL dump of database"""
    try:
        conn = sqlite3.connect(db_path)
        with open(dump_path, 'w') as f:
            for line in conn.iterdump():
                f.write(f'{line}\n')
        conn.close()
    except Exception as e:
        logging.warning(f"Could not create SQL dump: {e}")

def backup_configurations(backup_dir: Path):
    """Backup configuration files"""
    config_files = [
        "config.py",
        "scraper_config.json",
        ".env",
        "requirements.txt"
    ]
    
    config_backup = backup_dir / "configurations"
    config_backup.mkdir(exist_ok=True)
    
    for config_file in config_files:
        source = Path(config_file)
        if source.exists():
            shutil.copy2(source, config_backup / source.name)
    
    logging.info("⚙️ Configuration files backed up")

def backup_analysis_data(backup_dir: Path):
    """Backup analysis results and exports"""
    data_dirs = [
        "data/exports",
        "data/scraped_data/processed",
        "data/models"
    ]
    
    data_backup = backup_dir / "analysis_data"
    data_backup.mkdir(exist_ok=True)
    
    for data_dir in data_dirs:
        source = Path(data_dir)
        if source.exists() and any(source.iterdir()):
            dest = data_backup / source.name
            shutil.copytree(source, dest, dirs_exist_ok=True)
    
    logging.info("📊 Analysis data backed up")

def create_manifest(backup_dir: Path):
    """Create backup manifest file"""
    manifest = {
        "backup_timestamp": datetime.now().isoformat(),
        "backup_version": "1.0",
        "contents": [],
        "total_size": 0
    }
    
    # Calculate total size and list contents
    total_size = 0
    for file_path in backup_dir.rglob('*'):
        if file_path.is_file():
            file_size = file_path.stat().st_size
            total_size += file_size
            manifest["contents"].append({
                "path": str(file_path.relative_to(backup_dir)),
                "size": file_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    manifest["total_size"] = total_size
    
    # Save manifest
    with open(backup_dir / "backup_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logging.info(f"📋 Backup manifest created (total size: {total_size / 1024 / 1024:.2f} MB)")

def cleanup_old_backups(max_backups: int = 10):
    """Clean up old backups, keeping only the most recent ones"""
    backup_dir = Path("backups")
    if not backup_dir.exists():
        return
    
    backups = sorted(
        [d for d in backup_dir.iterdir() if d.is_dir() and d.name.startswith("backup_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if len(backups) > max_backups:
        for old_backup in backups[max_backups:]:
            shutil.rmtree(old_backup)
            logging.info(f"🗑️ Removed old backup: {old_backup.name}")

if __name__ == "__main__":
    setup_logging()
    
    try:
        backup_path = create_backup()
        cleanup_old_backups()
        print(f"🎉 Backup completed: {backup_path}")
    except Exception as e:
        print(f"❌ Backup failed: {e}")
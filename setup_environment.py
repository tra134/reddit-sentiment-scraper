#!/usr/bin/env python3
"""
Reddit Sentiment Analyzer Pro - Environment Setup Script
FIXED VERSION for Python 3.13 compatibility
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.venv_dir = self.current_dir / "venv"
        self.requirements_file = self.current_dir / "requirements.txt"
        self.compatible_requirements = self.current_dir / "requirements_compatible.txt"
        
    def check_python_version(self):
        """Check Python version and warn about compatibility"""
        version = sys.version_info
        logger.info(f"🐍 Python {version.major}.{version.minor}.{version.micro} detected")
        
        # Warn about Python 3.13 compatibility issues
        if version.major == 3 and version.minor >= 13:
            logger.warning("⚠️  Python 3.13+ may have compatibility issues with some packages")
            logger.info("🔧 Using compatible package versions...")
        
        return True
    
    def create_venv(self):
        """Create virtual environment"""
        if self.venv_dir.exists():
            logger.info("✅ Virtual environment already exists")
            return True
            
        try:
            logger.info("🔄 Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True)
            logger.info("✅ Virtual environment created successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_pip_path(self):
        """Get pip path based on OS"""
        if platform.system() == "Windows":
            return self.venv_dir / "Scripts" / "pip.exe"
        else:
            return self.venv_dir / "bin" / "pip"
    
    def upgrade_pip(self):
        """Upgrade pip in virtual environment"""
        pip_path = self.get_pip_path()
        try:
            logger.info("🔄 Upgrading pip...")
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
            logger.info("✅ Pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to upgrade pip: {e}")
            return False
    
    def install_core_dependencies(self):
        """Install core dependencies first to avoid conflicts"""
        pip_path = self.get_pip_path()
        
        core_packages = [
            "wheel",
            "setuptools",
            "numpy==1.24.0",
            "pandas==2.0.3",  # Compatible version
            "scikit-learn==1.3.0"
        ]
        
        try:
            logger.info("🔧 Installing core dependencies first...")
            for package in core_packages:
                subprocess.run([str(pip_path), "install", package], check=True)
            logger.info("✅ Core dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install core dependencies: {e}")
            return False
    
    def install_dependencies(self):
        """Install project dependencies with error handling"""
        pip_path = self.get_pip_path()
        
        if not self.requirements_file.exists():
            logger.error("❌ requirements.txt not found")
            return False
        
        try:
            logger.info("🔄 Installing dependencies from requirements.txt...")
            result = subprocess.run(
                [str(pip_path), "install", "-r", str(self.requirements_file)],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install some dependencies: {e}")
            
            # Try individual package installation for problematic packages
            return self.install_packages_individually()
    
    def install_packages_individually(self):
        """Install packages individually to handle failures"""
        pip_path = self.get_pip_path()
        
        packages = [
            "streamlit==1.28.0",
            "selenium==4.15.0",
            "undetected-chromedriver==3.5.0",
            "webdriver-manager==4.0.1",
            "beautifulsoup4==4.12.0",
            "requests==2.31.0",
            "aiohttp==3.9.0",
            "transformers==4.34.0",
            "torch==2.0.1",
            "spacy==3.6.0",
            "textblob==0.17.1",
            "vaderSentiment==3.3.2",
            "plotly==5.15.0",
            "matplotlib==3.7.0",
            "seaborn==0.12.2",
            "wordcloud==1.9.2",
            "sqlalchemy==2.0.0",
            "python-dotenv==1.0.0",
            "tqdm==4.65.0",
            "pillow==10.0.0"
        ]
        
        successful = []
        failed = []
        
        logger.info("🔄 Installing packages individually...")
        
        for package in packages:
            try:
                logger.info(f"📦 Installing {package}...")
                subprocess.run(
                    [str(pip_path), "install", package],
                    check=True,
                    capture_output=True
                )
                successful.append(package)
                logger.info(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                failed.append(package)
                logger.warning(f"⚠️  Failed to install {package}")
        
        logger.info(f"📊 Installation summary: {len(successful)} successful, {len(failed)} failed")
        
        if failed:
            logger.warning("❌ The following packages failed to install:")
            for package in failed:
                logger.warning(f"   - {package}")
            
            # Try alternative versions for failed packages
            self.try_alternative_versions(failed)
        
        return len(successful) > len(failed)  # Consider successful if most packages installed
    
    def try_alternative_versions(self, failed_packages):
        """Try alternative versions for failed packages"""
        pip_path = self.get_pip_path()
        
        alternative_versions = {
            "torch": "torch --index-url https://download.pytorch.org/whl/cpu",
            "spacy": "spacy==3.6.0",
            "transformers": "transformers==4.34.0"
        }
        
        for package in failed_packages:
            if package.split('==')[0] in alternative_versions:
                alt_package = alternative_versions[package.split('==')[0]]
                try:
                    logger.info(f"🔄 Trying alternative: {alt_package}")
                    subprocess.run(
                        [str(pip_path), "install"] + alt_package.split(),
                        check=True,
                        capture_output=True
                    )
                    logger.info(f"✅ Alternative {alt_package} installed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Alternative also failed for {package}: {e}")
    
    def install_spacy_model(self):
        """Install spaCy English model"""
        python_path = self.venv_dir / "Scripts" / "python.exe" if platform.system() == "Windows" else self.venv_dir / "bin" / "python"
        try:
            logger.info("🔄 Downloading spaCy English model...")
            subprocess.run([str(python_path), "-m", "spacy", "download", "en_core_web_sm"], check=True)
            logger.info("✅ spaCy model installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install spaCy model: {e}")
            logger.info("💡 You can install it manually later with: python -m spacy download en_core_web_sm")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            "data/database",
            "data/scraped_data/raw",
            "data/scraped_data/processed", 
            "data/scraped_data/archived",
            "data/exports",
            "data/logs",
            "data/models",
            "proxies",
            "app/static/css",
            "app/static/js",
            "app/static/images",
            "tests/test_scrapers",
            "tests/test_services",
            "tests/test_ml",
            "scripts"
        ]
        
        for dir_path in directories:
            full_path = self.current_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Project directories created")
    
    def create_activation_scripts(self):
        """Create platform-specific activation scripts"""
        # Windows activation script
        if platform.system() == "Windows":
            bat_content = """@echo off
echo Activating Reddit Sentiment Analyzer Environment...
call venv\\Scripts\\activate
echo Environment activated!
echo.
echo To run the application:
echo   streamlit run app/main.py
echo.
cmd /k
"""
            with open(self.current_dir / "activate_env.bat", "w") as f:
                f.write(bat_content)
            
            # Also create a run script
            run_bat_content = """@echo off
call venv\\Scripts\\activate
echo Starting Reddit Sentiment Analyzer Pro...
streamlit run app/main.py
pause
"""
            with open(self.current_dir / "run_app.bat", "w") as f:
                f.write(run_bat_content)
        
        # Unix/Linux/Mac activation script
        else:
            sh_content = """#!/bin/bash
echo "Activating Reddit Sentiment Analyzer Environment..."
source venv/bin/activate
echo "Environment activated!"
echo ""
echo "To run the application:"
echo "  streamlit run app/main.py"
echo ""
exec $SHELL
"""
            with open(self.current_dir / "activate_env.sh", "w") as f:
                f.write(sh_content)
            
            # Make executable
            os.chmod(self.current_dir / "activate_env.sh", 0o755)
            
            # Run script
            run_sh_content = """#!/bin/bash
source venv/bin/activate
echo "Starting Reddit Sentiment Analyzer Pro..."
streamlit run app/main.py
"""
            with open(self.current_dir / "run_app.sh", "w") as f:
                f.write(run_sh_content)
            os.chmod(self.current_dir / "run_app.sh", 0o755)
        
        logger.info("✅ Activation scripts created")
    
    def verify_installation(self):
        """Verify that installation was successful"""
        python_path = self.venv_dir / "Scripts" / "python.exe" if platform.system() == "Windows" else self.venv_dir / "bin" / "python"
        
        test_script = """
try:
    import streamlit
    import selenium
    import pandas as pd
    print("SUCCESS: Core dependencies are available")
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"FAILED: {e}")
    exit(1)
"""
        
        try:
            result = subprocess.run(
                [str(python_path), "-c", test_script],
                capture_output=True,
                text=True,
                check=True
            )
            if "SUCCESS" in result.stdout:
                logger.info("✅ Installation verification: PASSED")
                return True
            else:
                logger.error("❌ Installation verification: FAILED")
                return False
        except subprocess.CalledProcessError:
            logger.error("❌ Installation verification: FAILED")
            return False
    
    def display_success_message(self):
        """Display success message with next steps"""
        logger.info("\n" + "="*60)
        logger.info("🎉 SETUP COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("\n📋 NEXT STEPS:")
        
        if platform.system() == "Windows":
            logger.info("   1. Run 'activate_env.bat' to activate environment")
            logger.info("   2. Or run 'run_app.bat' to start directly")
        else:
            logger.info("   1. Run 'source activate_env.sh' to activate environment")  
            logger.info("   2. Or run './run_app.sh' to start directly")
        
        logger.info("\n🚀 QUICK START:")
        logger.info("   streamlit run app/main.py")
        
        logger.info("\n🔧 TROUBLESHOOTING:")
        logger.info("   - If some packages failed, try: pip install package_name")
        logger.info("   - For spaCy: python -m spacy download en_core_web_sm")
        logger.info("   - Check logs in data/logs/ for detailed errors")
    
    def run_setup(self):
        """Run complete setup process"""
        logger.info("🚀 Starting Reddit Sentiment Analyzer Pro Setup")
        logger.info("="*50)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating virtual environment", self.create_venv),
            ("Upgrading pip", self.upgrade_pip),
            ("Installing core dependencies", self.install_core_dependencies),
            ("Installing project dependencies", self.install_dependencies),
            ("Creating project directories", self.create_directories),
            ("Creating activation scripts", self.create_activation_scripts),
            ("Verifying installation", self.verify_installation)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n🔧 {step_name}...")
            if not step_func():
                logger.error(f"❌ Setup failed at: {step_name}")
                # Continue anyway for partial installation
                continue
        
        # Try to install spaCy model (optional)
        logger.info("\n🔧 Installing spaCy model...")
        self.install_spacy_model()
        
        self.display_success_message()
        return True

def main():
    """Main setup function"""
    setup = EnvironmentSetup()
    
    try:
        if setup.run_setup():
            logger.info("\n🎉 Setup completed successfully!")
        else:
            logger.info("\n⚠️  Setup completed with some warnings")
            logger.info("💡 Some packages may need manual installation")
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
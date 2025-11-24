"""
Reddit Sentiment Analyzer Pro Application Package
WITH WARNING FIXES
"""

import os
import sys
import warnings
from pathlib import Path

# ==================== SUPPRESS ALL WARNINGS ====================
# Add this to suppress NumPy and other warnings globally

# Suppress NumPy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.core.getlimits')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress other common warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow
os.environ['PYTHONWARNINGS'] = 'ignore'   # Suppress all Python warnings

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("✅ Application initialized - Warnings suppressed")

__version__ = "3.0.0"
__author__ = "Reddit Sentiment Analyzer Team"
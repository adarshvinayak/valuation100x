"""
Runtime Dependency Installer for Railway
Installs heavy ML dependencies after the container starts to bypass build timeouts
"""

import subprocess
import sys
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Heavy dependencies that cause Railway build timeouts
HEAVY_DEPENDENCIES = [
    "sentence-transformers==2.2.2",
    "faiss-cpu==1.7.4", 
    "llama-index==0.10.57",
    "llama-index-llms-openai==0.1.29",
    "llama-index-embeddings-openai==0.1.11",
    "llama-index-vector-stores-faiss==0.1.2"
]

def install_heavy_dependencies():
    """Install heavy ML dependencies at runtime"""
    try:
        logger.info("🔄 Installing heavy ML dependencies at runtime...")
        
        # Check if already installed
        try:
            import sentence_transformers
            import faiss
            logger.info("✅ Heavy dependencies already installed")
            return True
        except ImportError:
            pass
        
        # Install each dependency with timeout and retries
        for dep in HEAVY_DEPENDENCIES:
            logger.info(f"📦 Installing {dep}...")
            
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--no-cache-dir", "--timeout=300", "--retries=3",
                dep
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"❌ Failed to install {dep}: {result.stderr}")
                return False
            else:
                logger.info(f"✅ Successfully installed {dep}")
        
        logger.info("🎉 All heavy dependencies installed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Runtime installation failed: {e}")
        return False

def check_dependencies():
    """Check if heavy dependencies are available"""
    missing = []
    
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")
    
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
    
    try:
        import llama_index
    except ImportError:
        missing.append("llama-index")
    
    return missing

def install_if_needed():
    """Install heavy dependencies if they're missing"""
    missing = check_dependencies()
    
    if missing:
        logger.info(f"🔍 Missing dependencies: {missing}")
        return install_heavy_dependencies()
    else:
        logger.info("✅ All heavy dependencies are available")
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = install_if_needed()
    sys.exit(0 if success else 1)

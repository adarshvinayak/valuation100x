#!/usr/bin/env python3
"""
Railway Deployment Helper Script for DeepResearch
This script helps automate the Railway deployment process.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and return the result."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Success: {description}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"Command failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_railway_cli():
    """Check if Railway CLI is installed."""
    try:
        subprocess.run(["railway", "--version"], capture_output=True, check=True)
        print("✅ Railway CLI is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Railway CLI not found")
        return False

def install_railway_cli():
    """Install Railway CLI."""
    print("\n📦 Installing Railway CLI...")
    
    # Try npm first
    if run_command("npm install -g @railway/cli", "Installing Railway CLI via npm"):
        return True
    
    # If npm fails, provide manual instructions
    print("\n⚠️  npm installation failed. Please install Railway CLI manually:")
    print("1. Visit: https://docs.railway.app/develop/cli")
    print("2. Or try: curl -fsSL https://railway.app/install.sh | sh")
    print("3. Or download from: https://github.com/railwayapp/cli/releases")
    return False

def setup_environment_variables():
    """Guide user through setting up environment variables."""
    print("\n🔧 Environment Variables Setup")
    print("You'll need to set these in Railway dashboard or CLI:")
    
    required_vars = [
        "OPENAI_API_KEY",
        "FMP_API_KEY", 
        "TAVILY_API_KEY",
        "SEC_API_KEY"
    ]
    
    optional_vars = [
        "ALPHAVANTAGE_API_KEY",
        "POLYGON_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY"
    ]
    
    print("\n📋 Required Variables:")
    for var in required_vars:
        print(f"  - {var}")
    
    print("\n📋 Optional Variables:")
    for var in optional_vars:
        print(f"  - {var}")
    
    print("\n💡 Set variables using Railway CLI:")
    print("railway variables set OPENAI_API_KEY=your_key_here")
    print("railway variables set FMP_API_KEY=your_key_here")
    print("# ... and so on")
    
    print("\n💡 Or set them in Railway dashboard:")
    print("1. Go to your project in Railway dashboard")
    print("2. Click 'Variables' tab")
    print("3. Add each environment variable")

def main():
    """Main deployment workflow."""
    print("🚀 Railway Deployment Helper for DeepResearch")
    print("=" * 50)
    
    # Check current directory
    if not Path("railway.json").exists():
        print("❌ Error: railway.json not found. Make sure you're in the backend/ directory")
        sys.exit(1)
    
    print("✅ Found railway.json - you're in the right directory")
    
    # Check Railway CLI
    if not check_railway_cli():
        if not install_railway_cli():
            print("\n❌ Please install Railway CLI and run this script again")
            sys.exit(1)
    
    # Login to Railway
    print("\n🔐 Logging into Railway...")
    if not run_command("railway login", "Railway login"):
        print("❌ Login failed. Please try manually: railway login")
        sys.exit(1)
    
    # Link project (user needs to do this manually or provide repo URL)
    print("\n🔗 Linking to GitHub repository...")
    print("Please run this command manually:")
    print("railway link https://github.com/nocommitmentsyet/valuation100x")
    
    input("\nPress Enter after you've linked the repository...")
    
    # Deploy
    print("\n🚀 Deploying to Railway...")
    if run_command("railway up", "Deploying application"):
        print("\n🎉 Deployment initiated successfully!")
        
        # Get domain
        print("\n🌐 Getting your application URL...")
        run_command("railway domain", "Getting application domain")
        
        print("\n✅ Next steps:")
        print("1. Set up environment variables (see above)")
        print("2. Visit your Railway dashboard to monitor deployment")
        print("3. Check your app URL for health status: /api/health")
        
    else:
        print("\n❌ Deployment failed. Check the error messages above.")
    
    # Environment variables reminder
    setup_environment_variables()

if __name__ == "__main__":
    main()

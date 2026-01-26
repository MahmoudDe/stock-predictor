#!/usr/bin/env python3
"""
Helper script to set up Kaggle API credentials and download competition data.
Supports modern authentication methods (environment variables or access token file).
"""

import os
import sys
import subprocess
from pathlib import Path

def check_authentication():
    """Check if Kaggle authentication is already set up."""
    # Check environment variables
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        print("✓ Found Kaggle credentials in environment variables")
        return True
    
    # Check for access token file
    token_file = Path.home() / ".kaggle" / "access.token"
    if token_file.exists():
        print(f"✓ Found access token at {token_file}")
        return True
    
    # Check for legacy kaggle.json
    json_file = Path.home() / ".kaggle" / "kaggle.json"
    if json_file.exists():
        print(f"✓ Found legacy kaggle.json at {json_file}")
        return True
    
    return False

def setup_access_token():
    """Interactive setup for access token file."""
    print("\n=== Setting up Access Token File ===")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Scroll to 'API' section and click 'Create New Token'")
    print("3. Copy your API key")
    print()
    
    api_key = input("Enter your Kaggle API key: ").strip()
    
    if not api_key:
        print("✗ No API key provided. Exiting.")
        return False
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Write access token file
    token_file = kaggle_dir / "access.token"
    token_file.write_text(api_key)
    token_file.chmod(0o600)
    
    print(f"✓ Access token saved to {token_file}")
    return True

def setup_environment_variables():
    """Interactive setup for environment variables."""
    print("\n=== Setting up Environment Variables ===")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Scroll to 'API' section and click 'Create New Token'")
    print("3. Copy your username and API key")
    print()
    
    username = input("Enter your Kaggle username: ").strip()
    api_key = input("Enter your Kaggle API key: ").strip()
    
    if not username or not api_key:
        print("✗ Username and API key are required. Exiting.")
        return False
    
    # Create a shell script to export variables
    shell_script = Path.cwd() / "kaggle_env.sh"
    with open(shell_script, "w") as f:
        f.write(f"export KAGGLE_USERNAME={username}\n")
        f.write(f"export KAGGLE_KEY={api_key}\n")
    
    shell_script.chmod(0o600)
    
    print(f"✓ Environment variables script created: {shell_script}")
    print("\nTo use these credentials, run:")
    print(f"  source {shell_script}")
    print("\nOr add them to your ~/.zshrc or ~/.bashrc")
    return True

def download_data():
    """Download the competition data."""
    print("\n=== Downloading Competition Data ===")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download data
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "download", 
             "-c", "predicting-stock-trends-rise-or-fall", 
             "-p", "data"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Data downloaded successfully!")
        
        # Extract zip files
        zip_files = list(data_dir.glob("*.zip"))
        if zip_files:
            print("\nExtracting files...")
            import zipfile
            for zip_file in zip_files:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"✓ Extracted {zip_file.name}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download data: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Please install it with: pip install kaggle")
        return False

def main():
    """Main function."""
    print("Kaggle API Setup and Data Downloader")
    print("=" * 40)
    
    # Check if already authenticated
    if check_authentication():
        print("\nAuthentication already set up!")
        response = input("\nDownload competition data now? (y/n): ").strip().lower()
        if response == 'y':
            download_data()
        return
    
    # Setup authentication
    print("\nNo authentication found. Choose setup method:")
    print("1. Access Token File (Recommended - persists across sessions)")
    print("2. Environment Variables (Temporary - for current session)")
    print("3. Skip setup (use existing credentials)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        if setup_access_token():
            download_data()
    elif choice == "2":
        if setup_environment_variables():
            print("\n⚠ Note: You need to source the environment variables before downloading.")
            print("Run: source kaggle_env.sh")
            response = input("\nDownload data now? (y/n): ").strip().lower()
            if response == 'y':
                # Set environment variables for this session
                username = input("Enter your Kaggle username: ").strip()
                api_key = input("Enter your Kaggle API key: ").strip()
                os.environ["KAGGLE_USERNAME"] = username
                os.environ["KAGGLE_KEY"] = api_key
                download_data()
    elif choice == "3":
        download_data()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()


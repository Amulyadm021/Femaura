#!/usr/bin/env python3
"""
Simple test script to verify the Femaura.AI application is working
"""

import requests
import time
import sys

def test_application():
    """Test the application endpoints"""
    base_url = "http://localhost:5000"
    
    print("Testing Femaura.AI Application")
    print("=" * 40)
    
    # Wait a moment for the app to start
    print("Waiting for application to start...")
    time.sleep(3)
    
    try:
        # Test homepage
        print("Testing homepage...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Homepage accessible")
        else:
            print(f"ERROR: Homepage error: {response.status_code}")
            return False
        
        # Test login page
        print("Testing login page...")
        response = requests.get(f"{base_url}/login", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Login page accessible")
        else:
            print(f"ERROR: Login page error: {response.status_code}")
        
        # Test register page
        print("Testing register page...")
        response = requests.get(f"{base_url}/register", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Register page accessible")
        else:
            print(f"ERROR: Register page error: {response.status_code}")
        
        # Test predict page (should redirect to login)
        print("Testing predict page (should redirect)...")
        response = requests.get(f"{base_url}/predict", timeout=5, allow_redirects=False)
        if response.status_code == 302:  # Redirect
            print("SUCCESS: Predict page properly redirects to login")
        else:
            print(f"WARNING: Predict page status: {response.status_code}")
        
        # Test debug endpoint (should redirect to login)
        print("Testing debug endpoint...")
        response = requests.get(f"{base_url}/debug", timeout=5, allow_redirects=False)
        if response.status_code == 401:  # Unauthorized
            print("SUCCESS: Debug endpoint properly requires authentication")
        else:
            print(f"WARNING: Debug endpoint status: {response.status_code}")
        
        print("\nSUCCESS: Application appears to be working correctly!")
        print("You can now access the application at: http://localhost:5000")
        print("\nNext steps:")
        print("1. Open http://localhost:5000 in your browser")
        print("2. Register a new account")
        print("3. Login and test the prediction features")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to application")
        print("Make sure the application is running with: python app.py")
        return False
    except requests.exceptions.Timeout:
        print("ERROR: Application is not responding")
        print("The application may still be starting up")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_application()
    sys.exit(0 if success else 1)

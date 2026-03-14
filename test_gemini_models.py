"""
Test script to check which Gemini models are available
Run this to see which models work with your API key
"""

import google.generativeai as genai
import os

# Set your API key
API_KEY = ""
genai.configure(api_key=API_KEY)

print("="*60)
print("Testing Available Gemini Models")
print("="*60)

# List of model names to test
model_names = [
    'gemini-1.5-pro',
    'gemini-1.5-flash', 
    'gemini-pro',
    'gemini-1.0-pro',
    'gemini-1.5-pro-latest',
    'gemini-1.5-flash-latest'
]

available_models = []

for model_name in model_names:
    try:
        print(f"\nTesting: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        # Try a simple generation
        response = model.generate_content("Hello")
        
        if hasattr(response, 'text') and response.text:
            print(f"✅ {model_name} - WORKS!")
            available_models.append(model_name)
        else:
            print(f"⚠️  {model_name} - Initialized but no response")
            
    except Exception as e:
        print(f"❌ {model_name} - FAILED: {str(e)[:100]}")

print("\n" + "="*60)
print("Summary:")
print("="*60)

if available_models:
    print(f"\n✅ Available models: {', '.join(available_models)}")
    print(f"\nRecommendation: Use '{available_models[0]}' (first available)")
else:
    print("\n❌ No models are available. Please check:")
    print("  1. Your API key is correct")
    print("  2. You have internet connection")
    print("  3. Your API key has access to Gemini models")
    print("\nYou can also list all models using:")
    print("  for model in genai.list_models():")
    print("      print(model.name)")




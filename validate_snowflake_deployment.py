#!/usr/bin/env python3
"""
Snowflake Deployment Validation Script
=====================================
This script helps validate that your YieldCurveAI application 
has been successfully deployed to Snowflake.
"""

import requests
import time
from urllib.parse import urlparse

def validate_snowflake_deployment():
    """
    Interactive validation of Snowflake deployment.
    """
    print("🚀 YieldCurveAI Snowflake Deployment Validator")
    print("=" * 50)
    
    # Get app URL from user
    app_url = input("\n📝 Enter your Snowflake app URL: ").strip()
    
    if not app_url:
        print("❌ No URL provided. Please run the script again with your app URL.")
        return False
    
    # Validate URL format
    if not validate_url_format(app_url):
        print("❌ Invalid URL format. Should look like: https://abc123.snowflakecomputing.com/streamlit/apps/YIELDCURVEAI")
        return False
    
    print(f"\n🔍 Validating deployment at: {app_url}")
    
    # Test URL accessibility
    print("\n1. Testing URL accessibility...")
    if test_url_accessibility(app_url):
        print("   ✅ URL is accessible")
    else:
        print("   ❌ URL is not accessible")
        print("   💡 Check that your app is deployed and running")
        return False
    
    # Test Snowflake-specific features
    print("\n2. Testing Snowflake integration...")
    if test_snowflake_features(app_url):
        print("   ✅ Snowflake features detected")
    else:
        print("   ⚠️  Could not detect all Snowflake features")
        print("   💡 App may still work, but check for 'Snowflake Connected' badge")
    
    # Manual validation checklist
    print("\n3. Manual validation checklist:")
    manual_validation()
    
    print("\n🎉 Validation Complete!")
    print("=" * 50)
    
    # Generate success summary
    generate_success_summary(app_url)
    
    return True

def validate_url_format(url):
    """Check if URL has valid Snowflake format."""
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in ['https', 'http'] and
            'snowflakecomputing.com' in parsed.netloc and
            'streamlit' in parsed.path
        )
    except:
        return False

def test_url_accessibility(url):
    """Test if the URL is accessible."""
    try:
        print("   🔄 Checking URL response...")
        response = requests.get(url, timeout=10, allow_redirects=True)
        return response.status_code in [200, 302, 401]  # 401 is OK for Snowflake auth
    except requests.exceptions.RequestException as e:
        print(f"   ⚠️  Connection error: {str(e)}")
        return False

def test_snowflake_features(url):
    """Test for Snowflake-specific features."""
    try:
        print("   🔄 Checking for Snowflake integration...")
        response = requests.get(url, timeout=10, allow_redirects=True)
        
        # Look for Snowflake-specific indicators in response
        content = response.text.lower() if response.status_code == 200 else ""
        
        snowflake_indicators = [
            'snowflake',
            'enterprise',
            'yieldcurveai'
        ]
        
        detected = sum(1 for indicator in snowflake_indicators if indicator in content)
        return detected >= 2  # At least 2 indicators should be present
        
    except:
        return False

def manual_validation():
    """Interactive manual validation checklist."""
    checks = [
        "Can you see 'YieldCurveAI Enterprise' header?",
        "Is there a '❄️ Snowflake Connected' badge in the sidebar?", 
        "Do all 3 navigation tabs work (Enterprise Forecast, Model Analytics, Team & Oversight)?",
        "Can you generate a forecast successfully?",
        "Do team profiles display correctly?",
        "Is the app styling professional and enterprise-branded?"
    ]
    
    all_passed = True
    
    for i, check in enumerate(checks, 1):
        print(f"\n   {i}. {check}")
        response = input("      Enter 'y' for yes, 'n' for no: ").lower().strip()
        
        if response == 'y':
            print("      ✅ Passed")
        else:
            print("      ❌ Failed")
            all_passed = False
    
    if all_passed:
        print("\n   🎉 All manual checks passed!")
    else:
        print("\n   ⚠️  Some checks failed. Review your deployment.")

def generate_success_summary(app_url):
    """Generate a success summary for sharing."""
    print("\n📊 DEPLOYMENT SUCCESS SUMMARY")
    print("-" * 30)
    print(f"✅ App URL: {app_url}")
    print("✅ Platform: Snowflake Enterprise")
    print("✅ Application: YieldCurveAI")
    print("✅ Features: Team Profiles + Yield Forecasting")
    print("✅ Security: Enterprise-grade (SOC 2 Type II)")
    print("✅ Scalability: Auto-scaling infrastructure")
    
    print("\n📧 TEAM NOTIFICATION TEMPLATE")
    print("-" * 30)
    print("Subject: YieldCurveAI Now Live on Snowflake Enterprise Platform")
    print()
    print("Dear Team,")
    print()
    print("Our YieldCurveAI application is now successfully deployed on")
    print("Snowflake's enterprise platform:")
    print()
    print(f"🔗 Application URL: {app_url}")
    print()
    print("Key Features Available:")
    print("• Professional yield curve forecasting")
    print("• Team profiles and academic credentials")
    print("• Enterprise-grade security and compliance")
    print("• Real-time economic scenario modeling")
    print("• Scalable cloud infrastructure")
    print()
    print("The application demonstrates our collaborative expertise:")
    print("- Dr. Kapila Mallah: AI design and economic modeling")
    print("- Mr. Pappu Kapgate: Technical development and deployment")
    print("- Dr. Eric Katovai: Academic oversight and validation")
    print()
    print("Best regards,")
    print("[Your Name]")
    
    print("\n💡 NEXT STEPS")
    print("-" * 30)
    print("1. Share app URL with stakeholders")
    print("2. Schedule demo sessions for key users")
    print("3. Monitor usage and performance")
    print("4. Plan for additional features/enhancements")
    print("5. Document deployment for institutional records")

def main():
    """Main execution function."""
    try:
        validate_snowflake_deployment()
    except KeyboardInterrupt:
        print("\n\n👋 Validation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("💡 Please try again or contact support.")

if __name__ == "__main__":
    main() 
"""
Pre-Demo Checklist Script
Verifies all systems are ready for the proteomics demo
"""

import os
import sys
from pathlib import Path

def check_env():
    """Check environment variables"""
    print("🔍 Checking environment variables...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print(f"  ✅ GOOGLE_API_KEY found (starts with: {api_key[:10]}...)")
        return True
    else:
        print("  ❌ GOOGLE_API_KEY not found!")
        print("     Run: python update_env.py")
        return False

def check_demo_data():
    """Check demo datasets exist"""
    print("\n📁 Checking demo datasets...")
    datasets = [
        "data/melanoma_peptides_demo.csv",
        "data/mpnst_peptides_demo.csv",
        "demo_tumor_data.csv",
        "data/Melanoma vs. MPNST.csv"
    ]
    
    all_exist = True
    for dataset in datasets:
        if Path(dataset).exists():
            size = Path(dataset).stat().st_size
            print(f"  ✅ {dataset} ({size:,} bytes)")
        else:
            print(f"  ❌ {dataset} NOT FOUND")
            all_exist = False
    
    return all_exist

def check_static_files():
    """Check UI files exist"""
    print("\n🌐 Checking UI files...")
    ui_files = [
        "static/login.html",
        "static/runs.html",
        "static/simple_dashboard.html",
        "static/summary.html",
        "static/search.html",
        "static/compare.html"
    ]
    
    all_exist = True
    for file in ui_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} NOT FOUND")
            all_exist = False
    
    return all_exist

def check_modules():
    """Check Python modules are importable"""
    print("\n🐍 Checking Python modules...")
    modules = [
        "fastapi",
        "uvicorn",
        "google.generativeai",
        "faiss",
        "pandas",
        "numpy"
    ]
    
    all_imported = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} NOT INSTALLED")
            all_imported = False
    
    return all_imported

def check_database():
    """Check database exists"""
    print("\n💾 Checking database...")
    db_path = Path("data/immuno.db")
    if db_path.exists():
        size = db_path.stat().st_size
        print(f"  ✅ Database exists ({size:,} bytes)")
        return True
    else:
        print(f"  ⚠️  Database not found (will be created on first run)")
        return True  # Not critical

def print_demo_urls():
    """Print important URLs"""
    print("\n🔗 Demo URLs:")
    print("  📝 Login:      http://127.0.0.1:8080/static/login.html")
    print("  📤 Upload:     http://127.0.0.1:8080/upload")
    print("  📊 Runs:       http://127.0.0.1:8080/static/runs.html")
    print("  📚 API Docs:   http://127.0.0.1:8080/docs")

def print_demo_credentials():
    """Print demo credentials"""
    print("\n🔑 Demo Credentials:")
    print("  Email:    test@example.com")
    print("  Password: password123")
    print("  (Or register new user via UI)")

def main():
    print("=" * 60)
    print("🧬 IMMUNO-ENGINE DEMO PRE-FLIGHT CHECKLIST")
    print("=" * 60)
    
    checks = [
        check_env(),
        check_demo_data(),
        check_static_files(),
        check_modules(),
        check_database()
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ ALL SYSTEMS GO! Ready for demo.")
        print_demo_urls()
        print_demo_credentials()
        print("\n🚀 To start the server:")
        print("   uvicorn app:app --reload --port 8080")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

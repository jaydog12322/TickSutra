# test_data_logger.py
"""
Test script for the real-time data logger.
Validates setup and performs basic functionality tests.
"""

import sys
import os
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dependencies():
    """Test if all required packages are installed"""
    print("=== Testing Dependencies ===")

    required_packages = {
        'PyQt5': 'PyQt5',
        'pandas': 'pandas',
        'fastparquet': 'fastparquet',
        'openpyxl': 'openpyxl'
    }

    missing_packages = []

    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} - OK")
        except ImportError:
            print(f"✗ {package_name} - MISSING")
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\nERROR: Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False

    print("All dependencies satisfied ✓")
    return True


def test_directory_structure():
    """Test and create directory structure"""
    print("\n=== Testing Directory Structure ===")

    required_dirs = ['config', 'data', 'logs']

    for dir_name in required_dirs:
        dir_path = Path(f"./{dir_name}")
        if dir_path.exists():
            print(f"✓ {dir_name}/ - EXISTS")
        else:
            dir_path.mkdir(exist_ok=True)
            print(f"✓ {dir_name}/ - CREATED")

    return True


def test_symbol_file():
    """Test symbol universe file"""
    print("\n=== Testing Symbol File ===")

    symbol_file = Path("./config/symbol_universe.xlsx")

    if symbol_file.exists():
        try:
            df = pd.read_excel(symbol_file)
            symbol_count = len(df)
            print(f"✓ Symbol file exists with {symbol_count} symbols")

            # Show sample symbols
            if 'Symbol' in df.columns:
                sample_symbols = df['Symbol'].head(5).tolist()
            else:
                sample_symbols = df.iloc[:, 0].head(5).tolist()

            print(f"  Sample symbols: {sample_symbols}")
            return True

        except Exception as e:
            print(f"✗ Error reading symbol file: {e}")
            return False
    else:
        print("✗ Symbol file not found")
        print("  Creating sample file...")
        create_test_symbol_file()
        return True


def create_test_symbol_file():
    """Create a test symbol file"""
    test_symbols = [
        "005930",  # Samsung Electronics
        "000660",  # SK Hynix
        "035420",  # Naver
        "051910",  # LG Chem
        "006400",  # Samsung SDI
        "035720",  # Kakao
        "207940",  # Samsung Biologics
        "068270",  # Celltrion
        "323410",  # Kakao Bank
        "377300",  # Kakao Pay
    ]

    df = pd.DataFrame({'Symbol': test_symbols})

    output_path = Path("./config/symbol_universe.xlsx")
    df.to_excel(output_path, index=False)

    print(f"✓ Created test symbol file with {len(test_symbols)} symbols")


def test_kiwoom_import():
    """Test if Kiwoom OCX can be imported"""
    print("\n=== Testing Kiwoom API ===")

    try:
        from PyQt5.QAxContainer import QAxWidget
        from PyQt5.QtWidgets import QApplication

        # Try to create a QApplication (required for OCX)
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        # Try to create Kiwoom OCX control
        ocx = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")
        print("✓ Kiwoom OpenAPI+ OCX control accessible")

        # Test some basic methods
        if hasattr(ocx, 'CommConnect'):
            print("✓ CommConnect method available")
        if hasattr(ocx, 'SetRealReg'):
            print("✓ SetRealReg method available")
        if hasattr(ocx, 'GetCommRealData'):
            print("✓ GetCommRealData method available")

        return True

    except Exception as e:
        print(f"✗ Kiwoom API not available: {e}")
        print("  Make sure Kiwoom OpenAPI+ is installed")
        print("  This is normal if running on non-Windows or without Kiwoom")
        return False


def test_parquet_operations():
    """Test Parquet file operations"""
    print("\n=== Testing Parquet Operations ===")

    try:
        # Create sample data
        sample_data = [
            {
                "timestamp": "2025-09-06T09:00:30.123456",
                "symbol": "005930",
                "venue": "KRX",
                "real_type": "주식호가잔량",
                "fid_27": "84510",
                "fid_28": "84490",
                "fid_41": "100",
                "fid_51": "120",
                "raw_code": "005930"
            },
            {
                "timestamp": "2025-09-06T09:00:30.123456",
                "symbol": "005930",
                "venue": "NXT",
                "real_type": "주식호가잔량",
                "fid_27": "84560",
                "fid_28": "84540",
                "fid_41": "80",
                "fid_51": "90",
                "raw_code": "005930_NX"
            }
        ]

        # Create DataFrame
        df = pd.DataFrame(sample_data)

        # Write to Parquet
        test_file = Path("./data/test_output.parquet")
        df.to_parquet(test_file, index=False, engine="fastparquet")
        print("✓ Parquet write successful")

        # Read back from Parquet
        df_read = pd.read_parquet(test_file, engine="fastparquet")
        print(f"✓ Parquet read successful ({len(df_read)} records)")

        # Verify data integrity
        if len(df_read) == len(sample_data):
            print("✓ Data integrity verified")

        # Clean up
        test_file.unlink()
        print("✓ Test file cleaned up")

        return True

    except Exception as e:
        print(f"✗ Parquet operations failed: {e}")
        print("  Install with: pip install fastparquet")
        return False


def test_logger_import():
    """Test if main logger can be imported"""
    print("\n=== Testing Logger Import ===")

    try:
        # Add current directory to path
        sys.path.insert(0, '.')

        # Try to import main components
        from data_logger_main import KiwoomDataLogger, DataBuffer, SymbolLoader
        print("✓ Main logger components imported successfully")

        # Test component initialization (without Kiwoom connection)
        buffer = DataBuffer(buffer_size=100, output_dir="./data")
        print("✓ DataBuffer initialization successful")

        # Test symbol loader
        test_symbols = SymbolLoader.load_symbols("./config/symbol_universe.xlsx")
        print(f"✓ SymbolLoader successful ({len(test_symbols)} symbols loaded)")

        return True

    except Exception as e:
        print(f"✗ Logger import failed: {e}")
        return False


def generate_test_report():
    """Generate a test report"""
    print("\n" + "=" * 50)
    print("REAL-TIME DATA LOGGER TEST REPORT")
    print("=" * 50)

    tests = [
        ("Dependencies", test_dependencies),
        ("Directory Structure", test_directory_structure),
        ("Symbol File", test_symbol_file),
        ("Kiwoom API", test_kiwoom_import),
        ("Parquet Operations", test_parquet_operations),
        ("Logger Import", test_logger_import)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} - EXCEPTION: {e}")

    print(f"\n" + "=" * 50)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Ready to run data logger!")
    elif passed >= total - 1:  # Allow Kiwoom test to fail
        print("⚠️  MOSTLY READY - May need Kiwoom API setup")
    else:
        print("❌ SETUP INCOMPLETE - Fix errors before running")

    print("\nNext steps:")
    print("1. Ensure Kiwoom OpenAPI+ is installed (Windows only)")
    print("2. Login to Kiwoom HTS once to verify account")
    print("3. Run: python data_logger_main.py")
    print("4. Or run: python run_logger.py")


def main():
    """Main test function"""
    print("Real-time Data Logger - Setup Validation")
    print("This script will test your setup and configuration\n")

    generate_test_report()


if __name__ == "__main__":
    main()


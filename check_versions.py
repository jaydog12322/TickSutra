# simple_check.py
"""
Simple Python version check and package installation guide
"""
import sys


def check_python_version():
    """Check Python version and print installation commands"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    print(f"Architecture: {'64-bit' if sys.maxsize > 2 ** 32 else '32-bit'}")

    print(f"\n=== RECOMMENDED INSTALLATION COMMANDS FOR YOUR PYTHON {version.major}.{version.minor} ===")

    # Always update pip first
    print("1. UPDATE PIP:")
    print("python -m pip install --upgrade pip")
    print()

    # Version-specific recommendations
    if version.major == 3:
        if version.minor >= 11:
            print("2. FOR PYTHON 3.11+:")
            print("python -m pip install numpy==1.24.3")
            print("python -m pip install pandas==2.0.3")
            print("python -m pip install PyQt5==5.15.10")
            print("python -m pip install openpyxl==3.1.2")
        elif version.minor >= 9:
            print("2. FOR PYTHON 3.9-3.10:")
            print("python -m pip install numpy==1.23.5")
            print("python -m pip install pandas==1.5.3")
            print("python -m pip install PyQt5==5.15.9")
            print("python -m pip install openpyxl==3.1.2")
        elif version.minor >= 7:
            print("2. FOR PYTHON 3.7-3.8:")
            print("python -m pip install numpy==1.21.6")
            print("python -m pip install pandas==1.3.5")
            print("python -m pip install PyQt5==5.15.7")
            print("python -m pip install openpyxl==3.0.10")
        else:
            print("2. FOR OLDER PYTHON 3.x:")
            print("python -m pip install numpy==1.19.5")
            print("python -m pip install pandas==1.1.5")
            print("python -m pip install PyQt5==5.12.3")
            print("python -m pip install openpyxl==3.0.7")

    print()
    print("3. IF PANDAS STILL FAILS, TRY:")
    print("python -m pip install pandas --no-deps")
    print("python -m pip install numpy pytz python-dateutil")
    print()
    print("4. ALTERNATIVE - SKIP PYARROW AND USE CSV VERSION:")
    print("   Just install the packages above (without pyarrow)")
    print("   Then use data_logger_csv_fallback.py instead")


def test_imports():
    """Test if packages can be imported"""
    print(f"\n=== TESTING CURRENT PACKAGES ===")

    packages = ['numpy', 'pandas', 'PyQt5', 'openpyxl']

    for package in packages:
        try:
            if package == 'PyQt5':
                from PyQt5.QtWidgets import QApplication
                print(f"✓ {package} - OK")
            else:
                exec(f"import {package}")
                print(f"✓ {package} - OK")
        except ImportError as e:
            print(f"✗ {package} - MISSING ({str(e)[:50]}...)")
        except Exception as e:
            print(f"⚠ {package} - ERROR ({str(e)[:50]}...)")


if __name__ == "__main__":
    print("=== PYTHON ENVIRONMENT CHECK ===")
    check_python_version()
    test_imports()

    print(f"\n=== WHAT TO DO NEXT ===")
    print("1. Copy and run the installation commands above")
    print("2. If pandas fails, use the CSV fallback version")
    print("3. Test by running: python simple_test.py")

# simple_test.py
"""
Simple test to verify packages work
"""


def test_basic_imports():
    """Test basic package imports"""
    print("Testing basic imports...")

    try:
        import sys
        print(f"✓ Python {sys.version}")
    except:
        print("✗ Python import failed")
        return False

    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except ImportError:
        print("✗ pandas not available")
        return False
    except Exception as e:
        print(f"✗ pandas error: {e}")
        return False

    try:
        from PyQt5.QtWidgets import QApplication
        print("✓ PyQt5 available")
    except ImportError:
        print("✗ PyQt5 not available")
        return False
    except Exception as e:
        print(f"✗ PyQt5 error: {e}")
        return False

    try:
        import openpyxl
        print(f"✓ openpyxl {openpyxl.__version__}")
    except ImportError:
        print("✗ openpyxl not available")
        return False
    except Exception as e:
        print(f"✗ openpyxl error: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")

    try:
        import pandas as pd

        # Test DataFrame creation
        df = pd.DataFrame({'test': [1, 2, 3]})
        print("✓ pandas DataFrame creation works")

        # Test CSV writing
        df.to_csv('test.csv', index=False)
        print("✓ CSV writing works")

        # Test CSV reading
        df2 = pd.read_csv('test.csv')
        print("✓ CSV reading works")

        # Clean up
        import os
        os.remove('test.csv')
        print("✓ File operations work")

        return True

    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== SIMPLE PACKAGE TEST ===")

    if test_basic_imports():
        if test_basic_functionality():
            print("\n🎉 ALL TESTS PASSED!")
            print("You can now run the data logger!")
            print("Use: python data_logger_csv_fallback.py")
        else:
            print("\n⚠️ Imports OK but functionality issues")
    else:
        print("\n❌ Package imports failed")
        print("Run installation commands first")
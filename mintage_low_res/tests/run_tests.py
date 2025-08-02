"""Simple test runner for mintage_low_res package."""

import sys
import pytest
from pathlib import Path

def run_tests():
    """Run all tests for the mintage_low_res package."""
    
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    print("Running mintage_low_res test suite...")
    print(f"Test directory: {tests_dir}")
    
    # Run pytest with verbose output
    args = [
        str(tests_dir),
        "-v",                    # Verbose output
        "--tb=short",           # Short traceback format
        "-x",                   # Stop on first failure
        "--disable-warnings",   # Disable warnings for cleaner output
    ]
    
    # Add integration test marker if requested
    if len(sys.argv) > 1 and "integration" in sys.argv[1]:
        args.extend(["-m", "integration"])
        print("Running integration tests...")
    else:
        args.extend(["-m", "not integration"])
        print("Running unit tests (skipping integration tests)...")
    
    # Run the tests
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\\n✅ All tests passed!")
    else:
        print("\\n❌ Some tests failed.")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())
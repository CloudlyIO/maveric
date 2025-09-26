# test_export.py
# Test script for the Energy Savings export function
# Demonstrates usage and validates functionality

import pandas as pd
import os
import sys
import json

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_func.export_metrics import export_metrics, validate_metrics_format


def create_sample_es_metrics() -> pd.DataFrame:
    """
    Creates a sample DataFrame that mimics the output from rl_predictor.py
    """
    data = {
        'cell_id': [f'cell_{i:03d}' for i in range(1, 11)],
        'predicted_state': ['ON', 'OFF', 'ON', 'ON', 'OFF', 'ON', 'ON', 'OFF', 'ON', 'ON'],
        'predicted_cell_el_deg': [8.0, 'N/A', 12.0, 6.0, 'N/A', 14.0, 10.0, 'N/A', 4.0, 16.0]
    }
    return pd.DataFrame(data)


def test_export_function():
    """Test the export_metrics function with various formats"""
    print("=== Testing Energy Savings Export Function ===\n")
    
    # Create sample metrics data
    sample_metrics = create_sample_es_metrics()
    print("Sample ES metrics data:")
    print(sample_metrics.to_string(index=False))
    print()
    
    # Validate metrics format
    validation_result = validate_metrics_format(sample_metrics)
    print(f"Metrics validation: {'PASSED' if validation_result['valid'] else 'FAILED'}")
    if validation_result['warnings']:
        print(f"Warnings: {validation_result['warnings']}")
    print()
    
    # Test 1: CSV export
    print("1. Testing CSV export...")
    result_csv = export_metrics(sample_metrics, "./test_es_metrics_csv", format="csv")
    print(f"   Success: {result_csv['success']}")
    print(f"   Files: {result_csv['files_created']}")
    if result_csv['errors']:
        print(f"   Errors: {result_csv['errors']}")
    print()
    
    # Test 2: JSON export with metadata
    print("2. Testing JSON export with metadata...")
    result_json = export_metrics(sample_metrics, "./test_es_metrics_json", format="json", include_metadata=True)
    print(f"   Success: {result_json['success']}")
    print(f"   Files: {result_json['files_created']}")
    if result_json['errors']:
        print(f"   Errors: {result_json['errors']}")
    
    # Verify JSON content structure
    if result_json['success'] and result_json['files_created']:
        with open(result_json['files_created'][0], 'r') as f:
            json_content = json.load(f)
            print(f"   JSON has metadata: {'metadata' in json_content}")
            print(f"   JSON has data: {'data' in json_content}")
            print(f"   Records in JSON: {len(json_content.get('data', []))}")
    print()
    
    # Test 3: Excel export
    print("3. Testing Excel export...")
    result_excel = export_metrics(sample_metrics, "./test_es_metrics_excel", format="excel")
    print(f"   Success: {result_excel['success']}")
    print(f"   Files: {result_excel['files_created']}")
    if result_excel['errors']:
        print(f"   Errors: {result_excel['errors']}")
    print()
    
    # Test 4: All formats export
    print("4. Testing all formats export...")
    result_all = export_metrics(sample_metrics, "./test_es_metrics_all", format="all")
    print(f"   Success: {result_all['success']}")
    print(f"   Files: {result_all['files_created']}")
    if result_all['errors']:
        print(f"   Errors: {result_all['errors']}")
    print()
    
    # Test 5: Error handling - invalid format
    print("5. Testing error handling - invalid format...")
    result_error = export_metrics(sample_metrics, "./test_es_metrics_error", format="invalid")
    print(f"   Success: {result_error['success']}")
    print(f"   Errors: {result_error['errors']}")
    print()
    
    print("=== Export Function Test Completed ===\n")
    
    return {
        'csv': result_csv['success'],
        'json': result_json['success'],
        'excel': result_excel['success'],
        'all': result_all['success'],
        'error_handling': not result_error['success']  # Should fail for invalid format
    }


def demonstrate_integration_with_rl_predictor():
    """
    Demonstrates how the export function would integrate with rl_predictor.py
    """
    print("=== Integration with rl_predictor.py Demo ===\n")
    
    # Simulate the output from run_rl_prediction function
    print("Simulating rl_predictor.py output...")
    sample_metrics = create_sample_es_metrics()
    
    print("Step 1: RL prediction generates metrics DataFrame")
    print(f"DataFrame shape: {sample_metrics.shape}")
    print(f"Columns: {list(sample_metrics.columns)}")
    print()
    
    print("Step 2: Export the metrics to multiple formats")
    export_result = export_metrics(sample_metrics, "./demo_es_integration", format="all")
    
    if export_result['success']:
        print("✅ Integration successful!")
        print(f"Exported files: {export_result['files_created']}")
    else:
        print("❌ Integration failed!")
        print(f"Errors: {export_result['errors']}")
    
    print("\n=== Integration Demo Completed ===\n")


if __name__ == "__main__":
    # Run tests
    test_results = test_export_function()
    
    # Run integration demo
    demonstrate_integration_with_rl_predictor()
    
    # Summary
    print("=== Test Summary ===")
    all_tests_passed = all(test_results.values())
    print(f"All tests passed: {'✅ YES' if all_tests_passed else '❌ NO'}")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")

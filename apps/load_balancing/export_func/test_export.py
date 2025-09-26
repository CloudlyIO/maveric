# test_export.py
# Test script for the Load Balancing export function
# Demonstrates usage and validates functionality

import pandas as pd
import os
import sys
import json

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_func.export_metrics import export_metrics, validate_metrics_format


def create_sample_lb_metrics() -> pd.DataFrame:
    """
    Creates a sample DataFrame that mimics the output from load_balancing rl_predictor.py
    Based on the map_action_to_config_df function structure
    """
    data = {
        'cell_id': [f'cell_{i:03d}' for i in range(1, 16)],
        'predicted_cell_el_deg': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 5.0, 7.0, 9.0, 11.0]
    }
    return pd.DataFrame(data)


def test_export_function():
    """Test the export_metrics function with various formats"""
    print("=== Testing Load Balancing Export Function ===\n")
    
    # Create sample metrics data
    sample_metrics = create_sample_lb_metrics()
    print("Sample Load Balancing metrics data:")
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
    result_csv = export_metrics(sample_metrics, "./test_lb_metrics_csv", format="csv")
    print(f"   Success: {result_csv['success']}")
    print(f"   Files: {result_csv['files_created']}")
    if result_csv['errors']:
        print(f"   Errors: {result_csv['errors']}")
    print()
    
    # Test 2: JSON export with metadata
    print("2. Testing JSON export with metadata...")
    result_json = export_metrics(sample_metrics, "./test_lb_metrics_json", format="json", include_metadata=True)
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
            if 'metadata' in json_content:
                print(f"   Module: {json_content['metadata'].get('module', 'N/A')}")
                print(f"   Optimization type: {json_content['metadata'].get('optimization_type', 'N/A')}")
    print()
    
    # Test 3: Excel export
    print("3. Testing Excel export...")
    result_excel = export_metrics(sample_metrics, "./test_lb_metrics_excel", format="excel")
    print(f"   Success: {result_excel['success']}")
    print(f"   Files: {result_excel['files_created']}")
    if result_excel['errors']:
        print(f"   Errors: {result_excel['errors']}")
    print()
    
    # Test 4: All formats export
    print("4. Testing all formats export...")
    result_all = export_metrics(sample_metrics, "./test_lb_metrics_all", format="all")
    print(f"   Success: {result_all['success']}")
    print(f"   Files: {result_all['files_created']}")
    if result_all['errors']:
        print(f"   Errors: {result_all['errors']}")
    print()
    
    # Test 5: Error handling - invalid format
    print("5. Testing error handling - invalid format...")
    result_error = export_metrics(sample_metrics, "./test_lb_metrics_error", format="invalid")
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
    Demonstrates how the export function would integrate with load_balancing rl_predictor.py
    """
    print("=== Integration with load_balancing rl_predictor.py Demo ===\n")
    
    # Simulate the output from run_rl_prediction function
    print("Simulating load_balancing rl_predictor.py output...")
    sample_metrics = create_sample_lb_metrics()
    
    print("Step 1: RL prediction generates metrics DataFrame")
    print(f"DataFrame shape: {sample_metrics.shape}")
    print(f"Columns: {list(sample_metrics.columns)}")
    print(f"Tilt range: {sample_metrics['predicted_cell_el_deg'].min():.1f} - {sample_metrics['predicted_cell_el_deg'].max():.1f} degrees")
    print()
    
    print("Step 2: Export the metrics to multiple formats")
    export_result = export_metrics(sample_metrics, "./demo_lb_integration", format="all")
    
    if export_result['success']:
        print("✅ Integration successful!")
        print(f"Exported files: {export_result['files_created']}")
    else:
        print("❌ Integration failed!")
        print(f"Errors: {export_result['errors']}")
    
    print("\n=== Integration Demo Completed ===\n")


def test_validation_scenarios():
    """Test various validation scenarios"""
    print("=== Validation Scenarios Test ===\n")
    
    # Test 1: Valid data
    print("1. Testing valid data...")
    valid_df = create_sample_lb_metrics()
    validation = validate_metrics_format(valid_df)
    print(f"   Valid data: {'✅ PASS' if validation['valid'] else '❌ FAIL'}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    print()
    
    # Test 2: Out of range tilt values
    print("2. Testing out of range tilt values...")
    out_of_range_df = pd.DataFrame({
        'cell_id': ['cell_001', 'cell_002'],
        'predicted_cell_el_deg': [25.0, -5.0]  # Outside 0-20 range
    })
    validation = validate_metrics_format(out_of_range_df)
    print(f"   Out of range detection: {'✅ PASS' if validation['warnings'] else '❌ FAIL'}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    print()
    
    # Test 3: Missing columns
    print("3. Testing missing columns...")
    missing_cols_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    validation = validate_metrics_format(missing_cols_df)
    print(f"   Missing columns detection: {'✅ PASS' if not validation['valid'] else '❌ FAIL'}")
    if validation['errors']:
        print(f"   Errors: {validation['errors']}")
    print()
    
    # Test 4: Duplicate cell IDs
    print("4. Testing duplicate cell IDs...")
    duplicate_df = pd.DataFrame({
        'cell_id': ['cell_001', 'cell_001', 'cell_002'],
        'predicted_cell_el_deg': [5.0, 10.0, 15.0]
    })
    validation = validate_metrics_format(duplicate_df)
    print(f"   Duplicate detection: {'✅ PASS' if validation['warnings'] else '❌ FAIL'}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    print()


if __name__ == "__main__":
    # Run tests
    test_results = test_export_function()
    
    # Run integration demo
    demonstrate_integration_with_rl_predictor()
    
    # Run validation tests
    test_validation_scenarios()
    
    # Summary
    print("=== Test Summary ===")
    all_tests_passed = all(test_results.values())
    print(f"All tests passed: {'✅ YES' if all_tests_passed else '❌ NO'}")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")

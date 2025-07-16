#!/usr/bin/env python3
"""
🚨 EMERGENCY TEST HEALTH MONITOR
Diagnoses the complete breakdown of test infrastructure and provides emergency fixes
"""

import ast
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add src to path for imports
sys.path.insert(0, '../../src')

class TestHealthMonitor:
    def __init__(self):
        self.broken_tests = []
        self.missing_functions = {}
        self.import_errors = {}
        self.test_files = []
        
    def scan_test_files(self) -> List[str]:
        """Find all test files"""
        test_files = []
        test_dirs = ['tests/unit', 'tests/integration', 'tests']
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for root, dirs, files in os.walk(test_dir):
                    for file in files:
                        if file.startswith('test_') and file.endswith('.py'):
                            test_files.append(os.path.join(root, file))
        return test_files
    
    def extract_server_imports(self, filepath: str) -> List[str]:
        """Extract what functions each test file expects from server.py"""
        try:
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read())
            
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == 'music21_mcp.server':
                        if node.names:
                            for alias in node.names:
                                imports.append(alias.name)
            return imports
        except Exception as e:
            print(f"❌ Error parsing {filepath}: {e}")
            return []
    
    def check_server_exports(self) -> Set[str]:
        """Check what functions/classes are actually exported by server.py"""
        try:
            with open('src/music21_mcp/server.py', 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            exports = set()
            
            # Find function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    exports.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    exports.add(node.name)
                elif isinstance(node, ast.Assign):
                    # Variable assignments
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            exports.add(target.id)
            
            return exports
        except Exception as e:
            print(f"❌ Error reading server.py: {e}")
            return set()
    
    def test_import_health(self, filepath: str) -> Dict:
        """Test if a single test file can import successfully"""
        result = {
            'file': filepath,
            'can_import': False,
            'error': None,
            'expected_imports': [],
            'import_type': 'unknown'
        }
        
        try:
            # Extract expected imports
            result['expected_imports'] = self.extract_server_imports(filepath)
            
            # Try to import the module
            spec = self._import_test_module(filepath)
            result['can_import'] = True
            result['import_type'] = 'success'
            
        except ImportError as e:
            result['error'] = f"ImportError: {str(e)}"
            result['import_type'] = 'import_error'
        except SyntaxError as e:
            result['error'] = f"SyntaxError: {str(e)}"
            result['import_type'] = 'syntax_error'
        except Exception as e:
            result['error'] = f"Other error: {str(e)}"
            result['import_type'] = 'other_error'
        
        return result
    
    def _import_test_module(self, filepath: str):
        """Attempt to import a test module"""
        # Convert file path to module name
        module_path = filepath.replace('/', '.').replace('.py', '')
        if module_path.startswith('tests.'):
            module_path = module_path[6:]  # Remove 'tests.'
        
        # Try to execute the file to check for import errors
        with open(filepath, 'r') as f:
            code = f.read()
        
        # Create a temporary namespace
        namespace = {'__file__': filepath}
        exec(code, namespace)
        
        return True
    
    def analyze_missing_functions(self, expected_imports: List[str], actual_exports: Set[str]) -> List[str]:
        """Find which expected functions are missing from server.py"""
        missing = []
        for imp in expected_imports:
            if imp not in actual_exports:
                missing.append(imp)
        return missing
    
    def generate_emergency_fixes(self) -> Dict:
        """Generate emergency fixes for broken imports"""
        fixes = {
            'add_to_server': [],
            'update_imports': [],
            'create_stubs': []
        }
        
        # Analyze what's missing and suggest fixes
        for test_file, missing_funcs in self.missing_functions.items():
            for func in missing_funcs:
                # These are the common missing functions we need to restore
                if func in ['analyze_chords', 'analyze_key', 'import_score', 'delete_score']:
                    fixes['add_to_server'].append({
                        'function': func,
                        'test_file': test_file,
                        'fix_type': 'tool_wrapper'
                    })
                elif func.startswith('analyze_'):
                    fixes['add_to_server'].append({
                        'function': func,
                        'test_file': test_file,
                        'fix_type': 'analysis_wrapper'
                    })
        
        return fixes
    
    def run_complete_diagnosis(self):
        """Run complete test health diagnosis"""
        print("🚨 EMERGENCY TEST INFRASTRUCTURE DIAGNOSIS")
        print("=" * 60)
        
        # Scan all test files
        self.test_files = self.scan_test_files()
        print(f"📁 Found {len(self.test_files)} test files")
        
        # Check what server.py actually exports
        actual_exports = self.check_server_exports()
        print(f"📦 Server.py exports {len(actual_exports)} functions/classes: {sorted(actual_exports)}")
        print()
        
        # Test each file
        broken_count = 0
        working_count = 0
        
        for test_file in self.test_files:
            result = self.test_import_health(test_file)
            
            if result['can_import']:
                print(f"✅ {test_file}")
                working_count += 1
            else:
                print(f"❌ {test_file}")
                print(f"   Error: {result['error']}")
                print(f"   Expected imports: {result['expected_imports']}")
                
                # Track missing functions
                missing = self.analyze_missing_functions(result['expected_imports'], actual_exports)
                if missing:
                    self.missing_functions[test_file] = missing
                    print(f"   Missing from server.py: {missing}")
                
                self.broken_tests.append(result)
                broken_count += 1
                print()
        
        print(f"\n📊 HEALTH SUMMARY:")
        print(f"   ✅ Working tests: {working_count}")
        print(f"   ❌ Broken tests: {broken_count}")
        print(f"   🩺 Health score: {working_count/(working_count+broken_count)*100:.1f}%")
        
        # Generate fixes
        if broken_count > 0:
            print(f"\n🔧 EMERGENCY FIXES NEEDED:")
            fixes = self.generate_emergency_fixes()
            
            print(f"\n1. ADD TO SERVER.PY ({len(fixes['add_to_server'])} functions):")
            for fix in fixes['add_to_server']:
                print(f"   - {fix['function']} ({fix['fix_type']})")
            
            print(f"\n2. CRITICAL MISSING FUNCTIONS:")
            all_missing = set()
            for missing_list in self.missing_functions.values():
                all_missing.update(missing_list)
            for func in sorted(all_missing):
                print(f"   - {func}")
        
        return {
            'total_tests': len(self.test_files),
            'working': working_count,
            'broken': broken_count,
            'health_score': working_count/(working_count+broken_count)*100 if (working_count+broken_count) > 0 else 0,
            'missing_functions': self.missing_functions,
            'fixes': fixes if broken_count > 0 else {}
        }

def main():
    monitor = TestHealthMonitor()
    diagnosis = monitor.run_complete_diagnosis()
    
    print(f"\n🚨 EMERGENCY STATUS:")
    if diagnosis['health_score'] < 50:
        print(f"🔴 CRITICAL: {diagnosis['health_score']:.1f}% health - IMMEDIATE ACTION REQUIRED")
    elif diagnosis['health_score'] < 80:
        print(f"🟡 WARNING: {diagnosis['health_score']:.1f}% health - Action needed")
    else:
        print(f"🟢 GOOD: {diagnosis['health_score']:.1f}% health")
    
    return diagnosis

if __name__ == "__main__":
    main()
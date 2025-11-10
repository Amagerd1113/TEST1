#!/usr/bin/env python3
"""
Comprehensive code analysis script to identify missing classes,
imports, and other issues in VLA-GR codebase.
"""

import os
import re
import ast
from pathlib import Path
from collections import defaultdict

def find_python_files(root_dir):
    """Find all Python files in the project."""
    return list(Path(root_dir).rglob("*.py"))

def extract_class_definitions(file_path):
    """Extract all class definitions from a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def extract_class_usages(file_path):
    """Extract all class instantiations from a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Pattern for class instantiation
        # e.g., self.module = ClassName(...)
        pattern = r'\b([A-Z][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, content)
        return set(matches)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()

def find_undefined_classes(root_dir="src"):
    """Find all classes that are used but not defined."""

    # First pass: collect all defined classes
    defined_classes = set()
    file_to_classes = {}

    python_files = find_python_files(root_dir)

    for file_path in python_files:
        classes = extract_class_definitions(file_path)
        file_to_classes[file_path] = classes
        defined_classes.update(classes)

    # Second pass: find used classes
    undefined_classes = defaultdict(list)

    for file_path in python_files:
        used_classes = extract_class_usages(file_path)
        for cls in used_classes:
            if cls not in defined_classes:
                # Filter out built-ins and common external classes
                if cls not in ['Path', 'List', 'Dict', 'Tuple', 'Optional',
                              'Any', 'Union', 'Callable', 'Type', 'Enum',
                              'Config', 'Tensor', 'Module', 'Linear', 'Conv2d',
                              'Sequential', 'ModuleList', 'Parameter', 'GELU',
                              'ReLU', 'Dropout', 'LayerNorm', 'BatchNorm2d',
                              'MultiheadAttention', 'LSTM', 'GRU', 'Embedding',
                              'Softmax', 'Sigmoid', 'Tanh', 'Softplus', 'Identity',
                              'MaxPool2d', 'AdaptiveAvgPool2d', 'ConvTranspose2d',
                              'Image', 'AutoModel', 'AutoTokenizer', 'PhiForCausalLM',
                              'Normal', 'Categorical']:
                    undefined_classes[str(file_path)].append(cls)

    return undefined_classes, defined_classes, file_to_classes

def check_imports(file_path):
    """Check if all imports are valid."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    except Exception as e:
        return []

def analyze_vla_gr_agent():
    """Specifically analyze vla_gr_agent.py for missing components."""
    file_path = "src/core/vla_gr_agent.py"

    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        content = f.read()

    # Classes that are instantiated in ConferenceVLAGRAgent.__init__
    required_classes = [
        'AdvancedPerceptionModule',
        'UncertaintyAwareAffordanceModule',
        'AdaptiveGRFieldManager',
        'DifferentiableGeodesicPlanner',
        'FieldInjectedTransformer',
        'SpacetimeMemoryModule',
        'HierarchicalActionDecoder',
        'EpistemicUncertaintyModule'
    ]

    missing = []
    for cls in required_classes:
        if f"class {cls}" not in content:
            missing.append(cls)

    return missing

def main():
    print("="*80)
    print("VLA-GR Code Analysis Report")
    print("="*80)
    print()

    # Find undefined classes
    print("1. Checking for undefined classes...")
    undefined, defined, file_classes = find_undefined_classes()

    if undefined:
        print(f"\n⚠️  Found {sum(len(v) for v in undefined.values())} undefined class references:\n")
        for file_path, classes in sorted(undefined.items()):
            if classes:
                print(f"  {file_path}:")
                for cls in sorted(set(classes)):
                    print(f"    - {cls}")
                print()
    else:
        print("✓ No undefined classes found")

    # Analyze vla_gr_agent specifically
    print("\n2. Analyzing vla_gr_agent.py...")
    missing_vla = analyze_vla_gr_agent()

    if missing_vla:
        print(f"\n⚠️  Missing {len(missing_vla)} required classes in vla_gr_agent.py:")
        for cls in missing_vla:
            print(f"    - {cls}")
    else:
        print("✓ All required classes defined in vla_gr_agent.py")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total Python files analyzed: {len(find_python_files('src'))}")
    print(f"Total classes defined: {len(defined)}")
    print(f"Total undefined class references: {sum(len(v) for v in undefined.values())}")
    print(f"Files with undefined classes: {len(undefined)}")

    if missing_vla:
        print(f"\n❌ CRITICAL: vla_gr_agent.py is missing {len(missing_vla)} required classes!")

    if undefined:
        print(f"\n❌ ISSUES FOUND: {len(undefined)} files have undefined class references")
    else:
        print("\n✅ No major issues found!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Advanced code analysis to check for common issues.
"""

import ast
import os
from pathlib import Path
from collections import defaultdict

def analyze_file(filepath):
    """Analyze a single Python file for issues."""
    issues = []

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))

        # Check for common issues
        for node in ast.walk(tree):
            # Check for bare except clauses
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(f"Line {node.lineno}: Bare except clause (should specify exception type)")

            # Check for mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(f"Line {node.lineno}: Mutable default argument in function '{node.name}'")

            # Check for variable shadowing of builtins
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                builtins_list = ['list', 'dict', 'set', 'str', 'int', 'float', 'input', 'open', 'file', 'type', 'id']
                if node.id in builtins_list:
                    issues.append(f"Line {node.lineno}: Variable '{node.id}' shadows builtin")

        return issues

    except SyntaxError as e:
        return [f"Syntax Error: {e}"]
    except Exception as e:
        return [f"Analysis Error: {e}"]

def main():
    print("=" * 80)
    print("Advanced Code Analysis")
    print("=" * 80)
    print()

    # Find all Python files
    python_files = list(Path('src').rglob('*.py'))
    python_files.extend(Path('.').glob('*.py'))

    all_issues = {}
    total_issues = 0

    for filepath in python_files:
        issues = analyze_file(filepath)
        if issues:
            all_issues[str(filepath)] = issues
            total_issues += len(issues)

    if all_issues:
        print(f"Found {total_issues} potential issues:\n")
        for filepath, issues in sorted(all_issues.items()):
            print(f"{filepath}:")
            for issue in issues:
                print(f"  - {issue}")
            print()
    else:
        print("✅ No common code issues found!")

    print("\n" + "=" * 80)
    print(f"Summary: Analyzed {len(python_files)} files, found {total_issues} potential issues")
    print("=" * 80)

if __name__ == "__main__":
    main()

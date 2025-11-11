# Contributing to VLA-GR

Thank you for your interest in contributing to the VLA-GR Navigation Framework! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and considerate in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes and commit them
6. Push to your fork and submit a pull request

## Development Setup

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU support)
- Git
- Conda or virtualenv

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/vla-gr-navigation.git
cd vla-gr-navigation

# Create conda environment
conda create -n vla_gr_dev python=3.8
conda activate vla_gr_dev

# Install PyTorch
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117

# Install Habitat-Sim
conda install habitat-sim -c conda-forge -c aihabitat

# Install dependencies (including dev dependencies)
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Examples**: Add tutorials or example code

### Development Workflow

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our coding standards

3. **Write tests** for your changes

4. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```

5. **Format your code**:
   ```bash
   black src/
   flake8 src/
   mypy src/
   ```

6. **Commit your changes** with clear messages:
   ```bash
   git commit -m "feat: add new feature X"
   # or
   git commit -m "fix: resolve issue with Y"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a pull request** on GitHub

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters maximum
- Use type hints for all function signatures
- Use docstrings for all public functions, classes, and modules
- Follow the Black code formatter's style

### Docstring Format

Use Google-style docstrings:

```python
def example_function(arg1: int, arg2: str) -> bool:
    """
    Brief description of the function.

    Longer description if needed, explaining the purpose,
    algorithm, or any important details.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When and why this is raised

    Example:
        >>> result = example_function(42, "test")
        >>> print(result)
        True
    """
    # Implementation
    return True
```

### Code Organization

- Keep functions focused and single-purpose
- Use meaningful variable and function names
- Avoid deep nesting (max 3 levels)
- Group related functionality into modules
- Keep modules under 500 lines when possible

## Testing Guidelines

### Test Structure

Tests should be organized under the `tests/` directory:

```
tests/
├── unit/
│   ├── test_perception.py
│   ├── test_affordance.py
│   └── test_gr_field.py
├── integration/
│   ├── test_agent.py
│   └── test_training.py
└── fixtures/
    └── conftest.py
```

### Writing Tests

```python
import pytest
import torch
from src.core.perception import AdvancedPerceptionModule

def test_perception_module_output_shape():
    """Test that perception module outputs correct shapes."""
    config = {...}  # Test config
    module = AdvancedPerceptionModule(config)

    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 1, 224, 224)

    output = module(rgb=rgb, depth=depth)

    assert output['visual_features'].shape == (2, 196, 768)
    assert output['visual_uncertainty'].shape == (2, 196, 1)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_perception.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run with verbose output
pytest tests/ -v
```

## Pull Request Process

### Before Submitting

1. Ensure your code passes all tests
2. Update documentation if needed
3. Add your changes to CHANGELOG.md
4. Rebase on the latest main branch
5. Ensure commit messages are clear and descriptive

### PR Title Format

Use conventional commit format:

- `feat: add new feature`
- `fix: resolve bug in module X`
- `docs: update API documentation`
- `test: add tests for feature Y`
- `refactor: improve code structure`
- `perf: optimize performance of Z`
- `chore: update dependencies`

### PR Description Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. At least one maintainer must approve
2. All CI checks must pass
3. No unresolved discussions
4. Code coverage should not decrease

## Reporting Bugs

### Bug Report Template

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - OS and version
   - Python version
   - PyTorch version
   - CUDA version (if applicable)
   - VLA-GR version
6. **Stack Trace**: Full error message and stack trace
7. **Additional Context**: Any other relevant information

### Example Bug Report

```markdown
**Description**
The GR field computation fails when using batch size > 8

**Steps to Reproduce**
1. Initialize agent with batch_size=16
2. Run forward pass
3. Observe CUDA out of memory error

**Expected Behavior**
Should handle larger batch sizes with automatic gradient accumulation

**Actual Behavior**
RuntimeError: CUDA out of memory

**Environment**
- OS: Ubuntu 22.04
- Python: 3.8.10
- PyTorch: 2.0.0
- CUDA: 11.7
- GPU: RTX 3090 (24GB)
```

## Feature Requests

We welcome feature requests! Please:

1. Check if the feature already exists or is planned
2. Provide a clear use case
3. Describe the proposed solution
4. Consider alternatives
5. Be open to discussion and iteration

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
Detailed description of how it could work

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information, mockups, or examples
```

## Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and community discussions
- **Discord**: Real-time chat (if available)

## License

By contributing to VLA-GR, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- Academic papers (for major algorithmic contributions)

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search closed issues
3. Ask in GitHub Discussions
4. Contact the maintainers

Thank you for contributing to VLA-GR!

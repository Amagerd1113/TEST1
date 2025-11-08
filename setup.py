"""
Setup script for VLA-GR Navigation Framework
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Development requirements
dev_requirements = [
    "pytest>=7.3.0",
    "pytest-cov>=4.0.0",
    "black>=23.3.0",
    "flake8>=6.0.0",
    "mypy>=1.2.0",
    "pre-commit>=3.3.0",
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=1.2.0",
]

setup(
    name="vla-gr-navigation",
    version="1.0.0",
    author="VLA-GR Team",
    author_email="contact@vla-gr.ai",
    description="Vision-Language-Action with General Relativity Navigation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/vla-gr-navigation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "vis": ["matplotlib>=3.7.0", "seaborn>=0.12.0", "plotly>=5.14.0"],
        "ros": ["rclpy>=3.0.0", "geometry_msgs", "sensor_msgs", "nav_msgs"],
    },
    entry_points={
        "console_scripts": [
            "vla-gr-train=training.train:main",
            "vla-gr-evaluate=evaluation.evaluate:main",
            "vla-gr-demo=demo:main",
            "vla-gr-export=deployment.export:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-org/vla-gr-navigation/issues",
        "Documentation": "https://vla-gr.readthedocs.io",
        "Source": "https://github.com/your-org/vla-gr-navigation",
    },
)

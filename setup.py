import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "1.0.0"

REPO_NAME = "Brevity Constraints Reverse Performance Hierarchies in Language Models"
AUTHOR_USER_NAME = "MD Azizul Hakim"
SRC_REPO = "Brevity_Constraints_Reverse"
AUTHOR_EMAIL = "useforprofessional@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A framework that Analyzes Language Model's Reverse performance Hierarchies under Brevity Constraints",
    long_description=long_description,
    long_description_content_type="text/markdown",  
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.2",
        "pandas",
        "matplotlib",
        "seaborn"
        
    ],
    python_requires=">=3.7",
    keywords="large language model, compression, reasoning, robustness, intelligence framework",
)
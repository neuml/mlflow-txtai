# pylint: disable = C0111
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    # Remove GitHub dark mode images
    DESCRIPTION = "".join([line for line in f if "gh-dark-mode-only" not in line])

setup(
    name="mlflow-txtai",
    version="0.1.0",
    author="NeuML",
    description="MLflow automatic tracing for txtai",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neuml/mlflow-txtai",
    project_urls={
        "Documentation": "https://github.com/neuml/mlflow-txtai",
        "Issue Tracker": "https://github.com/neuml/mlflow-txtai/issues",
        "Source Code": "https://github.com/neuml/mlflow-txtai",
    },
    license="Apache 2.0: http://www.apache.org/licenses/LICENSE-2.0",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    keywords="search embedding machine-learning nlp",
    python_requires=">=3.9",
    install_requires=["mlflow>=2.20.2", "txtai>=8.3.1"],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Utilities",
    ],
)

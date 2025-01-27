import os

from setuptools import setup

package = ["zinnia"]
version = "0.0.1"
description = (
    "A tool for synthesizing discrete data-generating processes from noisy I/O examples"
)
author = "CipherInsight Ltd."

if os.path.exists("README.md"):
    with open("README.md", "r") as f:
        long_description = f.read()
else:
    long_description = description

setup(
    name="zinnia",
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    packages=package,
    python_requires=">=3.10",
)
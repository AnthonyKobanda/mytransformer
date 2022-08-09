import os

from setuptools import setup, find_packages



current_folder_path = os.path.dirname(os.path.realpath(__file__))

with open(current_folder_path + "/README.md", 'r') as file:
    long_description = file.read()
    file.close()

with open(current_folder_path + "/requirements.txt", 'r') as file:
    requirements = file.readlines()
    file.close()


setup(
    name="mytransformer",
    version="0.0.1",
    python_requires=">=3.10",
    install_requires=requirements,
    packages=find_packages(),
    description="Implementations of transformers with Jax, PyTorch and TensorFlow .",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnthonyKobanda/mytransformer",
    author="Anthony Kobanda",
    author_email="anthony.kobanda@gmail.com",
    project_urls={
        "Issues": "https://github.com/AnthonyKobanda/mytransformer/issues",
        "Source": "https://github.com/AnthonyKobanda/mytransformer"
    },
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="streamlit-demo-self-driving",
    version="0.0.3",
    author="Streamlit Inc",
    author_email="hello@streamlit.io",
    description="Self-deriving car demo for Streamlit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://streamlit.io",
    packages=setuptools.find_packages(),
    install_requires = requirements
)


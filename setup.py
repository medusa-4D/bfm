import os
from setuptools import setup, find_packages

PACKAGES = find_packages()

# Get version and release info, which is all stored in shablona/version.py
ver_file = os.path.join("bfm", "version.py")

with open(ver_file) as f:
    exec(f.read())

# Long description will go up on the pypi page
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

opts = dict(
    name=NAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    version=VERSION,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    # install_requires=[
    #     "click",
    #     "pyyaml",
    #     "numpy",
    #     "scipy",
    #     "chumpy",   # for loading in FLAME model
    #     "scikit-image",  # for estimating crop transform
    #     "face_alignment"  # for determining crop bbox
    # ]
)

if __name__ == "__main__":
    setup(**opts)

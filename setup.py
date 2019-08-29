import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emotion predictor",
    version="0.0.1",
    author="Kilian & wshop19 ",
    description="Emotion predictor + GEIST data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaros1024/emotion-predictor",
    packages=setuptools.find_packages(),
    install_requires=['pandas', 'matplotlib', 'sklearn',
                      'biosppy', 'pyEDFlib', 'NeuroKit'],
    dependency_link=['git+https://github.com/neuropsychology/NeuroKit.py/zipball/master'],
    scripts=['bin/preprocess_geist', 'bin/emotion_predictor_main'],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
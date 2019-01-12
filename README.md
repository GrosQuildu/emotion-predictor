# Emotion predicting model
This is an implementation of model that predicts emotion basing on BVP and GSR syntax, trained by the DEAP dataset.

#### Additional required external libraries
* [HeartPy](https://github.com/paulvangentcom/heartrate_analysis_python) (integrated; no need to install)
* [NeuroKit](https://github.com/neuropsychology/NeuroKit.py)
* [BioSPPy](https://github.com/PIA-Group/BioSPPy)
* [pyEDFlib](https://github.com/holgern/pyedflib)

#### HOWTO
All you need to do is:
1. Install the libraries listed above.
1. Edit the config.py file
1. Set paths:
    * `DATA_PATH` - should be set to directory where initially preprocessed files (.dat) are stored.
    * `ORIGINALS_PATH` - should be set to directory where original, unprocessed files (.bdf) are stored.
    * `OUT_FILE` - should be set to exact path to file which will be created in order to cache preprocessing results.
1. Run `main.py` with Python 3 (prepared for 3.6).

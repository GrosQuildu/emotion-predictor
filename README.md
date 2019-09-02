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


### wshop 19 - GEIST data

##### Uruchomienie

Najpierw ustawić ścieżki i inne zmienne w `emotion_predictor/config.py` oraz `emotion_predictor/preprocess_geist.py`.
Następnie:

```bash
mkvirtualenv --python=/usr/bin/python3.7 emotion_predictor
python ./setup.py build && python ./setup.py install

preprocess_geist  # zainstalowane w ~/.virtualenvs/emotion_predictor/bin/preprocess_geist
emotion_predictor_main
```

##### Wykonano

Wstępne przetworzenie danych GEIST

    * plik preprocess_geist.py (konfiguracja na początku skryptu)

    * korzystanie tylko z pomiarów narzędzia BITalino

    * filtracja eksperymentów (obecność plików itp.)

    * przetworzenie pliku z nacechowaniem emocjonalnym obrazków (NAPS_valence_arousal_2014.csv)
        ```
        format danych:
        pictures: dict[picture_name] = {'valence': 1.82, 'arousal': 7.05}
        ```

    * przetworzenie sygnałów z BITalino
        * naprawa stref czasowych w plikach

        * pominięcie eksperymentów bez odpowiednio długiego czasu spoczynku

        * pominięcie źle wykonanych eksperymentów (stałe wartości sygnałów)

        * konwersja jednostek (GSR microSiemens -> Ohm)

        * samplowanie do odpowiedniej częstotliwości

        * podział danych na nacechowane emocjonalnie i spoczynkowe

        * wykresy z przetworzonymi danymi

        * zapis do plików .pickle
        ```
        format danych wyjściowych:

        emotionized: dict[path_to_experiment][picture_names][signal] = [signal_value, signal_value2,...]
        resting: dict[path_to_experiment][signal] = [signal_value, signal_value2,...]
        ```

Poprawa struktury projektu

    * dodanie pliku setup.py

    * dodanie skryptów wykonywalnych
    
Modyfikacja oryginalnego kodu, tak żeby działał z nowymi danymi


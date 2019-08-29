from emotion_predictor.config import CLASSIFICATION_METHOD


def get_class_for_values(valence, arousal):
    """
    Returns class by given valence and arousal values, regarding the classification method in config
    :param valence: float, emotional valence
    :param arousal: float, emotional arousal
    :return: string, class
    """
    if CLASSIFICATION_METHOD == "both":
        return _get_class_by_both(valence, arousal)
    if CLASSIFICATION_METHOD == "valence":
        return _get_class_by_valence(valence)
    if CLASSIFICATION_METHOD == "arousal":
        return _get_class_by_arousal(arousal)

    raise ValueError(f"Invalid classification method: {CLASSIFICATION_METHOD}")


def _get_class_by_both(valence, arousal):
    if valence < 4:
        if arousal < 4:
            return "vL_aL"
        if 4 <= arousal < 6.5:
            return "vL_aM"
        return "vL_aH"
    elif 4 <= valence < 6.5:
        if arousal < 4:
            return "vM_aL"
        if 4 <= arousal < 6.5:
            return "vM_aM"
        return "vM_aH"
    else:
        if arousal < 4:
            return "vH_aL"
        if 4 <= arousal < 6.5:
            return "vH_aM"
        return "vH_aH"


def _get_class_by_valence(valence):
    if valence < 4:
        return "vL"

    if 4 <= valence < 6.5:
        return "vM"

    return "vH"


def _get_class_by_arousal(arousal):
    if arousal < 4:
        return "aL"

    if 4 <= arousal < 6.5:
        return "aM"

    return "aH"

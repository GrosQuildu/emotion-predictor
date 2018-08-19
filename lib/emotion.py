
def get_class_for_values(valence, arousal):
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

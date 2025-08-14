"""
Transform module for TFX pipeline.
Defines preprocessing for numeric and binary features.
"""

import tensorflow_transform as tft

NUMERIC_FEATURES = [
    'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep',
    'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram',
    'sc_h', 'sc_w', 'talk_time'
]

BINARY_FEATURES = [
    'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'
]

LABEL_KEY = 'price_range'


def preprocessing_fn(inputs):
    """
    Preprocessing logic executed by TFX Transform component.
    """
    outputs = {}

    for key in NUMERIC_FEATURES:
        outputs[key] = tft.scale_to_z_score(inputs[key])

    for key in BINARY_FEATURES:
        outputs[key] = inputs[key]

    outputs[LABEL_KEY] = inputs[LABEL_KEY]

    return outputs
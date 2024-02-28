import tensorflow_transform as tft
from functools import partial
def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    # Since we are modifying some features and leaving others unchanged, we
    # start by setting `outputs` to a copy of `inputs.
    NUMERIC_FEATURE_KEYS = ['Elevation', 
                            'Slope', 
                            'Horizontal_Distance_To_Hydrology', 
                            'Vertical_Distance_To_Hydrology', 
                            'Horizontal_Distance_To_Roadways', 
                            'Hillshade_9am', 'Hillshade_Noon', 
                            'Horizontal_Distance_To_Fire_Points']
    CATEGORICAL_FEATURE_KEYS = ['Wilderness_Area', 'Soil_Type']


    outputs = {}

    # Scale numeric columns to have range [0, 1].
    for key in NUMERIC_FEATURE_KEYS:
        outputs[key] = tft.scale_to_0_1(inputs[key])

    # For all categorical columns except the label column, we generate a
    # vocabulary but do not modify the feature.  This vocabulary is instead
    # used in the trainer, by means of a feature column, to convert the feature
    # from a string to an integer id.
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[key] = tft.compute_and_apply_vocabulary(
                                            inputs[key],
                                            num_oov_buckets=1,
                                            vocab_filename=key)

    return outputs
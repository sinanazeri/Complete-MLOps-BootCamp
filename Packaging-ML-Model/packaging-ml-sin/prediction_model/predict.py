
from pathlib import Path
import os
import sys
# adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
# we have to use the reslut from training pipeline

import pandas as pd
import numpy as np
import joblib
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset


classification_pipeline = load_pipeline(config.MODEL_NAME)

def generate_prediction(data_input):
    data = pd.DataFrame(data_input)
    prediction = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(prediction==1,'Y','N')
    result = {"prediction":output}
    return result

# for prelimanary testing
# def generate_prediction():
    # test_data = load_dataset(config.TEST_FILE)
    # data = pd.DataFrame(test_data)
    # predictions = classification_pipeline.predict(data[config.FEATURES])
    # output = np.where(predictions==1,'Y','N')
    # return output

if __name__=='main':
    generate_prediction()
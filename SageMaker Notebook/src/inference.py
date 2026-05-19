import os
import json
import joblib
import pandas as pd
import sys

try:
    import pre_processing
    sys.modules['src.pre_processing'] = pre_processing
except ImportError:
    pass

def model_fn(model_dir):

    model_path = os.path.join(
        model_dir,
        "model.pkl"
    )

    model = joblib.load(model_path)

    return model


def input_fn(request_body, request_content_type):

    if request_content_type == "application/json":

        data = json.loads(request_body)

        df = pd.DataFrame([data])

        return df

    raise Exception(
        f"Unsupported content type: {request_content_type}"
    )


def predict_fn(input_data, model):

    prediction = model.predict(input_data)

    result = (
        "Transported"
        if int(prediction[0]) == 1
        else "Not Transported"
    )

    return {
        "prediction": result
    }


def output_fn(prediction, accept):

    return json.dumps(prediction)
from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# all these transformation must have .fit method
classification_pipeline = Pipeline(
    [
        ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('ModeImpuration', pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('DomainProcessing', pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY, variable_to_add=config.FEATURE_TO_ADD)),
        ('DropFeatures',pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransformation',pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('MinMaxScale',MinMaxScaler()),
        ('LogisticClassifier',LogisticRegression(random_state=0))
    ]
)


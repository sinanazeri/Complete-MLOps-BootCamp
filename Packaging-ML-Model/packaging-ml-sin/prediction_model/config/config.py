import pathlib
import os
import prediction_model

# i am interested in getting prediction_model root
PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve.parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

#now it is easy to access datasets
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

MODEL_NAME = 'classification.pkl'
#where we want to save model
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")

TARGET = 'Loan_Status'

#lets have feature list that we want to keep (final features)
FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

# in our case it is same as Categorical features
FEATURES_TO_ENCODE = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD = 'CoapplicantIncome'

DROP_FEATURES = ['CoapplicantIncome']

LOG_FEATURES = ['ApplicantIncome','LoanAmount']
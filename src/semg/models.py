import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, DMatrix


def make_model(name):
    if name.startswith("SVM"):
        return SVC(kernel="rbf", C=10. if name == "SVM_C10" else 1., gamma="scale",
                   decision_function_shape="ovr")
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    if name.startswith("XGB"):
        return XGBClassifier(n_estimators=400, max_depth=4 if name == "XGB_depth4" else 6,
                             learning_rate=.05, subsample=.9, colsample_bytree=.8,
                             objective="multi:softprob", reg_lambda=1.,
                             tree_method="hist", device="cuda", random_state=42)
    raise ValueError(name)


def predict(model, x):
    if isinstance(model, XGBClassifier):
        # DMatrix explicitly transfers CPU arrays; avoids inplace CPU fallback.
        return np.argmax(model.get_booster().predict(DMatrix(x)), axis=1)
    return model.predict(x)

from cuml.ensemble import RandomForestClassifier
from cuml.datasets import make_classification
import numpy as np
import cuml 
from cupy import asnumpy

X,y = make_classification(n_samples=1000, n_features=20, n_classes=4,n_informative=5)
X=X.astype(np.float32)
print(X.shape)
print(y.shape)
model=RandomForestClassifier(max_depth=10, n_estimators=25)
model.fit(X,y)
print(model.predict_proba(X))
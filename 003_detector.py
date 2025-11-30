# detector.py
import numpy as np, os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

clean = np.load("features/clean.npy")
X = [clean]
y = [np.zeros(1000)]

for f in os.listdir("features"):
    if f != "clean.npy":
        X.append(np.load(f"features/{f}"))
        y.append(np.ones(1000))

X = np.vstack(X)
y = np.hstack(y)

clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
clf.fit(X, y)
print(classification_report(y, clf.predict(X)))

onnx_model = convert_sklearn(clf, initial_types=[('input', FloatTensorType([None, X.shape[1]]))])
onnx.save(onnx_model, "TinyBERT-Defender.onnx")
print("Defender size:", round(os.path.getsize("TinyBERT-Defender.onnx")/1e6, 2), "MB")
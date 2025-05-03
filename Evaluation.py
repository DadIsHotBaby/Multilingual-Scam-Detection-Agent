import numpy as np
from sklearn.metrics import classification_report

preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)

print("\n🧪 Classification Report:")
print(classification_report(y_test, y_pred))

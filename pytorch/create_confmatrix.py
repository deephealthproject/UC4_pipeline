import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size' : 16
})
target_names = ['Background', 'Nodule']
cm = np.zeros((2,2))
sens = 0.82
spec = 0.74
cm[0,0] = spec
cm[0,1] = 1-spec
cm[1,1] = sens
cm[1,0] = 1 -sens
print("TN: {}".format(cm[0,0]))
print("FN: {}".format(cm[1,0]))
print("TP: {}".format(cm[1,1]))
print("FP: {}".format(cm[0,1]))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues")
#disp.ax_.set_title("Normalized Confusion Matrix")
plt.tight_layout()
#plt.show()
plt.savefig("confusion_matrix_normalized_resnet.pdf")

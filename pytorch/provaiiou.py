from sklearn.metrics import jaccard_score

y_pred = [0, 0, 0, 0]
y_true = [0, 0, 0, 0]
print(jaccard_score(y_true, y_pred))
import numpy as np


data = [
{'epoch': 49, 'time_epoch': 39.98404, 'loss': 0.03544966, 'lr': 0, 'params': 13815836, 'time_iter': 0.02921, 'accuracy': 0.9848, 'auc': 0.88459, 'ap': 0.3038},
{'epoch': 49, 'time_epoch': 40.36411, 'loss': 0.0366794, 'lr': 0, 'params': 13815836, 'time_iter': 0.02948, 'accuracy': 0.98478, 'auc': 0.88182, 'ap': 0.2629},
{'epoch': 49, 'time_epoch': 39.98404, 'loss': 0.03544966, 'lr': 0, 'params': 13815836, 'time_iter': 0.02921, 'accuracy': 0.9848, 'auc': 0.88459, 'ap': 0.3038},
{'epoch': 49, 'time_epoch': 40.36411, 'loss': 0.0366794, 'lr': 0, 'params': 13815836, 'time_iter': 0.02948, 'accuracy': 0.98478, 'auc': 0.88182, 'ap': 0.2629},
{'epoch': 49, 'time_epoch': 39.98404, 'loss': 0.03544966, 'lr': 0, 'params': 13815836, 'time_iter': 0.02921, 'accuracy': 0.9848, 'auc': 0.88459, 'ap': 0.3038},

]
eval_auroc_values = [i['ap'] for i in data]
mean_auroc = np.mean(eval_auroc_values)
std_auroc = np.std(eval_auroc_values)

print("Mean eval_auroc:", mean_auroc)
print("Standard deviation of eval_auroc:", std_auroc)

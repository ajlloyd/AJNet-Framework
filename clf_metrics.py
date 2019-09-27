import numpy as np

##### Confusion_matrix and clf metrics classes:

class matrix:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    def confusion_matrix(self):
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        true_pred = np.c_[self.y_true, self.y_pred]
        for row in true_pred:
            if row[0] == 1 and row[1] == 1:
                TP += 1
            if row[0] == 0 and row[1] == 0:
                TN += 1
            if row[0] == 1 and row[1] == 0:
                FN += 1
            if row[0] == 0 and row[1] == 1:
                FP += 1
        return np.array(([TP, FP],[FN, TN]))

class metrics(matrix):
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)
        self.cm = super().confusion_matrix().ravel()
        self.TP = self.cm[0]
        self.FP = self.cm[1]
        self.FN = self.cm[2]
        self.TN = self.cm[3]
    def accuracy(self):
        accuracy = (self.TN + self.TP) / (self.TP + self.FP + self.FN + self.TN)
        return accuracy
    #When it is {A} how often does {B} occur?
    def recall(self):
        # Sensitivity/Recall/TPR --> when {y_true = 1} how often does {y_pred = 1}?
        return self.TP / (self.TP + self.FN)
    def false_positive_rate(self):
        # FPR --> when {y_true = 0} how often does {y_pred = 1}?
        return self.FP / (self.TN + self.FP)
    def true_negative_rate(self):
        # Specificity --> when {y_true = 0} how often does {y_pred = 0}?
        actual_negatives = len(self.y_true[self.y_true==0])
        return self.TN / (self.TN + self.FP)
    def precision(self):
        # Precision --> when {y_pred = 1} how often does {y_true = 1}
        return self.TP / (self.TP + self.FP)
    def F1_score(self):
        return (2 * ((self.precision() * self.recall())/(self.precision() + self.recall())))

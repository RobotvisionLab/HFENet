#########Caculate the average FN,FP###########

import numpy as np

   #prediction
#label
# P\L     P    N
# P      TP    FN
# N      FP    TN

# PA = (TP + TN) / (TP + FP + FN + TN)
# MPA = (TP/(TP + FP)  + TN/(TN + FN)) / 2
# IoU:
#    defect：IoU1 = TP / (TP + FN + FP)
#    background：IoU2 = TN / (TN + FP + FN)
# MIoU = (IoU1 + IoU2) / 2
class Bin_classification_cal:
    TP_total = 0
    FP_total = 0
    FN_total = 0
    TN_total = 0
    
    img_total = 0
    threshold = 0

    def __new__(self, output, label, thr, clear=False):
        self.img_total += 1
        return super(Bin_classification_cal, self).__new__(self)
    
    def __init__(self, output, label, thr, clear=False):
        self.threshold = thr
        self.output = output
        self.count = 0
        self.label = label
        if clear:
            self.clear()
        self.getParams()
    
    def getParams(self):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        
        #二值化
        self.output = np.where(self.output > self.threshold, 1, 0)
        self.output =  self.output.reshape(-1, self.output.size).squeeze().tolist()
        self.count = self.label.size
        self.label = self.label.reshape(-1,  self.count).squeeze().tolist()
        
        for i in range(len(self.output)):
            if self.label[i] == 1 and self.output[i] == 1:
                TP += 1
                Bin_classification_cal.TP_total += 1
            elif self.label[i] == 1:
                FN += 1
                Bin_classification_cal.FN_total += 1
            elif self.output[i] == 1:
                FP += 1
                Bin_classification_cal.FP_total += 1
            else:
                TN += 1
                Bin_classification_cal.TN_total += 1
        
        self.TP = TP
        self.FP = FP
        self.FN = FN
        self.TN = TN
        
        
    def caculate_single(self):
        PA = ( self.TP +  self.TN) / ( self.TP +  self.FP +  self.FN +  self.TN)
        MPA = (self.TP/( self.TP + self.FP + 0.01)  + self.TN/(self.TN + self.FN + 0.01)) / 2
        IoU1 = self.TP / ( self.TP + self.FN + self.FP + 0.01)

        if self.TP == 0 and self.FN == 0 and self.FP == 0:
            IoU1 = 1

        IoU2 = self.TN / ( self.TN + self.FP + self.FN + 0.01)
        MIoU = (IoU1 + IoU2) / 2
        
        return self.caculate_total()
        
        
    def caculate_total(self):
        tp = Bin_classification_cal.TP_total
        tn = Bin_classification_cal.TN_total
        fp = Bin_classification_cal.FP_total
        fn = Bin_classification_cal.FN_total
           
        PA = (tp +  tn) / ( tp + fp +  fn +  tn)
        fp_total =  self.count * (fp / (tp + fp +  fn +  tn))
        fn_total = self.count * (fn / (tp + fp +  fn +  tn))

        return PA, fp_total, fn_total

    def clear(self):
        print('clear...')
        Bin_classification_cal.TP_total = 0
        Bin_classification_cal.TN_total = 0
        Bin_classification_cal.FP_total = 0
        Bin_classification_cal.FN_total = 0
        Bin_classification_cal.img_total = 1

    


from data import srdata
import os
class SR291(srdata.SRData):
    def __init__(self, args, name='SR291', train=True, benchmark=False):
        super(SR291, self).__init__(args, name=name)

    def _set_filesystem(self, dir_data):  # dir_data="E:/data/ffoutput"
        self.apath = os.path.join(dir_data, 'Train_291')
        self.dir_hr = os.path.join(self.apath, 'Train_291_HR')
        self.dir_lr = os.path.join(
            self.apath, 'Train291_train_LR_bicubic'
        )
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.bmp', '.bmp')
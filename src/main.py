import torch

import utility
import data
import model
import loss
import thop
from option import args
from trainer import Trainer

if __name__ == '__main__':

    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)

    test = [args.r_mean, args.g_mean, args.b_mean]
    print(test)

    if args.data_test == 'video':
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)  
            model = model.Model(args, checkpoint)        
            params = list(model.parameters())
            k = 0
            for i in params:
                l = 1          
                for j in i.size():
                    l *= j          
                k = k + l
            print("parameters：" + str(k))
            loss = loss.Loss(args, checkpoint) if not args.test_only else None  
            t = Trainer(args, loader, model, loss, checkpoint)  
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()


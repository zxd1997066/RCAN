import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    if args.channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
            print("---- Use NHWC model")
        except:
            print("---- Use normal model")
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    if args.precision == "bfloat16":
        print("---- Running with bfloat16...")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16): 
            while not t.terminate():
                #t.train()
                t.test()
    else:
        while not t.terminate():
            t.test()

    checkpoint.done()


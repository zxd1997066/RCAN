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
    if args.compile:
        model = torch.compile(model, backend=args.backend, options={"freezing": True})
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        args.device = "cuda"
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
    if args.precision == "float16":
        print("---- Running with float16...")
        if args.device == "cpu":
            print('---- Enable CPU AMP float16')
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.half): 
                while not t.terminate():
                    #t.train()
                    t.test()
        elif args.device == "cuda":
            print('---- Enable CUDA AMP float16')
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.half): 
                while not t.terminate():
                    #t.train()
                    t.test()
            
    else:
        while not t.terminate():
            t.test()

    checkpoint.done()


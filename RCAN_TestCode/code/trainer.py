import os
import math
import time
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        total_time = 0.0
        total_sample = 0
        batch_time_list = []
        
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]
                    if self.args.channels_last:
                        lr = lr.to(memory_format=torch.channels_last)

                    if self.args.profile:
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU],
                            record_shapes=True,
                            schedule=torch.profiler.schedule(
                                wait=int(self.args.num_iter/2),
                                warmup=2,
                                active=1,
                            ),
                            on_trace_ready=self.trace_handler,
                        ) as p:
                            for i in range(self.args.num_iter):
                                tic = time.time()
                                sr = self.model(lr, idx_scale)
                                p.step()
                                toc = time.time()
                                elapsed = toc - tic
                                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                                if i >= self.args.num_warmup:
                                    total_time += elapsed
                                    total_sample += 1
                                    batch_time_list.append((toc - tic) * 1000)
                    else:
                        for i in range(self.args.num_iter):
                            tic = time.time()
                            sr = self.model(lr, idx_scale)
                            toc = time.time()
                            elapsed = toc - tic
                            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                            if i >= self.args.num_warmup:
                                total_time += elapsed
                                total_sample += 1
                                batch_time_list.append((toc - tic) * 1000)

                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])
                    if self.args.save_results:
                        #self.ckp.save_results(filename, save_list, scale)
                        self.ckp.save_results_nopostfix(filename, save_list, scale)

                    break


                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
                break

        print("\n", "-"*20, "Summary", "-"*20)
        latency = total_time / total_sample * 1000
        throughput = total_sample / total_time
        print("inference latency:\t {:.3f} ms".format(latency))
        print("inference Throughput:\t {:.2f} samples/s".format(throughput))
        # P50
        batch_time_list.sort()
        p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
        p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
        p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
        print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
                % (p50_latency, p90_latency, p99_latency))
 
        self.ckp.write_log(
            'Total time: {:.2f}s, ave time: {:.2f}s\n'.format(timer_test.toc(), timer_test.toc()/len(self.loader_test)), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def trace_handler(self, p):
        output = p.key_averages().table(sort_by="self_cpu_time_total")
        print(output)
        import pathlib
        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
        if not os.path.exists(timeline_dir):
            try:
                os.makedirs(timeline_dir)
            except:
                pass
        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                    'RCAN-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
        p.export_chrome_trace(timeline_file)


    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import os
import contextlib


from .fixmatch_utils import consistency_loss, Get_Scalar, CSLLabeled, CSLUnlabeled

from utils import get_metrics
import wandb

from sklearn.metrics import recall_score, confusion_matrix

class FixMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, vlr,\
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None,classes =[]):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """
        
        super(FixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m
        self.classes = classes
        self.lambdas = [1.0/self.num_classes] * self.num_classes
        self.prior = None
        self.val_lr = vlr
        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        
        self.train_model = net_builder(num_classes=num_classes) 
        self.eval_model = net_builder(num_classes=num_classes)
        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T) #temperature params function
        self.p_fn = Get_Scalar(p_cutoff) #confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label
        
        self.optimizer = None
        self.scheduler = None
        
        self.it = 0
        
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        
        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
            
        self.eval_model.eval()
            
            
    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.train_model.module.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        
        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)            
    
     
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')    
            
    
    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    
    def train(self, args, logger=None):
        """
        Train function of FixMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()
        if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            wandb.init(project=args.wandb_project, id=args.wandb_runid, entity=args.wandb_entity)
        #lb: labeled, ulb: unlabeled
        self.train_model.train()
        
        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        start_batch.record()
        best_eval_acc, best_it = 0.0, 0
        
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        for (x_lb, y_lb), (x_ulb_w, x_ulb_s, _) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
            if self.it % self.num_eval_iter == 0:
                criterion_l, criterion_u = self.validate(args)
                
            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()
            
            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]
            
            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            
            # inference and calculate sup/unsup losses
            with amp_cm():
                logits = self.train_model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                del logits

                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                sup_loss = criterion_l(logits_x_lb, y_lb, reduction='mean')
                unsup_loss, mask = criterion_u(logits_x_ulb_w, logits_x_ulb_s)
                if args.vanilla_opt:
                    total_loss = sup_loss + self.lambda_u * unsup_loss
                else:
                    loss_r = 0
                    for parameter in self.train_model.parameters():
                        loss_r += torch.sum(parameter ** 2)
                    total_loss = sup_loss + self.lambda_u * unsup_loss + 1e-4 * loss_r

            
            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()
            self.train_model.zero_grad()
            
            with torch.no_grad():
                self._eval_model_update()
            
            end_run.record()
            torch.cuda.synchronize()
            
            #tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach() 
            tb_dict['train/unsup_loss'] = unsup_loss.detach() 
            tb_dict['train/total_loss'] = total_loss.detach() 
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach() 
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch)/1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run)/1000.
            
            
            if self.it % self.num_eval_iter == 0:
                eval_dict, metrics = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                
                save_path = os.path.join(args.save_dir, args.save_name)
                
                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it
                
                self.print_fn(f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")
            
            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if self.it % self.num_eval_iter == 0:
                    wandb.log(metrics)
                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)
                
                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)
                
            self.it +=1
            del tb_dict
            start_batch.record()
            if self.it > 2**19:
                self.num_eval_iter = 1000
        
        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict
            
            
    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        use_ema = hasattr(self, 'eval_model')
        
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        logits_, labels_ = [], []
        for x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = eval_model(x)
            
            logits_.append(torch.argmax(logits, dim=1).cpu().detach().numpy())
            labels_.append(y.cpu().detach().numpy())

            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)
            
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()
        
        if not use_ema:
            eval_model.train()
        logits_, labels_ = (np.concatenate(logits_, axis=0), np.concatenate(labels_, axis=0))
        metrics = get_metrics(logits_, labels_, self.classes)
        for i, l in enumerate(self.lambdas):
            metrics["val/lambda " + str(i)] = l
        return {'eval/loss': total_loss/total_num, 'eval/top-1-acc': total_acc/total_num}, metrics
    
    @torch.no_grad()
    def validate(self, args):
        torch.cuda.synchronize()
        print("doing a validation run")
        use_ema = hasattr(self, 'eval_model')
        device = torch.device('cuda:' + str(args.rank))
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        val_loader = self.loader_dict['val']
        
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        logits_, labels_ = [], []
        for x, y in val_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = eval_model(x)
            
            logits_.append(torch.argmax(logits, dim=1).cpu().detach().numpy())
            labels_.append(y.cpu().detach().numpy())

            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)
            
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()
        
        if not use_ema:
            eval_model.train()
        logits_, labels_ = (np.concatenate(logits_, axis=0), np.concatenate(labels_, axis=0))

        G = self.get_gain(logits_, labels_)
                
        criterion_l = CSLLabeled(G, device)
        criterion_u = CSLUnlabeled(G, kl_thresh=args.kl_thresh, device=device)
        return criterion_l, criterion_u
    
    def get_gain(self, logits, labels):
        if self.args.M == "min_recall":
            recall = recall_score(labels, logits, average=None, zero_division=0)
            new_lamdas = [x * np.exp(-1 * self.val_lr * r) for x, r in zip(self.lambdas, recall.tolist())]
            new_lamdas = [x/sum(new_lamdas) for x in new_lamdas]
            self.lambdas = new_lamdas
            diagonal = [x/p for x,p in zip(self.lambdas, self.prior)]
            G = np.diag(diagonal)
        elif self.args.M == "mean_recall_coverage":
            CM = confusion_matrix(labels, logits, normalize="all")
            new_lamdas = []
            C = np.sum(CM, axis=0).tolist()
            for i, (l, c, p) in enumerate(zip(self.lambdas, C, self.prior)):
                l_ = l - self.val_lr * (c - 0.95/self.num_classes)
                l_ = max(0, l_)
                new_lamdas.append(l_)
            G = np.zeros((self.num_classes, self.num_classes))
            D = np.zeros((self.num_classes, self.num_classes))
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i==j:
                        G[i, i] = (1.0/self.num_classes)/self.prior[i] + new_lamdas[j]
                        D[i, i] = (1.0/self.num_classes)/self.prior[i] + new_lamdas[j]
                    else:
                        G[i, j] = new_lamdas[j]
        elif self.args.M == "min_ht_recall":
            logits_, labels_ = (np.concatenate(logits_, axis=0), np.concatenate(labels_, axis=0))

            recall = recall_score(labels_, logits_, average=None, zero_division=0)
            head_recall = np.mean(recall[:int(0.9 * self.num_classes)])
            tail_recall = np.mean(recall[int(0.9 * self.num_classes):])

            lamda_h, lamda_t = self.lambdas[0], self.lambdas[-1]

            lamda_h = lamda_h * np.exp(-1 * self.val_lr * head_recall)
            lamda_t = lamda_t * np.exp(-1 * self.val_lr * tail_recall)

            lamda_h = lamda_h/(lamda_h + lamda_t)
            lamda_t = 1 - lamda_h

            new_lamdas_ = [lamda_h/(0.9 * self.num_classes)] * int(0.9 * self.num_classes) + \
                        [lamda_t/(0.1 * self.num_classes)] * int(0.1 * self.num_classes)
            self.lambdas = [lamda_h] * int(0.9 * self.num_classes) + [lamda_t] * int(0.1 * self.num_classes)
        
            diagonal = [x/p for x,p in zip(self.lambdas, self.prior)]
            G = np.diag(diagonal)
        elif self.args.M == "mean_recall_ht_coverage":
            new_lamdas = []
            CM = confusion_matrix(labels, logits, normalize="all")
            C = np.sum(CM, axis=0).tolist()

            l_head, l_tail = self.lambdas[0], self.lambdas[-1]
            head_coverage, tail_coverage =  np.mean(C[:int(0.9 * self.num_classes)]),\
                                            np.mean(C[int(0.9 * self.num_classes):]) 
            l_head = l_head - self.val_lr * (head_coverage - 0.95/self.num_classes)
            l_tail = l_tail - self.val_lr * (tail_coverage - 0.95/self.num_classes)
            l_head = max(0, l_head)
            l_tail = max(0, l_tail)
            new_lamdas = [l_head] * int(0.9 * self.num_classes) + [l_tail] * int(0.1 * self.num_classes)

            G = np.zeros((self.num_classes, self.num_classes))
            D = np.zeros((self.num_classes, self.num_classes))

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i==j:
                        G[i, i] = (1.0/self.num_classes)/self.prior[i] + new_lamdas[j]/int(0.9 * self.num_classes)
                        D[i, i] = (1.0/self.num_classes)/self.prior[i] + new_lamdas[j]/int(0.1 * self.num_classes)
                    else:
                        G[i, j] = new_lamdas[j]
        return G

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)
        
        self.print_fn(f"model saved: {save_filename}")
    
    
    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        
        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key]) 
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")

if __name__ == "__main__":
    pass

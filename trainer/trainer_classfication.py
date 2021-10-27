import numpy as np
import torch
from torchvision.utils import make_grid
from base.base_trainer_seg import BaseTrainer
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from utils import label_accuracy_score, add_hist


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,# data_loader, valid_data_loader=None, 
                data_set  = None, transform = None,
                 lr_scheduler=None, len_epoch=None, cut_mix = False, beta= 0.8, mix_up= False):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        # self.data_loader = data_loader
        # self.valid_data_loader = valid_data_loader
        # self.do_validation = self.valid_data_loader is not None
        self.do_validation = True
        self.data_set = data_set
        self.transform = transform
        self.cut_mix = cut_mix 
        self.beta = beta
        self.mix_up = mix_up
        if len_epoch is None:
        #     # epoch-based training
            self.len_epoch = 1000
        else:
            # iteration-based training
            self.data_loader = inf_loop(self.data_set)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        # self.log_step = int(np.sqrt(data_loader.batch_size)) * 4
        self.log_step = int(np.sqrt(64)) * 4

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
        labels = [value["label"] for key, value in self.data_set.data_dic.items()]
        k_idx = 0
        for train_index, validation_index in stratified_kfold.split(self.data_set, labels):
            k_idx+=1
            print(f'####### K-FOLD :: {k_idx}th')
            train_dataset = torch.utils.data.dataset.Subset(self.data_set,train_index)
            
            copied_dataset = copy.deepcopy(self.data_set)
            valid_dataset = torch.utils.data.dataset.Subset(copied_dataset,validation_index)
            
            train_dataset.dataset.set_transform(self.transform.transformations['train'])
            valid_dataset.dataset.set_transform(self.transform.transformations['val'])        
            self.data_loader = DataLoader(
                        train_dataset,
                        batch_size=50, 
                        num_workers=4,
                        shuffle=True)

            self.valid_data_loader = DataLoader(
                        valid_dataset,
                        batch_size=50,
                        num_workers=4,
                        shuffle=False) 
            target_count = 0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.type(torch.FloatTensor).to(self.device), target.to(self.device)
                target_count+=len(target)
                self.optimizer.zero_grad()
                rand_num = np.random.random_integers(3) # 같이 넣어야 할까?
                if self.cut_mix and rand_num==1: # cutmix가 실행될 경우 
                    data, target_a, target_b, lam = cutmix_data(data, target, self.beta, self.device)
                    outputs = self.model(data)
                    loss= mixs_criterion(self.criterion, outputs, target_a, target_b, lam)
                elif self.mix_up and rand_num ==2:
                    data, target_a, target_b, lam = mixup_data(data, target)
                    outputs = self.model(data)
                    loss= mixs_criterion(self.criterion, outputs, target_a, target_b, lam)
                else:
                    outputs= self.model(data) 
                    loss= self.criterion(outputs, target) 

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(outputs, target))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
            log = self.train_metrics.result()

            if self.do_validation:
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_'+k : v for k, v in val_log.items()})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(loss)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        target_count = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                target_count += len(target)
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

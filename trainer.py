import os
import time
import rich
import torch 
import matplotlib.pyplot as plt
from torch import nn

class Trainer:
    def __init__(self, problem, iterators, devices, params):
        self.model = problem['model']
        self.criterion = problem['criterion']
        self.optimizer = problem['optimizer']
        self.lr_scheduler = problem['lr_scheduler']

        self.train_loader = iterators['train']
        self.val_loader = iterators['val']
        
        self.device = devices['device']
        self.comp = devices['comp']

        self.epochs = params['epochs']
        self.gradient_clipping = params['gradient_clipping']
        self.clip = params['clip']
        self.best_val_loss = params['best_val_loss']
        self.is_wholebody = params['is_wholebody']
        self.use_physics = params['use_physics']
        self.links = params['links']
        self.lr_warm_up = params['lr_warm_up']
        self.weights = params['weights']

        # NOTE: no physics-informed loss here!
        self.loss_labels = ["all", "s", "sdot"]
        self.train_losses = {
            "all": [],
            "s": [],
            'sdot': []
        }
        self.val_losses = {
            "all": [],
            "s": [],
            "sdot": []
        }

    def accumulate_epoch_loss(self, existing_losses, new_losses):
        r"""Accumulate loss for each epoch training/validation."""
        for item in self.loss_labels:
            existing_losses[item] += new_losses[item]
        return existing_losses

    def avg_epoch_loss(self, existing_losss, length):
        r"""Compute the average loss for each epoch."""
        avg_losses = {}
        for item in self.loss_labels:
            avg_losses[item] = existing_losss[item] / length
        return avg_losses

    @staticmethod
    def time_gap(t0, t1):
        if not isinstance(t0, (int, float)) or not isinstance(t1, (int, float)):
            raise TypeError("Both start_time and end_time must be numbers (int or float).")
        if t0 > t1:
            raise ValueError("start_time cannot be greater than end_time.")
        mins, secs = divmod(int(t1 - t0), 60)
        return mins, secs

    def get_input(self, sample):
        r"""
        Return the batched sample inputs:
            - acc: (bs, T_in, 3*5)
            - ori: (bs, T_in, 9*5)
            - s_buffer & sdot_buffer: (bs, T_in, 31)
        """
        acc = sample['acc'].to(dtype=torch.float64, device=self.device)
        ori = sample['ori'].to(dtype=torch.float64, device=self.device)
        s_buffer = sample['s_buffer'].to(dtype=torch.float64, device=self.device)
        sdot_buffer = sample['sdot_buffer'].to(dtype=torch.float64, device=self.device)
        return {'acc': acc, 'ori': ori, 's_buffer': s_buffer, 'sdot_buffer': sdot_buffer}
    
    def get_target(self, sample):
        r"""
        Return the batched sample output targets:
            - s & sdot: (bs, T_out, 31)
            - pb: (bs, T_out, 3)
            - rb: (bs, T_out, 9)
            - vb: (bs, T_out, 6)
        NOTE: no link pose or twist, since not considering PI loss here.
        """
        s = sample['s'].to(dtype=torch.float64, device=self.device)
        sdot = sample['sdot'].to(dtype=torch.float64, device=self.device)

        pb = sample['pb'].to(dtype=torch.float64, device=self.device)
        rb = sample['rb'].to(dtype=torch.float64, device=self.device)
        vb = sample['vb'].to(dtype=torch.float64, device=self.device)
        return {'s': s, 'sdot': sdot, 'pb': pb, 'rb': rb, 'vb': vb}
    
    def compute_loss(self, y_gt, y_pred):
        s_loss = self.weights['s'] * self.criterion(y_gt['s'], y_pred['s'])
        sdot_loss = self.weights['sdot'] * self.criterion(y_gt['sdot'], y_pred['sdot'])
        return {'s': s_loss, 'sdot': sdot_loss}
    
    def step_train(self, sample):
        self.model.train()
        self.model.to(self.device)
        # prepare inputs and targets
        inputs = self.get_input(sample)
        targets = self.get_target(sample)

        # forward process
        s_pred, sdot_pred = self.model(
            inputs['acc'], inputs['ori'],
            inputs['s_buffer'], inputs['sdot_buffer']
        )
        # compute the loss
        loss_dict = self.compute_loss(
            y_gt={'s': targets['s'], 'sdot': targets['sdot']},
            y_pred={'s': s_pred, 'sdot': sdot_pred}
        )
        # back propagation and update
        batch_loss = loss_dict['s'] + loss_dict['sdot']
        self.optimizer.zero_grad()
        batch_loss.backward()

        if self.gradient_clipping:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # summarize the per-batch loss
        loss_per_batch = {
            'all': batch_loss.item(),
            's': loss_dict['s'].item(),
            'sdot': loss_dict['sdot'].item()
        }
        return loss_per_batch
    
    def step_val(self, sample):
        self.model.eval()
        self.model.to(self.device)
        # get IO features
        inputs = self.get_input(sample)
        targets = self.get_target(sample)
        # apply prediction
        s_pred, sdot_pred = self.model(
            inputs['acc'], inputs['ori'],
            inputs['s_buffer'], inputs['sdot_buffer']
        )
        # compute evaluation loss
        loss_dict = self.compute_loss(
            y_gt={'s': targets['s'], 'sdot': targets['sdot']},
            y_pred={'s': s_pred, 'sdot': sdot_pred}
        )
        batch_loss = loss_dict['s'] + loss_dict['sdot']
        loss_per_batch = {
            'all': batch_loss.item(),
            's': loss_dict['s'].item(),
            'sdot': loss_dict['sdot'].item()
        }
        return loss_per_batch
    
    def save(self, epoch, val_loss):
        save_dir = "./models/finetuned"
        os.makedirs(save_dir, exist_ok=True)
        model_filename = f"{save_dir}/model_epoch{epoch}_{val_loss:.5f}.pt"
        torch.save(self.model.state_dict(), model_filename)
    
    def plot(self):
        plt.figure(figsize=(8, 6))
        for item in self.loss_labels:
            plt.plot(self.train_losses[item], label=f'train {item}', marker='o')
            plt.plot(self.val_losses[item], label=f"val {item}", marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Training & Validation Loss Curves")
        plt.legend()
        plt.grid()
        plt.show()
    
    def run(self):
        train_t0 = time.time()
        try:
            for epoch in range(1, self.epochs+1):
                # reset the losses at the beginning of each epoch
                epoch_train_losses = {"all": 0, "s": 0, "sdot": 0}
                epoch_val_losses = {"all": 0, "s": 0, "sdot": 0}
                
                epoch_t0 = time.time()
                # step training 
                for _, train_sample in enumerate(self.train_loader):
                    # return a dict of training losses based on current batch
                    train_losses = self.step_train(train_sample)
                    epoch_train_losses = self.accumulate_epoch_loss(epoch_train_losses, train_losses)
                # compute the average loss for current epoch
                avg_epoch_train_losses = self.avg_epoch_loss(epoch_train_losses, len(self.train_loader))
                # save the average epoch loss
                for  item in self.loss_labels:
                    self.train_losses[item].append(avg_epoch_train_losses[item])

                # step validation
                with torch.no_grad():
                    for _, val_sample in enumerate(self.val_loader):
                        val_losses = self.step_val(val_sample)
                        epoch_val_losses = self.accumulate_epoch_loss(epoch_val_losses, val_losses)
                    avg_epoch_val_losses = self.avg_epoch_loss(epoch_val_losses, len(self.val_loader))
                    for item in self.loss_labels:
                        self.val_losses[item].append(avg_epoch_val_losses[item])
                # compute cost time per epoch
                epoch_t1 = time.time()
                epoch_mins, epoch_secs = self.time_gap(epoch_t0, epoch_t1)
                
                # save the model if the val loss decreases
                if avg_epoch_val_losses['all'] < self.best_val_loss:
                    self.save(epoch, avg_epoch_val_losses['all'])
                    # update the best val loss
                    self.best_val_loss = avg_epoch_val_losses['all']
                    rich.print(f"Best model saved at epoch {epoch} with eval loss {avg_epoch_val_losses['all']:.5f}.")
                
                # adjust the learning rate after warm-up epochs
                if epoch > self.lr_warm_up:
                    self.lr_scheduler.step()

                # print epoch information
                print(f"Epoch: [{epoch}/{self.epochs}] | "
                      f"Cost time: {epoch_mins}m {epoch_secs}s | "
                      f"Train loss: {avg_epoch_train_losses['all']:.5f} - "
                      f"s: {avg_epoch_train_losses['s']:.5f}, sdot: {avg_epoch_train_losses['sdot']:.5f} || "
                      f"Val loss: {avg_epoch_val_losses['all']:.5f} - "
                      f"s: {avg_epoch_val_losses['s']:.5f}, sdot: {avg_epoch_val_losses['sdot']:.5f} || "
                      f"Learning rate: {self.optimizer.param_groups[0]['lr']}"
                )
            # compute the cost time for the whole training
            train_t1 = time.time()
            total_mins, total_secs = self.time_gap(train_t0, train_t1)
            print(f"Training loop cost {total_mins}m {total_secs}s after {self.epochs} epochs!")
            # plot loss curves
            self.plot()
        except KeyboardInterrupt:
            print("Training loop broken due to external interruption!")



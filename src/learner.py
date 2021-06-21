import torch
from utils.custom_enumerator import enumerateWithEstimate

def noop(*a, **k):
    return None

class Learner:
    def __init__(self, model, train_dl, val_dl, loss_func, lr, cbs, opt_func):
        self.model=model
        self.train_dl=train_dl
        self.val_dl=val_dl
        self.loss_func=loss_func
        self.lr=lr
        self.cbs=cbs
        self.opt_func=opt_func

        for cb in cbs: 
            cb.learner=self

    def one_batch(self):
        self('before_batch')
        xb,yb=self.batch 
        self.preds=self.model(xb)
        self.loss=self.loss_func(self.preds, yb)
        if self.model.training:
            self.opt.zero_grad()
            self.loss.backward()
            self.opt.step()
        self('after_batch')
        
    def one_epoch(self, is_train):
        self('before_epoch')
        self.model.training=is_train

        if self.model.training:
            self.model.train()
        else:
            self.model.eval()

        dl=self.train_dl if is_train else self.val_dl
        for self.batch_idx,self.batch in enumerate(dl):
            self.one_batch()
        self('after_epoch')

    def fit(self, n_epochs):
        self('before_fit')
        self.opt=self.opt_func(self.model.parameters(), self.lr)
        self.n_epochs=n_epochs

        for self.epoch_idx in enumerateWithEstimate(range(n_epochs), desc_str="Training status"):
            self.one_epoch(is_train=True)
            with torch.no_grad():
                self.one_epoch(is_train=False)
        self('after_fit')

    def __call__(self, cb_method_name):
        for cb in self.cbs:
            getattr(cb, cb_method_name, noop)()

    def save(self,pth):
        torch.save(self.model.state_dict(), pth)

    def load(self,pth):
        self.model.load_state_dict(torch.load(pth))

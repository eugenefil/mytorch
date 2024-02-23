import svetoch.autograd as ag


class Learner:
    def __init__(self, train_data_loader, model, optim, loss_fn,
            valid_data_loader=None, callbacks=None):
        self.train_dl = train_data_loader
        self.valid_dl = valid_data_loader
        self.model = model
        self.optim, self.loss_fn = optim, loss_fn
        self.callbacks = [] if callbacks is None else callbacks
        for cb in self.callbacks:
            cb.learn = self

    def do_callbacks(self, name):
        for cb in self.callbacks:
            method = getattr(cb, name, None)
            if method:
                method()

    def fit(self, n_epochs):
        self.do_callbacks("before_fit")
        for epoch in range(n_epochs):
            self.epoch = epoch
            self.training = True
            self.do_callbacks("before_epoch")
            for self.xb, self.yb in self.train_dl:
                self.preds = self.model(self.xb)
                self.loss = self.loss_fn(self.preds, self.yb)
                self.do_callbacks("after_loss")
                self.loss.backward()
                self.optim.step()

            self.training = False
            with ag.no_grad():
                for self.xb, self.yb in self.valid_dl:
                    self.preds = self.model(self.xb)
                    self.loss = self.loss_fn(self.preds, self.yb)
                    self.do_callbacks("after_loss")
            self.do_callbacks("after_epoch")
        self.do_callbacks("after_fit")

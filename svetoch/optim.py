
class SGD:
    def __init__(self, params, lr, l2_decay=0., zero_grad=True,
                 set_to_none=False):
        self.params, self.lr, self.l2_decay = params, lr, l2_decay
        self._zero_grad = zero_grad
        self.set_to_none = set_to_none

    @no_grad()
    def step(self):
        for p in self.params:
            if self.l2_decay > 0.:
                p *= 1. - self.l2_decay
            p -= self.lr * p.grad
        if self._zero_grad:
            self.zero_grad(self.set_to_none)

    def zero_grad(self, set_to_none=False):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

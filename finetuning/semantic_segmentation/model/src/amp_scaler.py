import torch.cuda.amp as amp

class AmpScaler:
    def __init__(self):
        self.scaler = amp.GradScaler()

    def __call__(self, loss, optimizer):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        self.scaler.step(optimizer)
        self.scaler.update()

        return


    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict):
        self.scaler.load_state_dict(state_dict)
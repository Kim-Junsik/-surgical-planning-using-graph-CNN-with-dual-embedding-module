import torch
# from pytorch_lightning.callbacks import Callback

"""
기본 callback 함수
progress.ProgressBar
model_checkpoint.ModelCheckpoint --> lightning_logs/version * 디렉터리에 log저장
"""

class AdjMatrixSetting:
    def on_train_batch_end(self, pl_module):
        sd = pl_module.state_dict()
        sd['adj'] = sd['adj']-torch.diag(torch.diag(sd['adj']))
        pl_module.load_state_dict(sd)


def set_callbacks():
    callbacks = []
    callbacks.append(AdjMatrixSetting())
    return callbacks
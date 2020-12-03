import torch
import os

def save_checkpoint(model, optimizer, scheduler, epoch, device, path):
    model.to("cpu")
    model_name = path + str(epoch).zfill(4) + ".tar"
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, os.path.join(path, model_name))
    model.to(device)

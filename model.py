import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import os

      
def get_model(num_classes, pre_train=True):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pre_train)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


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


def load_checkpoint(model, optimizer, scheduler, device, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return model, optimizer, scheduler, checkpoint['epoch']

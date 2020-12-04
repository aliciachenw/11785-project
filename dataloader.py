import os, json, cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class FLIRDataset(Dataset):

    def __init__(self, root, json_file_name, transforms=None):
        self.root = root
        print(root)
        json_file = os.path.join(root, json_file_name)
        print(json_file)
        with open(json_file) as f:
            json_data = json.load(f)
        imgs = json_data["images"]
        self.categories = json_data["categories"]
        self.info = json_data["info"]
        self.license = json_data["licenses"]

        # filter out not pedestrians annotation
        self.annotations = list(filter(lambda x:x["category_id"]==1, json_data["annotations"]))

        self.imgs = []
        self.labels = {}

        # filter out negative instances without pedestrians
        for i, img in enumerate(imgs):
            annotation = list(filter(lambda x:x["image_id"] == img["id"], self.annotations))
            if len(annotation) > 0:
                self.imgs.append(img)
                self.labels[img["id"]] = annotation
        
        self.transforms = transforms

      
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images ad masks
        img = self.imgs[idx]
        img_id = img["id"]
        img_filename = img["file_name"]
        img_path = os.path.join(self.root, img_filename)
        # img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)


        img_annotations = self.labels[img_id]
        num_objs = len(img_annotations)
        boxes = []
        area = []
        for i, anno in enumerate(img_annotations):
            bbox = anno["bbox"]
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            area.append(anno["area"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def pseudo_labeling():
        pass
    

def split_dataset(dataset, ratio, seed=1):
    torch.manual_seed(seed)
    data_len = dataset.__len__()
    num_part = int(np.ceil(data_len * ratio))
    dataset_1, dataset_2 = random_split(dataset, [num_part, data_len - num_part])
    return dataset_1, dataset_2

def split_training_dataset(dataset, labeled_ratio, training_ratio):
    # split the dataset in train and vaidation set
    labeled_dataset, unlabeled_dataset = split_dataset(dataset, labeled_ratio)
    labeled_train_dataset, labeled_val_dataset = split_dataset(labeled_dataset, training_ratio)
    unlabeled_train_dataset, unlabeled_val_dataset = split_dataset(unlabeled_dataset, training_ratio)
    return labeled_train_dataset, labeled_val_dataset, unlabeled_train_dataset, unlabeled_val_dataset
        
    
def get_dataloader(dataset, batch_size, is_train=False, labeled_ratio=1, training_ratio=0.7):
    if is_train:
        # split into labeled and unlabeled
        labeled_dataset, unlabeled_dataset = split_dataset(dataset, labeled_ratio)
        
        # split into train and validation
        labeled_train_dataset, labeled_val_dataset = split_dataset(labeled_dataset, training_ratio)
        unlabeled_train_dataset, unlabeled_val_dataset = split_dataset(unlabeled_dataset, training_ratio)

        # create dataloader
        labeled_train_dataloader = DataLoader(labeled_train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        labeled_val_dataloader = DataLoader(labeled_val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        unlabeled_val_dataloader = DataLoader(unlabeled_val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return labeled_train_dataloader, unlabeled_train_dataloader, labeled_val_dataloader, unlabeled_val_dataloader
    else:
        dataloader = DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=False)
        return dataloader

def collate_fn(batch):
    return tuple(zip(*batch))

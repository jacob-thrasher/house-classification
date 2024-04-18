import os
import torch
import transformers
import timm
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from data import ErieParcels

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

root = 'D:\\Big_Data'
model_path = "google/vit-base-patch16-224"
pretrained_model = 'vit_adam-lr3e-5_no-wd_seed_99/best_model.pt'

test_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), year_regression=False, model_path=model_path)
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# model = transformers.Swinv2ForImageClassification.from_pretrained(model_path)
# model = transformers.ViTForImageClassification.from_pretrained(model_path)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = torch.nn.Linear(768, 2)

state_dict = torch.load(pretrained_model)
new_dict = OrderedDict()
for key in state_dict:
    value = state_dict[key]
    if 'module' in key:
        key = key.replace('module.', '')

    new_dict[key] = value

model.load_state_dict(new_dict)
model.eval()
imgs, _ = next(iter(dataloader))
print(imgs[0])

imgs = imgs.to('cuda')
model.to('cuda')

cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], reshape_transform=reshape_transform)

# Try settings targets
# target = [ClassifierOutputTarget(0)]
target = None
grayscale_cam = cam(input_tensor=imgs, targets=target)

imgs = imgs.cpu()

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

all_plots = []
for i in range(4):
    viz = show_cam_on_image(imgs[i].permute(1, 2, 0).numpy(), grayscale_cam[i, :], use_rgb=True)
    all_plots.append(viz)

ax[0, 0].imshow(all_plots[0])
ax[0, 1].imshow(all_plots[1])
ax[1, 0].imshow(all_plots[2])
ax[1, 1].imshow(all_plots[3])

plt.show()
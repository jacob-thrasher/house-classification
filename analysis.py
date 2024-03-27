import os
import torch
import transformers
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from data import ErieParcels

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

root = '/scratch/jdt0025/erie_data'
model_path = "vit-base-patch16-224"
pretrained_model = 'vit_adam-lr1e-4-wd0/best_model.pt'

test_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erietest.csv'), year_regression=False)
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# model = transformers.Swinv2ForImageClassification.from_pretrained(model_path)
model = transformers.ViTForImageClassification.from_pretrained(model_path)
model.classifier = torch.nn.Linear(768, 2)
model.load_state_dict(torch.load(pretrained_model))
model.eval()

imgs, _ = next(iter(dataloader))

imgs = imgs.to('cuda')
model.to('cuda')

cam = GradCAM(model=model, target_layers=[model.blacks[-1].norm1], reshape_transform=reshape_transform)


viz = show_cam_on_image(imgs, cam, use_rgb=True)

print(type(viz))
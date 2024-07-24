import mmcv
import os
import pandas as pd
from PIL import Image
from mmseg.apis import MMSegInferencer

# MMSegInferencer.list_models('mmseg')

root = 'D:\\Big_Data\\Erie'
model = 'upernet'

dst = os.path.join(root, f'masks/models/{model}')
os.mkdir(os.path.join(dst))


df = pd.read_csv(os.path.join(root, 'cv/0/top-4s.csv'))
parcels = df['parcel_number'].tolist()
paths = [os.path.join(root, 'parcels_cleaned', f'{p}.png') for p in parcels]




inferencer = MMSegInferencer(model='upernet_r50_4xb2-40k_cityscapes-512x1024')

results = inferencer(paths, wait_time=0.5)

for pred, parcel in zip(results['predictions'], parcels):
    preds_norm = pred / 20 # number of classes


    pil_img = Image.fromarray(preds_norm*255).convert('RGB')
    pil_img.save(os.path.join(dst, f'{parcel}.png'))

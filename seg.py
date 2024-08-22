import mmcv
import os
import pandas as pd
from PIL import Image
from mmseg.apis import MMSegInferencer

# MMSegInferencer.list_models('mmseg')

root = '/users/jdt0025/scratch/Erie'
model = 'segformer'

dst = os.path.join(root, f'active_learning/masks/{model}')
os.mkdir(os.path.join(dst))


df = pd.read_csv(os.path.join(root, 'active_learning/active_learning/test.csv'))
parcels = df['parcel_number'].tolist()
parcels = [f'{p}.png' for p in parcels]
parcels = list(set(parcels).intersection(set(os.listdir(os.path.join(root, 'parcels_cleaned')))))
paths = [os.path.join(root, 'parcels_cleaned', p) for p in parcels]


print(len(paths))
inferencer = MMSegInferencer(model='segformer_mit-b0_8xb1-160k_cityscapes-1024x1024')


batch_size = 2
for i in range(int(len(paths) / batch_size)):
    batch = paths[i*batch_size:(i+1)*batch_size]
    batch_parcels = parcels[i*batch_size:(i+1)*batch_size]


    results = inferencer(paths, wait_time=0.5)

    for pred, parcel in zip(results['predictions'], parcels):
        preds_norm = pred / 20 # number of classes


        pil_img = Image.fromarray(preds_norm*255).convert('RGB')
        pil_img.save(os.path.join(dst, f'{parcel}.png'))

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

plt.ion()

root = 'D:\\Big_Data\\Erie'
status = 'Inactive'
split = 'val'

df = pd.read_csv(os.path.join(root, f'cv/0/test.csv'))
# labeled = pd.read_csv(os.path.join(root, f'cv/0/top-4s.csv'))
# df = df[~df['parcel_number'].isin(labeled)]

# Open csv file, writer header if file does not yet exist
csvpath = os.path.join(root, f'cv/0/top-4s.csv')
if not os.path.exists(csvpath):
    f = open(csvpath, 'w')
    writer = csv.writer(f)
    header = ['parcel_number', 'homestead_status', 'top1', 'top2', 'top3', 'top4']
    writer.writerow(header)
else:
    labeled_df = pd.read_csv(csvpath)
    df = df[~df['parcel_number'].isin(labeled_df['parcel_number'])]

    f = open(csvpath, 'a')
    writer = csv.writer(f)

# Filter dataframe
if status == 'Active':
    df = df[df['homestead_status'] == 'Active']
elif status == 'Inactive':
    df = df[df['homestead_status'] == 'Inactive']
else:
    raise ValueError(f'Incorrect stats, expected one of [Active, Inactive], got {status}')

# Iterate dataframe rows
exit_loop = False
counter = 0

for i, row in df.iterrows():
    parcel_id = row['parcel_number']
    img_path = os.path.join(root, 'parcels', parcel_id, '0.png')
    if os.path.exists(img_path):
        img = Image.open(img_path)
    else:
        continue

    plt.imshow(img)
    plt.show()

    entry = [parcel_id, status]
    i = 0
    print(f'\n({counter})')
    # Get top 4s
    while i < 4:
        _input = input(f'Input top-{i+1} feature: ')

        if _input == 'exit': 
            exit_loop = True
            break
        elif _input in ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']:
            entry.append(_input)
            i += 1
        else:
            # If invalid input, do not increment loop control variable
            print(f'Invalid input ({_input}), please try again')
    
    if exit_loop: break
    writer.writerow(entry)
    plt.close()
    counter += 1

f.close()

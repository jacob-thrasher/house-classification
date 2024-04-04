import numpy as np
import cv2
import os
from datasets import Dataset
from PIL import Image
#from patchify import patchify  # Only to handle large images
from torch.utils.data import Dataset as TorchDataset
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai
from tqdm import tqdm
import torch
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Install necessary packages
# !pip install opencv-python
# !pip install patchify
# !pip install datasets
# !pip install monai

# Mount Google Drive
# drive.mount('/content/drive')

# Define the folder paths in your Google Drive
images_folder = "/home/shivam/Desktop/maya_research/Resized/images"
masks_folder = "/home/shivam/Desktop/maya_research/Resized/masks"

# List all files in the images folder
image_files = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if
               file.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

# List all files in the masks folder
mask_files = [os.path.join(masks_folder, file) for file in os.listdir(masks_folder) if
              file.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

# Sort both lists to ensure images and masks are loaded in the same order
image_files.sort()
mask_files.sort()

# Read images and masks together to preserve correspondence
images_and_masks = [(cv2.imread(image_file), cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE))  # Remove cv2.IMREAD_GRAYSCALE
                    for image_file, mask_file in zip(image_files, mask_files)]

# Resize images and masks to 256x256
resized_images_and_masks = [(cv2.resize(image, (256, 256)), cv2.resize(mask, (256, 256))) for image, mask in
                             images_and_masks]

# Unpack images and masks into separate lists
resized_images, resized_masks = zip(*resized_images_and_masks)

# Convert list of images to NumPy array
resized_images_array = np.array(resized_images)
resized_masks_array = np.array(resized_masks)

# Define functions and classes

def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


class SAMDataset(TorchDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs


# Convert the NumPy arrays to Pillow images and store them in a dictionary
dataset_dict = {
    "image": [Image.fromarray(img) for img in resized_images_array],  # Ensure proper conversion to uint8
    "label": [Image.fromarray(mask) for mask in resized_masks_array],  # Ensure proper conversion to uint8
}

train_images, test_images, train_masks, test_masks = train_test_split(resized_images_array, resized_masks_array, test_size=0.2, random_state=42)
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)



train_dataset_dict = {
    "image": [Image.fromarray(img) for img in train_images],
    "label": [Image.fromarray(mask) for mask in train_masks],
}

val_dataset_dict = {
    "image": [Image.fromarray(img) for img in val_images],
    "label": [Image.fromarray(mask) for mask in val_masks],
}

test_dataset_dict = {
    "image": [Image.fromarray(img) for img in test_images],
    "label": [Image.fromarray(mask) for mask in test_masks],
}

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(train_dataset_dict)
val_dataset = Dataset.from_dict(val_dataset_dict)
test_dataset = Dataset.from_dict(test_dataset_dict)




processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

train_dataset = SAMDataset(dataset=dataset, processor=processor)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(SAMDataset(val_dataset, processor), batch_size=2, shuffle=False)

model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 250

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

best_val_loss = float('inf')
best_model_state = None
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in train_dataloader:
        model.train()

        # Forward pass
        inputs = {k: v.to(device) for k, v in batch.items() if k != "ground_truth_mask"}
        outputs = model(**inputs, multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)

        # Calculate loss
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    mean_train_loss = mean(epoch_losses)
    epoch_losses.append(mean_train_loss)

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')



        # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "ground_truth_mask"}
            outputs = model(**inputs, multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)

            # Calculate loss
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            val_losses.append(loss.item())
    mean_val_loss = mean(val_losses)
    val_losses.append(mean_val_loss)

    print(f'Validation loss: {mean(val_losses)}')


    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        best_model_state = model.state_dict()
        # Save the best model
        torch.save(best_model_state, "/home/shivam/Desktop/save/best_model_SAM_1.pth")

plt.plot(epoch_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig("/home/shivam/Desktop/save/training_validation_losses_plot_1.png")
plt.show()

model.load_state_dict(torch.load("/home/shivam/Desktop/save/best_model_SAM_1.pth"))
#filename = '/home/shivam/Desktop/save/model_checkpoint_4.pth'
from sklearn.metrics import accuracy_score



model.eval()

# Create a folder to save predicted masks
save_folder = "/home/shivam/Desktop/save/predicted_masks_val_1_2"
os.makedirs(save_folder, exist_ok=True)

# Define functions to calculate IoU and Dice scores
def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculate_dice(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    dice_score = 2 * np.sum(intersection) / (np.sum(gt_mask) + np.sum(pred_mask))
    return dice_score

iou_scores = []
dice_scores = []

# Loop through the test dataset for evaluation
for idx in range(len(test_dataset)):
    image = test_dataset[idx]["image"]
    ground_truth_mask = np.array(test_dataset[idx]["label"])
    prompt = get_bounding_box(ground_truth_mask)
    
    # Perform inference
    inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    medsam_seg = (medsam_seg_prob > 0.5).cpu().numpy().squeeze()
    
    # Calculate IoU and DICE
    iou_score = calculate_iou(ground_truth_mask, medsam_seg)
    dice_score = calculate_dice(ground_truth_mask, medsam_seg)
    
    iou_scores.append(iou_score)
    dice_scores.append(dice_score)

    # Visualize and save the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title("Actual Image")
    axs[1].imshow(ground_truth_mask, cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(medsam_seg, cmap='gray')
    axs[2].set_title("Predicted Mask")
    plt.savefig(os.path.join(save_folder, f"image_{idx}.png"))
    plt.close()

# Compute mean IoU and DICE scores
mean_iou = np.mean(iou_scores)
mean_dice = np.mean(dice_scores)

print(f"Mean IoU: {mean_iou}")
print(f"Mean DICE: {mean_dice}")
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv


config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')
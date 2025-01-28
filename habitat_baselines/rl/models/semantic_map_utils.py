import torch
import torch.nn.functional as F
from habitat_baselines.rl.models.networks.resnetUnet import ResNetUNet
from habitat_baselines.common.utils import Model


def get_network_from_options(config, img_segmentor=None):
    if config.WITH_IMG_SEGM:
        # with torch.no_grad():
        img_segmentor = ResNetUNet(n_channel_in=3, n_class_out=config.N_OBJECT_CLASSES)

        for name, p in img_segmentor.named_parameters():
            if "base_model" in name:
                p.requires_grad = False
            # print(name, p.requires_grad)

        model_utils = Model()
        latest_checkpoint = model_utils.get_latest_model(save_dir=config.IMG_SEGM_MODEL_DIR)
        print(f"Loading image segmentation checkpoint: {latest_checkpoint}")

        checkpoint = torch.load(latest_checkpoint)
        img_segmentor.load_state_dict(checkpoint['models']['img_segm_model'], strict=False)
        # img_segmentor.eval()

    return img_segmentor


def run_img_segm(model, input_obs, img_labels=None):
    input = input_obs['rgb'] / 255.0  # No need to permute or divide in-place

    # Assuming input is [bs, H, W, C]
    B, H, W, _ = input.shape

    # If the model allows batch processing, you can pass the entire batch
    pred_segm_raw = model(input.permute(0, 3, 1, 2))  # Assuming the model expects [bs, C, H, W]
    C = pred_segm_raw.shape[1]
    pred_segm_raw = pred_segm_raw.view(B, C, H, W)
    pred_segm = F.softmax(pred_segm_raw, dim=1)

    return pred_segm

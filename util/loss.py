import torch
import torch.nn.functional as F

import dnnlib.util


class ReconstructionLoss(torch.nn.Module):
    """
    L1
    """

    def __init__(
        self,
        coarse_hole_alpha,
        coarse_nohole_alpha,
        refine_hole_alpha,
        refine_nohole_alpha,
    ):
        super(ReconstructionLoss, self).__init__()
        self.coarse_hole_alpha = coarse_hole_alpha
        self.coarse_nohole_alpha = coarse_nohole_alpha
        self.refine_hole_alpha = refine_hole_alpha
        self.refine_nohole_alpha = refine_nohole_alpha

    def forward(self, image, coarse, refined, mask):
        mask_flat = mask.view(mask.size(0), -1)

        if mask.min() == 0.0 and mask.max() == 1.0:
            loss_a = self.coarse_nohole_alpha * torch.mean(
                torch.abs(image - coarse)
                * mask
                / mask_flat.mean(1).view(-1, 1, 1, 1)
            )
            loss_b = self.coarse_hole_alpha * torch.mean(
                torch.abs(image - coarse)
                * (1 - mask)
                / (1 - mask_flat.mean(1).view(-1, 1, 1, 1))
            )
            loss_c = self.refine_nohole_alpha * torch.mean(
                torch.abs(image - refined)
                * mask
                / mask_flat.mean(1).view(-1, 1, 1, 1)
            )
            loss_d = self.refine_hole_alpha * torch.mean(
                torch.abs(image - refined)
                * (1 - mask)
                / (1 - mask_flat.mean(1).view(-1, 1, 1, 1))
            )

            return loss_a + loss_b + loss_c + loss_d
        else:
            return torch.abs(coarse - image).mean() + torch.abs(refined - image).mean()



class PerceptionLoss(torch.nn.Module):
    """
    VGG16
    """
    def __init__(self):
        # url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytуorch/pretrained/metrics/vgg16.pt"
        with dnnlib.util.open_url(util/vgg16.pt) as f:
            self.vgg16 = torch.jit.load(f).eval()
    
    def forward(self, images, coarses, refineds, masks):
        if images.shape[2] > 256:
            images = F.interpolate(images, size=(256, 256), mode="area")
            coarses = F.interpolate(coarses, size=(256, 256), mode="area")
            refineds = F.interpolate(refineds, size=(256, 256), mode="area")
        target_features = self.vgg16(images, resize_images=False, return_lpips=True)
        coarse_features = self.vgg16(coarses, resize_images=False, return_lpips=True)
        refineds_features = self.vgg16(refineds, resize_images=False, return_lpips=True)

        return (target_features - coarse_features).square().mean() + \
               (target_features - refineds_features).square().mean()

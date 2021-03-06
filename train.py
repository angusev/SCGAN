from util import arguments
import comet_ml
from pytorch_lightning.loggers import WandbLogger, CometLogger
from pytorch_lightning import Trainer
from util import constants
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl
import os
from pathlib import Path
from util import constants
from util.misc import count_parameters
import torch
import numpy as np
from model import get_generator
from model.InpaintSADiscriminator import InpaintSADiscriminator
from dataset import InpaintDataset
from util.loss import ReconstructionLoss, PerceptionLoss
from util.transforms import ToNumpyRGB256
from PIL import Image

from dataset.InpaintDataset import SCDataModule


pl.utilities.seed.seed_everything(None)
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


class DeepFillV2(pl.LightningModule):
    def __init__(self, args):
        super(DeepFillV2, self).__init__()
        self.hparams = args

        self.net_G = get_generator(args)
        self.net_D = InpaintSADiscriminator(args.input_nc)

        if args.load_G:
            self.net_G.load_state_dict(torch.load(args.load_G), strict=False)
        if args.load_D:
            self.net_D.load_state_dict(torch.load(args.load_D), strict=False)

        self.recon_loss = ReconstructionLoss(
            args.l1_c_h, args.l1_c_nh, args.l1_r_h, args.l1_r_nh
        )
        if args.vgg_weight > 0:
            self.vgg_loss = PerceptionLoss()
        self.refined_as_discriminator_input = args.refined_as_discriminator_input
        # self.visualization_dataloader = self.setup_dataloader_for_visualizations()

    def configure_optimizers(self):
        lr = self.hparams.lr
        decay = self.hparams.weight_decay
        opt_g = torch.optim.Adam(self.net_G.parameters(), lr=lr, weight_decay=decay)
        opt_d = torch.optim.Adam(self.net_D.parameters(), lr=4 * lr, weight_decay=decay)
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, colormap, sketch, mask = (
            batch["image"],
            batch["colormap"],
            batch["sketch"],
            batch["mask"],
        )

        generator_input = torch.cat(
            (image * mask, colormap * (1 - mask), sketch * (1 - mask), mask), dim=1
        )
        coarse_image, refined_image = self.net_G(generator_input)
        reconstruction_loss = args.l1_weight * self.recon_loss(
            image, coarse_image, refined_image, mask
        )
        if args.vgg_weight > 0:
            vgg_loss = args.vgg_weight * self.vgg_loss(
                image, coarse_image, refined_image, mask
            )
        else:
            vgg_loss = 0.0

        if not self.hparams.l1_only:
            discriminator_input_real = torch.cat(
                (image, colormap * (1 - mask), sketch * (1 - mask), mask), dim=1
            )
            discriminator_input_fake = torch.cat(
                (coarse_image, colormap * (1 - mask), sketch * (1 - mask), mask), dim=1
            )
            d_fake = self.net_D(discriminator_input_fake)
        else:
            d_fake = 0.0

        if optimizer_idx == 0:
            # generator training
            if batch_idx == 0:
                self.visualize_batch(batch, batch_idx, "train")

            gen_loss = (
                -self.hparams.gen_loss_alpha * torch.mean(d_fake)
                if not self.hparams.l1_only
                else 0.0
            )
            total_loss = gen_loss + reconstruction_loss + vgg_loss
            return {
                "loss": total_loss,
                "progress_bar": {
                    "gen_loss": gen_loss,
                    "recon_loss": reconstruction_loss,
                },
                "log": {
                    "gen_loss": gen_loss,
                    "train_reconstruction_loss": reconstruction_loss,
                    "train_perceptual_loss": vgg_loss,
                    "total_loss": total_loss,
                },
            }
        if optimizer_idx == 1 and not self.hparams.l1_only:
            # discriminator training
            d_real = self.net_D(discriminator_input_real)

            real_loss = torch.mean(torch.nn.functional.relu(1.0 - d_real))
            fake_loss = torch.mean(torch.nn.functional.relu(1.0 + d_fake))
            disc_loss = self.hparams.disc_loss_alpha * (real_loss + fake_loss)
            return {
                "loss": disc_loss,
                "progress_bar": {"d_real_loss": real_loss, "d_fake_loss": fake_loss},
                "log": {
                    "disc_loss": disc_loss,
                    "real_loss": real_loss,
                    "fake_loss": fake_loss,
                },
            }

    def validation_step(self, batch, batch_idx):
        image, colormap, sketch, mask = (
            batch["image"],
            batch["colormap"],
            batch["sketch"],
            batch["mask"],
        )

        generator_input = torch.cat(
            (image * mask, colormap * (1 - mask), sketch * (1 - mask)), dim=1
        )
        generator_input = torch.cat((generator_input, mask), dim=1)
        coarse_image, refined_image = self.net_G(generator_input)

        reconstruction_loss = self.recon_loss(image, coarse_image, refined_image, mask)
        if batch_idx == 0:
            self.visualize_batch(batch, batch_idx, "valid")
        return {
            "test_loss": torch.FloatTensor(
                [
                    -1.0,
                ]
            ),
            "log": {"val_reconstruction_loss": reconstruction_loss},
        }

    def generate_images(self, batch):
        image, colormap, sketch, mask = (
            batch["image"],
            batch["colormap"],
            batch["sketch"],
            batch["mask"],
        )

        generator_input = torch.cat(
            (image * mask, colormap * (1 - mask), sketch * (1 - mask), mask), dim=1
        )
        coarse_image, refined_image = self.net_G(generator_input)
        completed_image = (
            (refined_image * (1 - mask) + image * mask).detach().cpu().numpy()
        )

        coarse_image = coarse_image.detach().cpu().numpy()
        refined_image = refined_image.detach().cpu().numpy()

        masked_image = image * mask
        masked_sketch = sketch * (1 - mask)
        masked_colormap = colormap * (1 - mask)

        masked_image = masked_image.cpu().numpy()
        masked_sketch = masked_sketch.cpu().numpy()
        masked_colormap = masked_colormap.cpu().numpy()

        return (
            masked_image,
            masked_sketch,
            masked_colormap,
            coarse_image,
            refined_image,
            completed_image,
        )

    def visualize_batch(self, batch, batch_idx, stage):
        torgb = ToNumpyRGB256(-1.0, 1.0)
        (
            masked_image,
            masked_sketch,
            masked_colormap,
            coarse_image,
            refined_image,
            completed_image,
        ) = self.generate_images(batch)
        for j in range(min(4, batch["image"].size(0))):
            visualization = np.hstack(
                [
                    torgb(masked_image[j]),
                    torgb(masked_sketch[j]),
                    torgb(masked_colormap[j]),
                    torgb(coarse_image[j]),
                    torgb(refined_image[j]),
                    torgb(completed_image[j]),
                    torgb(batch["image"][j].cpu().numpy()),
                ]
            )
            image_fromarray = Image.fromarray(visualization[:, :, [2, 1, 0]])
            # image_fromarray.save(
            #     os.path.join(
            #         constants.RUNS_FOLDER,
            #         self.hparams.dataset,
            #         self.hparams.experiment,
            #         "visualization",
            #         str(j) + ".jpg",
            #     )
            # )
            self.logger.experiment.log_image(
                image_fromarray, name=f"{stage}_epoch{self.current_epoch}_{batch_idx}_{j}"
            )

    def validation_epoch_end(self, _):
        torch.save(self.net_G.state_dict(), self.hparams.checkpoint_path / "last_G.pth")
        torch.save(self.net_D.state_dict(), self.hparams.checkpoint_path / "last_D.pth")


if __name__ == "__main__":
    args = arguments.parse_arguments()

    i = 0
    while (Path(constants.RUNS_FOLDER) / f"{args.experiment}_{i}").is_dir():
        i += 1
    checkpoint_path = Path(constants.RUNS_FOLDER) / f"{args.experiment}_{i}"
    checkpoint_path.mkdir(parents=True)
    args.checkpoint_path = checkpoint_path

    # logger = WandbLogger(name="try", project="thesis")
    logger = CometLogger(
        "eM513qOnoTSydF2BDo4Z43su3",
        workspace="angusev",
        project_name="thesis",
        experiment_name=checkpoint_path.name,
    )

    model = DeepFillV2(args)
    train_loader = SCDataModule(
        "/datadrive/mrartemev/data/Students/Andrey/CelebAMask-HQ/",
        dry_try=args.dry_try,
        sc_only=args.sc_only,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    trainer = Trainer(
        gpus=-1,
        precision=16 if args.fp16 else 32,
        logger=logger,
        check_val_every_n_epoch=2,
        profiler="simple",
        deterministic=False,
    )

    trainer.fit(model, train_loader)

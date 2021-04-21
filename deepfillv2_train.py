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
from util.loss import ReconstructionLoss
from util.transforms import ToNumpyRGB256
from PIL import Image

from dataset.InpaintDataset import SCDataModule


torch.backends.cudnn.benchmark = True


class DeepFillV2(pl.LightningModule):
    def __init__(self, args):
        super(DeepFillV2, self).__init__()
        self.hparams = args
        self.net_G = get_generator(args)
        self.net_D = InpaintSADiscriminator(args.input_nc)
        print("#Params Generator: ", f"{count_parameters(self.net_G) / 1e6}M")
        print("#Params Discriminator: ", f"{count_parameters(self.net_D) / 1e6}M")
        self.recon_loss = ReconstructionLoss(
            args.l1_c_h, args.l1_c_nh, args.l1_r_h, args.l1_r_nh
        )
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

        generator_input = torch.cat((image * mask,
                                    colormap * (1 - mask),
                                    sketch * (1 - mask)), dim=1)
        generator_input = torch.cat((generator_input, mask), dim=1)
        coarse_image, refined_image = self.net_G(generator_input)

        reconstruction_loss = self.recon_loss(image, coarse_image, refined_image, mask)

        discriminator_input = torch.cat((colormap, sketch), dim=1)
        discriminator_input = discriminator_input * mask
        discriminator_input_fake = torch.cat(
            (coarse_image, discriminator_input, mask), dim=1
        )
        discriminator_input_real = torch.cat((image, discriminator_input, mask), dim=1)
        d_fake = self.net_D(discriminator_input_fake)

        if optimizer_idx == 0:
            # generator training

            gen_loss = -self.hparams.gen_loss_alpha * torch.mean(d_fake)
            total_loss = gen_loss + reconstruction_loss
            return {
                "loss": total_loss,
                "progress_bar": {
                    "gen_loss": gen_loss,
                    "recon_loss": reconstruction_loss,
                },
                "log": {
                    "gen_loss": gen_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "total_loss": total_loss,
                },
            }

        if optimizer_idx == 1:
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
        torgb = ToNumpyRGB256(-1, 1)
        if batch_idx == 0:
            with torch.no_grad():
                (
                    masked_image,
                    coarse_image,
                    refined_image,
                    completed_image,
                ) = self.generate_images(batch)
                for j in range(min(4, batch["image"].size(0))):
                    visualization = np.hstack(
                        [
                            torgb(masked_image[j]),
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
                    self.logger.experiment.log_image(image_fromarray)
        return {
            "test_loss": torch.FloatTensor(
                [
                    -1.0,
                ]
            )
        }

    def generate_images(self, batch):        
        image, colormap, sketch, mask = (
            batch["image"],
            batch["colormap"],
            batch["sketch"],
            batch["mask"],
        )

        generator_input = torch.cat((image * mask,
                                    colormap * (1 - mask),
                                    sketch * (1 - mask)), dim=1)
        generator_input = torch.cat((generator_input, mask), dim=1)

        generator_input = torch.cat((image, colormap, sketch), dim=1)
        generator_input = generator_input * mask
        generator_input = torch.cat((generator_input, mask), dim=1)
        coarse_image, refined_image = self.net_G(generator_input)
        completed_image = (refined_image * mask + image * (1 - mask)).detach().cpu().numpy()
        coarse_image = coarse_image.detach().cpu().numpy()
        refined_image = refined_image.detach().cpu().numpy()
        masked_image = image * mask
        masked_image = masked_image.cpu().numpy()

        return masked_image, coarse_image, refined_image, completed_image

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_end(self, outputs):
        return {
            "test_loss": torch.FloatTensor(
                [
                    -1.0,
                ]
            )
        }


if __name__ == "__main__":
    args = arguments.parse_arguments()

    # logger = WandbLogger(name="try", project="thesis")
    logger = CometLogger(
        "eM513qOnoTSydF2BDo4Z43su3", workspace="angusev", project_name="thesis"
    )

    i = 0
    while (Path(constants.RUNS_FOLDER) / f"{args.experiment}_{i}").is_dir():
        i += 1
    checkpoint_path = Path(constants.RUNS_FOLDER) / f"{args.experiment}_{i}"
    checkpoint_path.mkdir()
    checkpoint_callback = ModelCheckpoint(
        filename=checkpoint_path,
        period=args.save_epoch,
    )

    model = DeepFillV2(args)
    train_loader = SCDataModule(
        "/home/mrartemev/data/Students/Andrey/CelebAMask-HQ/", dry_try=args.dry_try
    )

    trainer = Trainer(
        gpus=1,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=2,
    )

    trainer.fit(model, train_loader)

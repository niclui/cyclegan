import pytorch_lightning as pl
import torch 
from torch.utils.data import DataLoader

from models import get_model
from eval import get_loss_fn, BinaryClassificationEvaluator
from data import ImageClassificationDemoDataset
from util import constants as C
from .logger import TFLogger


class CycleGANTask(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        # Get my component models
        self.G_A2B = get_model("Generator")
        self.G_B2A = get_model("Generator")
        self.D_A = get_model("Discriminator")
        self.D_B = get_model("Discriminator")

        # Define loss functions
        self.identity_loss = torch.nn.L1Loss()
        self.gan_loss = torch.nn.MSELoss()
        self.cycle_loss = torch.nn.L1loss()

    def forward(self, A, B):
        # Convert the real A/B images into synthetic B/A images
        B2A = self.G_B2A(B)
        A2B = self.G_A2B(A)

        # Then force roundtrip consistency
        B2A2B = self.G_A2B(B2A)
        A2B2A = self.G_B2A(A2B)

        # Separately, try to see if I can recover the exact same image by passing it into generator with the same output domain
        A2A = self.G_B2A(A)
        B2B = self.G_A2B(B)

        return B2A, A2B, B2A2B, A2B2A, A2A, B2B
        

    def training_step(self, batch, batch_nb, optimizer_idx):
        """
        Returns:
            A dictionary of loss and metrics, with:
                loss(required): loss used to calculate the gradient
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        A, B = batch

        B2A, A2B, B2A2B, A2B2A, A2A, B2B = self.forward(A, B)

        # Let's backpropagate
        # Identity Loss - punish myself if the generator alters real images
        loss_identity_A = self.identity_loss(A, A2A) * 5.0
        loss_identity_B = self.identity_loss(B, B2B) * 5.0

        

        # GAN loss - punish myself if discriminator can tell that my fake image is fake
        loss_GAN_A_convTo_B = self.gan_loss()



        loss = self.loss(logits.view(-1), y)
        self.log("loss", loss)
        
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits.view(-1), y)
        y_hat = (logits > 0).float()
        self.evaluator.update((torch.sigmoid(logits), y))
        return loss

    def validation_epoch_end(self, outputs):
        """
        Aggregate and return the validation metrics

        Args:
        outputs: A list of dictionaries of metrics from `validation_step()'
        Returns: None
        Returns:
            A dictionary of loss and metrics, with:
                val_loss (required): validation_loss
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        avg_loss = torch.stack(outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    def train_dataloader(self):
        dataset = ImageClassificationDemoDataset()
        return DataLoader(dataset, shuffle=True,
                          batch_size=2, num_workers=8)

    def val_dataloader(self):
        dataset = ImageClassificationDemoDataset()
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)

    def test_dataloader(self):
        dataset = ImageClassificationDemoDataset()
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)




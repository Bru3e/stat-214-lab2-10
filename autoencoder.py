import lightning as L
import torch


class Autoencoder(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        self.patch_size = patch_size
        self.n_input_channels = n_input_channels

        input_size = int(n_input_channels * (patch_size**2))
    
        self.encoder = torch.nn.Sequential(
            # First block
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),  # GELU activation
            
            # Second block
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            
            # Third block
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            
            # Fourth block
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            
            # Flatten and compress
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(64 * patch_size * patch_size, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, embedding_size),
        )

        # Matching decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 64 * patch_size * patch_size),
            torch.nn.GELU(),
            
            torch.nn.Unflatten(1, (64, patch_size, patch_size)),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            
            torch.nn.Conv2d(32, n_input_channels, kernel_size=3, padding=1),
        )

    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # all the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)

        self.log("train_loss", loss, prog_bar=True)
        # self.log("recon_loss", recon_loss)
        # self.log("l1_reg", l1_reg)
        
        return loss
        

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put for validation is the MSE
        # between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)

        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_recon_loss", recon_loss)
        
        # log the validation loss for experiment tracking purposes
        return loss


    def configure_optimizers(self):
        # set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)

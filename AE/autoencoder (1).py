import lightning as L
import torch
import torch.nn as nn

class Autoencoder(L.LightningModule):
    def __init__(self,n_input_channels=8,embedding_size=8,patch_size=9,sparsity_lambda=1e-2,optimizer_config=None):
        super().__init__()
        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config
        self.sparsity_lambda = sparsity_lambda
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

        #self.encoder_cnn = nn.Sequential(
        #    nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        #    nn.ReLU(),
        #)
        #self.encoder_fc = nn.Linear(32*3*3, embedding_size)

        #self.decoder_fc = nn.Linear(embedding_size, 32*3*3)
        #self.decoder_cnn = nn.Sequential(
        #    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(16, n_input_channels, kernel_size=3, stride=2, padding=1),
        #)

    def forward(self, batch):
        # all the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        mse_loss = torch.nn.functional.mse_loss(decoded, batch)
        reg_loss = self.sparsity_lambda * torch.mean(torch.abs(encoded))
        total_loss = mse_loss + reg_loss
        self.log("train_loss_total", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_mse", mse_loss, prog_bar=True, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        mse_loss = torch.nn.functional.mse_loss(decoded, batch)
        reg_loss = self.sparsity_lambda * torch.mean(torch.abs(encoded))
        total_loss = mse_loss + reg_loss
        self.log("val_loss_total", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss_mse", mse_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        return self.encoder(x)


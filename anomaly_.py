import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
import pickle

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class WaveformDataset(Dataset):
    """Custom dataset for waveform data"""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.FloatTensor(sample), torch.FloatTensor(sample)  # Input and target are the same for autoencoder

class Conv1DAutoencoder(nn.Module):
    """1D Convolutional Autoencoder for time series anomaly detection"""
    def __init__(self, num_channels, filter_size=7, num_filters=16, dropout_prob=0.2, num_downsamples=2):
        super(Conv1DAutoencoder, self).__init__()
        
        self.num_channels = num_channels
        self.num_downsamples = num_downsamples
        
        # Encoder layers
        encoder_layers = []
        in_channels = num_channels
        
        for i in range(num_downsamples):
            out_channels = (num_downsamples + 1 - i) * num_filters
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, filter_size, stride=2, padding=filter_size//2),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        
        for i in range(num_downsamples):
            out_channels = (i + 1) * num_filters
            decoder_layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, filter_size, stride=2, padding=filter_size//2, output_padding=1),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            in_channels = out_channels
        
        # Final reconstruction layer
        decoder_layers.append(
            nn.ConvTranspose1d(in_channels, num_channels, filter_size, stride=1, padding=filter_size//2)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_channels)
        # Conv1d expects: (batch_size, num_channels, sequence_length)
        x = x.transpose(1, 2)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Convert back to original format
        decoded = decoded.transpose(1, 2)
        return decoded

def prepare_data(data, num_downsamples=2):
    """Prepare data by cropping to make divisible by 2^num_downsamples"""
    processed_data = []
    sequence_lengths = []
    
    for sequence in data:
        seq_len = len(sequence)
        cropping = seq_len % (2**num_downsamples)
        if cropping > 0:
            sequence = sequence[:-cropping]
        processed_data.append(sequence)
        sequence_lengths.append(len(sequence))
    
    return processed_data, sequence_lengths

def create_synthetic_data():
    """Create synthetic waveform data for demonstration"""
    num_observations = 100
    sequence_length = 500
    num_channels = 4
    
    data = []
    for _ in range(num_observations):
        # Create synthetic time series with some patterns
        t = np.linspace(0, 10, sequence_length)
        sequence = np.zeros((sequence_length, num_channels))
        
        for ch in range(num_channels):
            # Different frequency patterns for each channel
            freq1 = 0.5 + ch * 0.2
            freq2 = 2.0 + ch * 0.5
            sequence[:, ch] = (np.sin(2 * np.pi * freq1 * t) + 
                              0.5 * np.sin(2 * np.pi * freq2 * t) + 
                              0.1 * np.random.randn(sequence_length))
        
        data.append(sequence)
    
    return data

def plot_training_data(data, num_channels):
    """Plot first 4 observations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(4, len(data))):
        for ch in range(num_channels):
            axes[i].plot(data[i][:, ch], label=f'Channel {ch+1}')
        axes[i].set_title(f'Observation {i+1}')
        axes[i].set_xlabel('Time Step')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def train_autoencoder(model, train_loader, val_loader, num_epochs=120, learning_rate=0.001):
    """Train the autoencoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Handle different sequence lengths
            min_len = min(output.size(1), target.size(1))
            loss = criterion(output[:, :min_len, :], target[:, :min_len, :])
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                min_len = min(output.size(1), target.size(1))
                loss = criterion(output[:, :min_len, :], target[:, :min_len, :])
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def calculate_rmse(model, data_loader, device):
    """Calculate RMSE for each sequence"""
    model.eval()
    rmse_values = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate RMSE for each sequence in the batch
            for i in range(data.size(0)):
                seq_len = target.size(1)
                pred_len = min(output.size(1), seq_len)
                
                mse = torch.mean((output[i, :pred_len, :] - target[i, :pred_len, :]) ** 2)
                rmse = torch.sqrt(mse)
                rmse_values.append(rmse.item())
    
    return np.array(rmse_values)

def inject_anomalies(data, num_anomalous=20, patch_start=50, patch_end=60, scale_factor=4):
    """Inject anomalies into random sequences"""
    data_with_anomalies = [seq.copy() for seq in data]
    anomaly_indices = random.sample(range(len(data)), num_anomalous)
    
    for idx in anomaly_indices:
        sequence = data_with_anomalies[idx]
        if len(sequence) > patch_end:
            patch = sequence[patch_start:patch_end, :]
            sequence[patch_start:patch_end, :] = scale_factor * np.abs(patch)
    
    return data_with_anomalies, anomaly_indices

def detect_anomalous_regions(original, reconstructed, rmse_baseline, window_size=7, threshold_factor=1.1):
    """Detect anomalous regions within a sequence"""
    # Calculate RMSE for each time step across all channels
    rmse_per_step = np.sqrt(np.mean((reconstructed - original) ** 2, axis=1))
    
    threshold = threshold_factor * rmse_baseline
    anomaly_mask = np.zeros(len(original), dtype=bool)
    
    # Apply sliding window
    for t in range(len(rmse_per_step) - window_size + 1):
        window_rmse = rmse_per_step[t:t + window_size]
        if np.all(window_rmse > threshold):
            anomaly_mask[t:t + window_size] = True
    
    return anomaly_mask

def plot_anomaly_detection(original, reconstructed, anomaly_mask, sequence_idx):
    """Plot original vs reconstructed with anomalous regions highlighted"""
    num_channels = original.shape[1]
    
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2*num_channels))
    if num_channels == 1:
        axes = [axes]
    
    fig.suptitle(f'Anomaly Detection - Sequence {sequence_idx}')
    
    for ch in range(num_channels):
        axes[ch].plot(original[:, ch], 'b-', label='Input')
        axes[ch].plot(reconstructed[:, ch], 'g--', label='Reconstructed' if ch == 0 else '')
        
        # Highlight anomalous regions
        anomalous_signal = np.full_like(original[:, ch], np.nan)
        anomalous_signal[anomaly_mask] = original[anomaly_mask, ch]
        axes[ch].plot(anomalous_signal, 'r-', linewidth=3, label='Anomalous' if ch == 0 else '')
        
        axes[ch].set_ylabel(f'Channel {ch+1}')
        axes[ch].grid(True)
        
        if ch == 0:
            axes[ch].legend()
    
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()

def main():
    # Load or create data
    print("Creating synthetic waveform data...")
    data = create_synthetic_data()
    num_channels = data[0].shape[1]
    
    # Plot training data
    print("Plotting training data...")
    plot_training_data(data, num_channels)
    
    # Split data
    num_observations = len(data)
    split_idx = int(0.9 * num_observations)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Prepare data
    print("Preparing data...")
    train_data, train_seq_lengths = prepare_data(train_data)
    val_data, _ = prepare_data(val_data)
    
    # Create datasets and dataloaders
    train_dataset = WaveformDataset(train_data)
    val_dataset = WaveformDataset(val_data)
    
    # Custom collate function to handle variable length sequences
    def collate_fn(batch):
        # Find max length in batch
        max_len = max([item[0].size(0) for item in batch])
        
        # Pad sequences
        padded_inputs = []
        padded_targets = []
        
        for input_seq, target_seq in batch:
            seq_len = input_seq.size(0)
            if seq_len < max_len:
                pad_size = max_len - seq_len
                input_seq = torch.cat([input_seq, torch.zeros(pad_size, input_seq.size(1))], dim=0)
                target_seq = torch.cat([target_seq, torch.zeros(pad_size, target_seq.size(1))], dim=0)
            
            padded_inputs.append(input_seq)
            padded_targets.append(target_seq)
        
        return torch.stack(padded_inputs), torch.stack(padded_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    min_length = min(train_seq_lengths)
    model = Conv1DAutoencoder(num_channels)
    
    print(f"Model architecture:")
    print(model)
    
    # Train model
    print("Training autoencoder...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_losses, val_losses = train_autoencoder(model, train_loader, val_loader)
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate baseline RMSE
    print("Calculating baseline RMSE...")
    val_rmse = calculate_rmse(model, val_loader, device)
    
    plt.figure(figsize=(8, 5))
    plt.hist(val_rmse, bins=20, alpha=0.7)
    plt.xlabel('Root Mean Square Error (RMSE)')
    plt.ylabel('Frequency')
    plt.title('Representative Samples')
    plt.grid(True)
    plt.show()
    
    rmse_baseline = np.max(val_rmse)
    print(f"RMSE Baseline: {rmse_baseline:.4f}")
    
    # Inject anomalies
    print("Injecting anomalies...")
    anomalous_data, anomaly_indices = inject_anomalies(val_data, num_anomalous=20)
    
    # Test on anomalous data
    anomalous_dataset = WaveformDataset(anomalous_data)
    anomalous_loader = DataLoader(anomalous_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    anomalous_rmse = calculate_rmse(model, anomalous_loader, device)
    
    plt.figure(figsize=(8, 5))
    plt.hist(anomalous_rmse, bins=20, alpha=0.7, label='Data')
    plt.axvline(rmse_baseline, color='red', linestyle='--', label='Baseline RMSE')
    plt.xlabel('Root Mean Square Error (RMSE)')
    plt.ylabel('Frequency')
    plt.title('New Samples')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Find top anomalous sequences
    top_indices = np.argsort(anomalous_rmse)[::-1]
    print(f"Top 10 anomalous sequence indices: {top_indices[:10]}")
    
    # Analyze most anomalous sequence
    most_anomalous_idx = top_indices[0]
    original_seq = torch.FloatTensor(anomalous_data[most_anomalous_idx]).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        reconstructed_seq = model(original_seq).squeeze(0).cpu().numpy()
    
    original_seq = original_seq.squeeze(0).cpu().numpy()
    
    # Plot reconstruction
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2*num_channels))
    if num_channels == 1:
        axes = [axes]
    
    fig.suptitle(f'Sequence {most_anomalous_idx}')
    
    for ch in range(num_channels):
        min_len = min(len(original_seq), len(reconstructed_seq))
        axes[ch].plot(original_seq[:min_len, ch], 'b-', label='Original' if ch == 0 else '')
        axes[ch].plot(reconstructed_seq[:min_len, ch], 'r--', label='Reconstructed' if ch == 0 else '')
        axes[ch].set_ylabel(f'Channel {ch+1}')
        axes[ch].grid(True)
        
        if ch == 0:
            axes[ch].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Detect anomalous regions
    print("Detecting anomalous regions...")
    min_len = min(len(original_seq), len(reconstructed_seq))
    anomaly_mask = detect_anomalous_regions(
        original_seq[:min_len], 
        reconstructed_seq[:min_len], 
        rmse_baseline
    )
    
    plot_anomaly_detection(
        original_seq[:min_len], 
        reconstructed_seq[:min_len], 
        anomaly_mask, 
        most_anomalous_idx
    )

if __name__ == "__main__":
    main()
    
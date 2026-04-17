#!/usr/bin/env python3
"""
AI-Guided Inverse Design Framework for Recyclable Vitrimeric Polymers

This module implements:
1. Gaussian Process calibration for MD-simulated Tg predictions
2. Graph Variational Autoencoder for molecular generation
3. Inverse design pipeline for targeting specific Tg values
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional

# ML imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# RDKit for molecular fingerprints
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch for GVAE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, kl_divergence

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Paths
DATA_DIR = "data"
OUTPUTS_DIR = "outputs"
REPORT_IMAGES_DIR = "report/images"

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)


class MolecularFingerprintGenerator:
    """Generate molecular fingerprints from SMILES strings."""
    
    def __init__(self, fp_radius: int = 2, fp_length: int = 2048):
        self.fp_radius = fp_radius
        self.fp_length = fp_length
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES to RDKit mol object."""
        try:
            # Handle polymer SMILES with * markers
            smiles_clean = smiles.replace('*', 'C')  # Replace wildcards with carbon
            mol = Chem.MolFromSmiles(smiles_clean)
            return mol
        except:
            return None
    
    def get_fingerprint(self, smiles: str) -> np.ndarray:
        """Get Morgan fingerprint as numpy array."""
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return np.zeros(self.fp_length)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_length)
        return np.array(fp)
    
    def get_descriptors(self, smiles: str) -> np.ndarray:
        """Get molecular descriptors."""
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return np.zeros(10)
        
        desc_list = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            mol.GetRingInfo().NumRings() if mol.GetRingInfo() else 0
        ]
        return np.array(desc_list)
    
    def get_combined_features(self, smiles: str) -> np.ndarray:
        """Combine fingerprint and descriptors."""
        fp = self.get_fingerprint(smiles)
        desc = self.get_descriptors(smiles)
        return np.concatenate([fp, desc])


class GaussianProcessCalibrator:
    """Gaussian Process model for calibrating MD predictions to experimental Tg."""
    
    def __init__(self, kernel_params: Dict = None):
        if kernel_params is None:
            kernel_params = {
                'constant': 1.0,
                'length_scale': 1.0,
                'noise': 0.1
            }
        
        kernel = C(kernel_params['constant']) * RBF(kernel_params['length_scale']) + WhiteKernel(kernel_params['noise'])
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.fingerprint_gen = MolecularFingerprintGenerator()
        
    def prepare_features(self, df: pd.DataFrame, use_md: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets from dataframe."""
        print("Preparing molecular features...")
        
        X_list = []
        y_list = []
        
        for idx, row in df.iterrows():
            if 'smiles' in df.columns:
                smiles = row['smiles']
            elif 'acid' in df.columns:
                # For vitrimer data, combine acid and epoxide
                smiles = row['acid'] + '.' + row['epoxide']
            else:
                continue
            
            features = self.fingerprint_gen.get_combined_features(smiles)
            
            if use_md and 'tg_md' in df.columns:
                # Include MD prediction as additional feature
                md_tg = row['tg_md']
                features = np.concatenate([features, [md_tg]])
            
            X_list.append(features)
            
            if 'tg_exp' in df.columns:
                y_list.append(row['tg_exp'])
            elif 'tg' in df.columns:
                y_list.append(row['tg'])
        
        X = np.array(X_list)
        y = np.array(y_list) if y_list else None
        
        return X, y
    
    def fit(self, df: pd.DataFrame) -> Dict:
        """Fit GP model on calibration data."""
        X, y = self.prepare_features(df, use_md=True)
        
        # Remove any samples with NaN or inf
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Training on {len(y)} samples...")
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Fit GP
        self.gp.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred, y_std = self.gp.predict(X_val_scaled, return_std=True)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred),
            'mean_uncertainty': np.mean(y_std),
            'n_train': len(y_train),
            'n_val': len(y_val)
        }
        
        print(f"GP Calibration Results: RMSE={metrics['rmse']:.2f}K, R²={metrics['r2']:.3f}")
        
        # Store training data for later prediction
        self.X_train = X_train_scaled
        self.y_train = y_train
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict calibrated Tg values."""
        X_raw, _ = self.prepare_features(df, use_md=True)        
        # Handle feature dimension mismatch by padding or truncating
        n_features_train = self.X_train.shape[1]
        if X_raw.shape[1] < n_features_train:
            # Pad with zeros
            padding = np.zeros((X_raw.shape[0], n_features_train - X_raw.shape[1]))
            X_raw = np.concatenate([X_raw, padding], axis=1)
        elif X_raw.shape[1] > n_features_train:
            # Truncate
            X_raw = X_raw[:, :n_features_train]
        
        X_scaled = self.scaler.transform(X_raw)
        y_pred, y_std = self.gp.predict(X_scaled, return_std=True)
        return y_pred, y_std
    
    def save_results(self, metrics: Dict, filepath: str = None):
        """Save GP results to file."""
        if filepath is None:
            filepath = os.path.join(OUTPUTS_DIR, "gp_calibration_results.json")
        
        result = {
            'metrics': metrics,
            'kernel': str(self.gp.kernel_)
        }
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"GP results saved to {filepath}")
        return filepath


class VitrimerDataset(Dataset):
    """PyTorch dataset for vitrimer SMILES."""
    
    def __init__(self, df: pd.DataFrame, max_length: int = 200):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.vocab = self._build_vocab()
        # Build char2idx from vocab_list which has special tokens
        self.char2idx = {c: i for i, c in enumerate(self.vocab_list)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab_list)}
        
    def _build_vocab(self) -> str:
        """Build character vocabulary from SMILES."""
        all_chars = set()
        for col in ['acid', 'epoxide']:
            if col in self.df.columns:
                for smiles in self.df[col]:
                    all_chars.update(str(smiles))
        
        # Build vocabulary - sort characters to ensure consistent ordering
        sorted_chars = sorted(list(all_chars))
        special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        
        # Create vocab as list first
        vocab_list = sorted_chars + special_tokens        
        self.vocab_list = vocab_list  # Store for reference
        vocab = ''.join(vocab_list)
        
        return vocab
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Combine acid and epoxide with separator
        smiles = str(row['acid']) + '.' + str(row['epoxide'])
        
        # Encode to indices
        encoded = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in smiles]
        
        # Pad/truncate
        if len(encoded) < self.max_length:
            encoded += [self.char2idx['<PAD>']] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
        
        return torch.tensor(encoded, dtype=torch.long)


class GrammarVAE(nn.Module):
    """Variational Autoencoder for SMILES generation."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, 
                 latent_dim: int = 64, max_length: int = 200):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size-1)
        self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder_rnn = nn.GRU(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        embedded = self.embedding(x)
        _, hidden = self.encoder_rnn(embedded)
        
        # Concatenate bidirectional hidden states
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=-1)
        
        mu = self.fc_mu(hidden_concat)
        logvar = self.fc_logvar(hidden_concat)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """Decode latent vector to output sequence."""
        batch_size = z.size(0)
        
        # Start token
        decoder_input = torch.full((batch_size, 1), self.vocab_size - 3, dtype=torch.long)  # <START>
        
        outputs = []
        hidden = z.unsqueeze(0)
        
        for t in range(self.max_length):
            embedded = self.embedding(decoder_input)
            decoder_input_cat = torch.cat([embedded, z.unsqueeze(1)], dim=-1)
            out, hidden = self.decoder_rnn(decoder_input_cat, hidden)
            logits = self.fc_out(out.squeeze(1))
            outputs.append(logits)
            
            # Teacher forcing during training
            if target is not None:
                decoder_input = target[:, t:t+1]
            else:
                # Greedy decoding
                _, top_idx = logits.topk(1, dim=-1)
                decoder_input = top_idx
        
        return torch.stack(outputs, dim=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with VAE loss computation."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, x[:, 1:])  # Shift target for teacher forcing
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Reconstruction loss
        recon_loss = F.cross_entropy(
            output.view(-1, self.vocab_size),
            x[:, 1:].reshape(-1),
            ignore_index=self.vocab_size - 1  # Ignore padding
        )
        
        return output, recon_loss, kl_loss
    
    def generate(self, n_samples: int = 10, temperature: float = 1.0) -> List[str]:
        """Generate new SMILES strings."""
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(n_samples, self.latent_dim)
            
            generated = []
            for i in range(n_samples):
                zi = z[i:i+1]
                decoder_input = torch.full((1, 1), self.vocab_size - 3, dtype=torch.long)
                hidden = zi.unsqueeze(0)
                
                smiles_chars = []
                for t in range(self.max_length):
                    embedded = self.embedding(decoder_input)
                    decoder_input_cat = torch.cat([embedded, zi.unsqueeze(1)], dim=-1)
                    out, hidden = self.decoder_rnn(decoder_input_cat, hidden)
                    logits = self.fc_out(out.squeeze(1)) / temperature
                    
                    probs = F.softmax(logits, dim=-1)
                    top_idx = torch.multinomial(probs, 1)
                    
                    char_idx = top_idx.item()
                    if char_idx == self.vocab_size - 2:  # <END>
                        break
                    if char_idx not in [self.vocab_size - 1, self.vocab_size - 3]:  # Not PAD or START
                        smiles_chars.append(self.idx2char[char_idx])
                    
                    decoder_input = top_idx
                
                generated.append(''.join(smiles_chars))
        
        return generated


class VitrimerGVAE:
    """Graph VAE wrapper for vitrimer generation."""
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self.dataset = None
        self.dataloader = None
        
    def prepare_data(self, df: pd.DataFrame, max_length: int = 200, batch_size: int = 64):
        """Prepare dataset and dataloader."""
        self.dataset = VitrimerDataset(df, max_length=max_length)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.max_length = max_length
        print(f"Prepared dataset with {len(self.dataset)} samples, vocab size: {len(self.dataset.vocab)}")
        
    def train(self, epochs: int = 50, lr: float = 0.001) -> Dict:
        """Train the GVAE model."""
        if self.dataset is None:
            raise ValueError("Must call prepare_data first")
        
        vocab_size = len(self.dataset.vocab)
        self.model = GrammarVAE(
            vocab_size=vocab_size,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            max_length=self.max_length
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        history = {'recon_loss': [], 'kl_loss': [], 'total_loss': []}
        
        print(f"Training GVAE for {epochs} epochs...")
        for epoch in range(epochs):
            self.model.train()
            total_recon, total_kl, total_loss = 0, 0, 0
            n_batches = 0
            
            for batch in self.dataloader:
                optimizer.zero_grad()
                output, recon_loss, kl_loss = self.model(batch)
                loss = recon_loss + 0.01 * kl_loss  # KL weight
                
                loss.backward()
                optimizer.step()
                
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                total_loss += loss.item()
                n_batches += 1
            
            avg_recon = total_recon / n_batches
            avg_kl = total_kl / n_batches
            avg_loss = total_loss / n_batches
            
            history['recon_loss'].append(avg_recon)
            history['kl_loss'].append(avg_kl)
            history['total_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.3f}, Recon={avg_recon:.3f}, KL={avg_kl:.3f}")
        
        print(f"Training complete. Final loss: {history['total_loss'][-1]:.3f}")
        return history
    
    def generate_candidates(self, n_samples: int = 100, target_tg: float = None, 
                           gp_calibrator: GaussianProcessCalibrator = None) -> pd.DataFrame:
        """Generate novel vitrimer candidates."""
        if self.model is None:
            raise ValueError("Must train model first")
        
        self.model.eval()
        generated_smiles = self.model.generate(n_samples)
        
        # Parse generated SMILES into acid/epoxide pairs
        candidates = []
        for smi in generated_smiles:
            if '.' in smi:
                parts = smi.split('.')
                if len(parts) >= 2:
                    candidates.append({
                        'acid': parts[0],
                        'epoxide': '.'.join(parts[1:]),
                        'generated_smiles': smi
                    })
        
        df_candidates = pd.DataFrame(candidates)
        
        # Predict Tg if GP calibrator provided
        if gp_calibrator is not None and len(df_candidates) > 0:
            pred_tg, pred_std = gp_calibrator.predict(df_candidates)
            df_candidates['predicted_tg'] = pred_tg
            df_candidates['prediction_std'] = pred_std
        
        return df_candidates
    
    def save_results(self, history: Dict, filepath: str = None):
        """Save training results."""
        if filepath is None:
            filepath = os.path.join(OUTPUTS_DIR, "gvae_training_results.json")
        
        # Convert lists to serializable format
        result = {
            'training_history': {
                'recon_loss': [float(x) for x in history['recon_loss']],
                'kl_loss': [float(x) for x in history['kl_loss']],
                'total_loss': [float(x) for x in history['total_loss']]
            },
            'final_loss': float(history['total_loss'][-1]),
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim
        }
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"GVAE results saved to {filepath}")
        return filepath


def plot_data_overview(calib_df: pd.DataFrame, vitrimer_df: pd.DataFrame, save_path: str = None):
    """Create data overview plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Tg distribution (calibration data)
    ax = axes[0, 0]
    ax.hist(calib_df['tg_exp'], bins=30, alpha=0.7, label='Experimental', edgecolor='black')
    ax.hist(calib_df['tg_md'], bins=30, alpha=0.7, label='MD Simulated', edgecolor='black')
    ax.set_xlabel('Tg (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Tg Distribution - Calibration Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tg distribution (vitrimer data)
    ax = axes[0, 1]
    ax.hist(vitrimer_df['tg'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Tg (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Tg Distribution - Vitrimer MD Data')
    ax.axvline(vitrimer_df['tg'].mean(), color='red', linestyle='--', label=f"Mean: {vitrimer_df['tg'].mean():.1f}K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MD vs Experimental parity (calibration data)
    ax = axes[1, 0]
    ax.scatter(calib_df['tg_md'], calib_df['tg_exp'], alpha=0.5, s=20)
    min_tg = min(calib_df['tg_md'].min(), calib_df['tg_exp'].min())
    max_tg = max(calib_df['tg_md'].max(), calib_df['tg_exp'].max())
    ax.plot([min_tg, max_tg], [min_tg, max_tg], 'r--', label='Perfect Agreement')
    ax.set_xlabel('MD Simulated Tg (K)')
    ax.set_ylabel('Experimental Tg (K)')
    ax.set_title('MD vs Experimental Tg - Calibration Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[1, 1]
    errors = calib_df['tg_md'] - calib_df['tg_exp']
    ax.hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel('MD Error (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('MD Simulation Error Distribution')
    ax.axvline(errors.mean(), color='red', linestyle='--', label=f"Mean: {errors.mean():.1f}K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Data overview plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_gp_calibration(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, 
                        save_path: str = None):
    """Plot GP calibration results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Parity plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=30)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('GP Calibrated Tg (K)')
    ax.set_title(f'GP Calibration Parity Plot\nR² = {r2_score(y_true, y_pred):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals
    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, alpha=0.5, s=30)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('Residual (K)')
    ax.set_title(f'Residuals Analysis\nMAE = {mean_absolute_error(y_true, y_pred):.1f}K')
    ax.grid(True, alpha=0.3)
    
    # Uncertainty calibration
    ax = axes[2]
    standardized_residuals = residuals / (y_std + 1e-6)
    ax.hist(standardized_residuals, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Standardized Residual')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Uncertainty Calibration\nMean |Std Res| = {np.abs(standardized_residuals).mean():.2f}')
    ax.axvline(0, color='red', linestyle='--')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"GP calibration plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_gvae_training(history: Dict, save_path: str = None):
    """Plot GVAE training history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['total_loss']) + 1)
    ax.plot(epochs, history['total_loss'], label='Total Loss', linewidth=2)
    ax.plot(epochs, history['recon_loss'], label='Reconstruction Loss', linewidth=2)
    ax.plot(epochs, history['kl_loss'], label='KL Divergence', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('GVAE Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"GVAE training plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_generated_candidates(df_generated: pd.DataFrame, df_original: pd.DataFrame, 
                              save_path: str = None):
    """Plot generated candidates vs original distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Tg distribution comparison
    ax = axes[0]
    ax.hist(df_original['tg'], bins=50, alpha=0.5, label='Original Vitrimer', density=True)
    if 'predicted_tg' in df_generated.columns:
        ax.hist(df_generated['predicted_tg'].dropna(), bins=50, alpha=0.5, 
                label='Generated Candidates', density=True)
    ax.set_xlabel('Tg (K)')
    ax.set_ylabel('Density')
    ax.set_title('Tg Distribution: Original vs Generated')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Uncertainty distribution
    ax = axes[1]
    if 'prediction_std' in df_generated.columns:
        ax.hist(df_generated['prediction_std'].dropna(), bins=30, alpha=0.7, 
                color='steelblue', edgecolor='black')
        ax.set_xlabel('Prediction Uncertainty (K)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Prediction Uncertainty Distribution\nMean: {df_generated["prediction_std"].mean():.1f}K')
        ax.axvline(df_generated['prediction_std'].mean(), color='red', linestyle='--')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Generated candidates plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main execution pipeline."""
    print("=" * 60)
    print("AI-Guided Inverse Design Framework for Vitrimeric Polymers")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    calib_df = pd.read_csv(os.path.join(DATA_DIR, "tg_calibration.csv"))
    vitrimer_df = pd.read_csv(os.path.join(DATA_DIR, "tg_vitrimer_MD.csv"))
    
    print(f"Calibration data: {len(calib_df)} samples")
    print(f"Vitrimer MD data: {len(vitrimer_df)} samples")
    
    # Data overview
    print("\n[2] Creating data overview plots...")
    plot_data_overview(
        calib_df, vitrimer_df,
        save_path=os.path.join(REPORT_IMAGES_DIR, "data_overview.png")
    )
    
    # GP Calibration
    print("\n[3] Training Gaussian Process Calibrator...")
    gp_calibrator = GaussianProcessCalibrator()
    gp_metrics = gp_calibrator.fit(calib_df)
    gp_calibrator.save_results(gp_metrics)
    
    # Validate GP on vitrimer data
    print("\n[4] Validating GP on vitrimer data...")
    vitrimer_subset = vitrimer_df.sample(min(500, len(vitrimer_df)), random_state=42)
    y_pred, y_std = gp_calibrator.predict(vitrimer_subset)
    
    # Create validation dataframe for plotting
    val_df = vitrimer_subset.copy()
    val_df['gp_pred'] = y_pred
    val_df['gp_std'] = y_std
    
    plot_gp_calibration(
        vitrimer_subset['tg'].values, y_pred, y_std,
        save_path=os.path.join(REPORT_IMAGES_DIR, "gp_calibration_results.png")
    )
    
    # GVAE Training
    print("\n[5] Training Graph Variational Autoencoder...")
    gvae = VitrimerGVAE(latent_dim=64, hidden_dim=256)
    gvae.prepare_data(vitrimer_df[['acid', 'epoxide']], max_length=150, batch_size=64)
    gvae_history = gvae.train(epochs=50, lr=0.001)
    gvae.save_results(gvae_history)
    
    plot_gvae_training(
        gvae_history,
        save_path=os.path.join(REPORT_IMAGES_DIR, "gvae_training_history.png")
    )
    
    # Generate candidates
    print("\n[6] Generating novel vitrimer candidates...")
    n_generate = 100
    generated_df = gvae.generate_candidates(n_samples=n_generate, gp_calibrator=gp_calibrator)
    
    print(f"Generated {len(generated_df)} candidate molecules")
    
    if len(generated_df) > 0:
        generated_df.to_csv(os.path.join(OUTPUTS_DIR, "generated_candidates.csv"), index=False)
        print(f"Candidates saved to outputs/generated_candidates.csv")
        
        plot_generated_candidates(
            generated_df, vitrimer_df,
            save_path=os.path.join(REPORT_IMAGES_DIR, "generated_candidates_analysis.png")
        )
    
    # Save summary statistics
    summary = {
        'data_summary': {
            'calibration_samples': len(calib_df),
            'vitrimer_samples': len(vitrimer_df),
            'tg_range_calib': [float(calib_df['tg_exp'].min()), float(calib_df['tg_exp'].max())],
            'tg_range_vitrimer': [float(vitrimer_df['tg'].min()), float(vitrimer_df['tg'].max())]
        },
        'gp_calibration': gp_metrics,
        'gvae_training': {
            'final_loss': float(gvae_history['total_loss'][-1]),
            'epochs': len(gvae_history['total_loss'])
        },
        'generation': {
            'n_candidates': len(generated_df),
            'mean_predicted_tg': float(generated_df['predicted_tg'].mean()) if len(generated_df) > 0 else None,
            'std_predicted_tg': float(generated_df['predicted_tg'].std()) if len(generated_df) > 0 else None
        }
    }
    
    with open(os.path.join(OUTPUTS_DIR, "summary_results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUTS_DIR}/")
    print(f"Figures saved to: {REPORT_IMAGES_DIR}/")
    
    return summary


if __name__ == "__main__":
    main()

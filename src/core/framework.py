"""
Uncertainty-Aware Safety-Preserving Deduplication Framework

Q1-Ready Implementation with:
- Bayesian Uncertainty Quantification
- Causal Safety Feature Discovery  
- Multi-Level Adaptive Deduplication
- Learned Similarity Embeddings
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

# FAISS for efficient similarity search (matches paper specification)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using fallback similarity search")

logger = logging.getLogger(__name__)

# Import required components
try:
    from .framework_components import (
        EnhancedBayesianEnsemble,
        ImprovedCausalSafetyDiscovery,
        ImprovedSiameseNetwork
    )
except ImportError:
    # Fallback: define minimal versions if not available
    from improved_uncertainty_aware_framework import (
        EnhancedBayesianEnsemble,
        ImprovedCausalSafetyDiscovery,
        ImprovedSiameseNetwork
    )


class UncertaintyAwareFramework:
    """
    Q1-Ready Uncertainty-Aware Safety-Preserving Deduplication Framework
    
    Novel Contributions:
    1. Multi-level adaptive deduplication using uncertainty gating
    2. Bayesian ensemble for uncertainty quantification
    3. Causal discovery for safety-critical feature identification
    4. Learned similarity embeddings via Siamese networks
    """
    
    def __init__(self, config):
        """
        Initialize framework with configuration
        
        Args:
            config: Dictionary with framework configuration
        """
        logger.info("=" * 80)
        logger.info("UNCERTAINTY-AWARE SAFETY-PRESERVING FRAMEWORK")
        logger.info("=" * 80)
        
        self.config = config
        self.input_dim = config['model']['input_dim']
        self.embedding_dim = config['model']['embedding_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.num_ensemble_models = config['model']['num_ensemble_models']
        
        # Device configuration
        self.device = torch.device(
            'cuda' if (config['device']['use_gpu'] and torch.cuda.is_available()) 
            else 'cpu'
        )
        
        # Initialize components
        # bayesian_dropout_rate is plumbed through from config so the
        # BNN-architecture ablation  can vary it without
        # editing source.
        self._bayesian_dropout_rate = float(
            config['model'].get('bayesian_dropout_rate', 0.3)
        )
        self.bayesian_ensemble = EnhancedBayesianEnsemble(
            self.input_dim,
            hidden_dim=self.hidden_dim,
            num_models=self.num_ensemble_models,
            dropout_rate=self._bayesian_dropout_rate,
        ).to(self.device)
        
        self.causal_discovery = ImprovedCausalSafetyDiscovery(
            alpha=config.get('causal', {}).get('alpha', 0.05)
        )
        
        self.siamese_net = ImprovedSiameseNetwork(
            self.input_dim, 
            self.embedding_dim
        ).to(self.device)
        
        # Optimizers
        self.optimizer_ensemble = optim.AdamW(
            self.bayesian_ensemble.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.optimizer_siamese = optim.AdamW(
            self.siamese_net.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Loss functions
        self.criterion_ensemble = nn.BCELoss()
        self.criterion_siamese = nn.TripletMarginLoss(margin=0.5)  # Matches paper specification
        
        # Scaler for data preprocessing
        self.scaler = StandardScaler()
        
        # State
        self.trained = False
        self.siamese_trained = False
        
        logger.info(f"Framework initialized on {self.device}")
        logger.info(f"Input dim: {self.input_dim}, Embedding dim: {self.embedding_dim}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the framework (Bayesian ensemble for uncertainty)
        
        Args:
            X_train: Training features
            y_train: Training labels (safety-critical)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING UNCERTAINTY-AWARE FRAMEWORK")
        logger.info("=" * 80)
        
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        # drop_last=True avoids a "BatchNorm with batch_size=1" crash that can
        # happen when len(dataset) % batch_size == 1 (the BNN architecture
        # uses BatchNorm1d which requires batch_size > 1). Observed on SWaT
        # with certain seeds (May 2026). Dropping the final partial batch
        # has negligible effect on training (one mini-batch per epoch lost).
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            drop_last=True,
        )
        
        epochs = self.config['training']['bayesian_epochs']
        patience = self.config['training']['early_stopping_patience']
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        for epoch in range(epochs):
            # Training
            self.bayesian_ensemble.train()
            train_loss = 0
            
            for batch_X, batch_y in dataloader:
                self.optimizer_ensemble.zero_grad()
                
                losses = []
                for model in self.bayesian_ensemble.models:
                    pred = model(batch_X)
                    loss = self.criterion_ensemble(pred, batch_y)
                    losses.append(loss)
                
                avg_loss = torch.mean(torch.stack(losses))
                avg_loss.backward()
                self.optimizer_ensemble.step()
                train_loss += avg_loss.item()
            
            train_loss /= len(dataloader)
            
            # Validation
            if X_val is not None and len(X_val) > 0:
                self.bayesian_ensemble.eval()
                with torch.no_grad():
                    val_losses = []
                    for model in self.bayesian_ensemble.models:
                        pred = model(X_val_tensor)
                        loss = self.criterion_ensemble(pred, y_val_tensor)
                        val_losses.append(loss.item())
                    val_loss = np.mean(val_losses)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}/{epochs} - Train: {train_loss:.4f}")
        
        self.trained = True
        logger.info("Framework training completed")
    
    def predict_with_uncertainty(self, X):
        """
        Predict uncertainty scores using Bayesian ensemble
        
        Args:
            X: Input features
            
        Returns:
            uncertainty: Uncertainty scores (can be tuple of (mean, uncertainty) or array)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get uncertainty using MC dropout
        self.bayesian_ensemble.train()  # Keep dropout active
        mc_samples = self.config['uncertainty']['mc_samples']
        
        predictions_list = []
        with torch.no_grad():
            for _ in range(mc_samples):
                preds = self.bayesian_ensemble(X_tensor)
                predictions_list.append(preds)
        
        all_predictions = torch.cat(predictions_list, dim=0)
        mean_pred = all_predictions.mean(dim=0)
        uncertainty = all_predictions.var(dim=0)
        
        # Return uncertainty scores (normalized variance)
        return uncertainty.cpu().numpy().flatten()
    
    def get_embeddings(self, X, use_siamese=True):
        """
        Get embeddings for similarity computation.
        
        Args:
            X: Input features (n_samples, n_features)
            use_siamese: If True, use learned Siamese embeddings. If False, return normalized raw features.
            
        Returns:
            embeddings: Learned embeddings or normalized raw features (n_samples, embedding_dim or n_features)
        """
        if not use_siamese:
            # Return normalized raw features instead of learned embeddings
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            return X_norm
        
        # Use Siamese network for learned embeddings
        self.siamese_net.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # Use forward_one method if available, else use forward
            if hasattr(self.siamese_net, 'forward_one'):
                embeddings = self.siamese_net.forward_one(X_tensor)
            else:
                # Fallback: create dummy pairs
                embeddings = self.siamese_net(X_tensor, X_tensor)[0]
        
        return embeddings.cpu().numpy()
    
    def find_duplicates_safety_aware(self, X, similarity_threshold=0.85, safety_labels=None):
        """
        Find duplicates with safety-critical preservation
        
        NEVER removes a sample if it's safety-critical, unless it's nearly identical
        to another safety-critical sample (>95% similarity).
        
        Args:
            X: Input data
            similarity_threshold: Standard threshold for non-critical data
            safety_labels: Binary array (1=safety-critical)
        
        Returns:
            List of duplicate pairs that preserve safety
        """
        embeddings = self.get_embeddings(X)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(embeddings_norm, embeddings_norm.T)
        
        n = len(similarities)
        safe_duplicates = []
        
        for i in range(n):
            for j in range(i + 1, min(i + 100, n)):
                i_critical = safety_labels[i] == 1 if safety_labels is not None else False
                j_critical = safety_labels[j] == 1 if safety_labels is not None else False
                sim = similarities[i, j]
                
                # Both non-critical: use normal threshold
                if not i_critical and not j_critical:
                    if sim >= similarity_threshold:
                        safe_duplicates.append((i, j))
                
                # One or both critical: use stricter threshold (95% similarity)
                elif i_critical or j_critical:
                    if sim >= 0.95:  # Stricter for safety-critical
                        # Always keep the critical sample
                        if i_critical:
                            safe_duplicates.append((i, j))  # Remove j, keep i
                        else:
                            safe_duplicates.append((j, i))  # Remove i, keep j
        
        return safe_duplicates
    
    def find_duplicates_multi_level(self, df, X_scaled, uncertainty_scores, 
                                    low_threshold=0.3, high_threshold=0.5,
                                    similarity_threshold=0.85,
                                    use_siamese=True,
                                    return_details=False):
        """
        CRITICAL FIX: Multi-level adaptive deduplication using uncertainty gating
        
        This is the NOVEL CONTRIBUTION that main_experiments.py must use.
        Uses uncertainty from Bayesian ensemble to gate deduplication:
        - Level 1 (low uncertainty < 0.3): Safe to deduplicate with standard threshold
        - Level 2 (high uncertainty >= 0.5): Preserve (do not deduplicate)
        - Level 3 (causal safety): Always preserve
        
        Args:
            df: DataFrame with original features (for causal feature checking)
            X_scaled: Scaled input features
            uncertainty_scores: Array of uncertainty scores from Bayesian ensemble
            low_threshold: Threshold for low uncertainty (default 0.3)
            high_threshold: Threshold for high uncertainty (default 0.5)
            similarity_threshold: Similarity threshold for Level 1 deduplication
            use_siamese: If True, use learned Siamese embeddings. If False, use raw feature similarity.
        
        Returns:
            duplicates: List of duplicate pairs (i, j)
            preserved_count: Number of samples preserved due to uncertainty/safety
        """
        # Get causal features (safety-critical features discovered earlier)
        causal_features = getattr(self.causal_discovery, 'causal_features', [])
        if hasattr(self.causal_discovery, 'safety_critical_features'):
            causal_features = self.causal_discovery.safety_critical_features
        
        # Create causal mask (Level 3: absolute preserve)
        # CRITICAL FIX: Use training statistics (stored in causal_discovery) instead of test set statistics
        causal_mask = np.zeros(len(df), dtype=bool)
        if causal_features and len(causal_features) > 0:
            # Get training statistics from causal discovery component
            training_stats = getattr(self.causal_discovery, 'training_stats', {})
            
            for idx in range(len(df)):
                for feat in causal_features:
                    if feat in df.columns:
                        value = df.iloc[idx][feat]
                        
                        # CRITICAL: Use training statistics, not test set statistics
                        if feat in training_stats:
                            train_mean = training_stats[feat]['mean']
                            train_std = training_stats[feat]['std']
                            z_score = np.abs((value - train_mean) / train_std)
                        else:
                            # Fallback: use test set stats if training stats not available (shouldn't happen)
                            logger.warning(f"Training stats not found for {feat}, using test set stats (may cause leakage)")
                            col_values = df[feat].values
                            z_score = np.abs((value - col_values.mean()) / (col_values.std() + 1e-10))
                        
                        if z_score > 3.0:  # Use 3-sigma threshold (consistent with paper)
                            causal_mask[idx] = True
                            break
        
        # Multi-level categorization
        level_1_mask = (uncertainty_scores < low_threshold) & (~causal_mask)
        level_2_mask = uncertainty_scores >= high_threshold
        level_3_mask = causal_mask
        
        logger.info(f"Multi-level categorization:")
        logger.info(f"  Level 1 (low uncertainty, safe to dedup): {level_1_mask.sum()} samples")
        logger.info(f"  Level 2 (high uncertainty, preserve): {level_2_mask.sum()} samples")
        logger.info(f"  Level 3 (causal safety, absolute preserve): {level_3_mask.sum()} samples")
        
        # Generate embeddings (respect use_siamese flag)
        if use_siamese:
            logger.info("  Using Siamese network for similarity computation")
        else:
            logger.info("  Using raw features for similarity computation (Siamese disabled)")
        embeddings = self.get_embeddings(X_scaled, use_siamese=use_siamese)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Find duplicates ONLY in Level 1 (low uncertainty, non-causal)
        level_1_indices = np.where(level_1_mask)[0]
        duplicates = []
        n = len(embeddings_norm)
        
        # Use FAISS for efficient similarity search (matches paper specification)
        if FAISS_AVAILABLE and len(level_1_indices) > 1:
            # Build FAISS HNSW index (matches paper: M=32, efConstruction=200, efSearch=100)
            embedding_dim = embeddings_norm.shape[1]
            index = faiss.IndexHNSWFlat(embedding_dim, 32)  # M=32
            index.hnsw.efConstruction = 200  # efConstruction=200
            index.hnsw.efSearch = 100  # efSearch=100
            
            # Add Level 1 embeddings to index
            level_1_embeddings = embeddings_norm[level_1_indices].astype('float32')
            index.add(level_1_embeddings)
            
            # Search for nearest neighbors
            for idx, i in enumerate(level_1_indices):
                query = embeddings_norm[i:i+1].astype('float32')
                k = min(100, len(level_1_indices))  # Search top-k neighbors
                distances, neighbors = index.search(query, k)
                
                for dist, neighbor_idx in zip(distances[0], neighbors[0]):
                    if neighbor_idx >= 0 and neighbor_idx < len(level_1_indices):
                        j = level_1_indices[neighbor_idx]
                        if j > i:  # Avoid duplicates (i, j) and (j, i)
                            # Convert L2 distance to cosine similarity
                            # For normalized vectors: cosine_sim = 1 - (L2_dist^2 / 2)
                            similarity = 1.0 - (dist * dist / 2.0)
                            if similarity >= similarity_threshold:
                                duplicates.append((i, j))
        else:
            # Fallback: nested loop (if FAISS not available)
            logger.warning("Using fallback similarity search (FAISS not available)")
            for i in level_1_indices:
                for j in range(i + 1, min(i + 100, n)):
                    if j in level_1_indices:  # Both must be Level 1
                        sim = np.dot(embeddings_norm[i], embeddings_norm[j])
                        if sim >= similarity_threshold:
                            duplicates.append((i, j))
        
        # Preserved count = Level 2 + Level 3
        preserved_count = level_2_mask.sum() + level_3_mask.sum()
        
        logger.info(f"Deduplication complete: {len(duplicates)} duplicates found, {preserved_count} samples preserved")
        
        if return_details:
            details = {
                'level_1_mask': level_1_mask,
                'level_2_mask': level_2_mask,
                'level_3_mask': level_3_mask,
                'causal_mask': causal_mask,
                'causal_features': causal_features,
                'uncertainty_scores': uncertainty_scores
            }
            return duplicates, preserved_count, details

        return duplicates, preserved_count


# Export required components for __init__.py
# These will be imported from improved_uncertainty_aware_framework.py if available
try:
    from improved_uncertainty_aware_framework import (
        EnhancedBayesianEnsemble,
        ImprovedCausalSafetyDiscovery,
        ImprovedSiameseNetwork
    )
except ImportError:
    # If not available, create placeholder classes
    logger.warning("Framework components not found, using placeholders")
    
    class EnhancedBayesianEnsemble(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.models = nn.ModuleList()
    
    class ImprovedCausalSafetyDiscovery:
        def __init__(self, *args, **kwargs):
            self.safety_critical_features = set()
        
        def discover_causal_structure(self, df, labels):
            return set()
    
    class ImprovedSiameseNetwork(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        
        def forward_one(self, x):
            return x

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LiteralDataset(Dataset):
    def __init__(self, dataset_dir, ent_idx, normalization="z-norm", sampling_ratio=None, selected_attributes=None, 
                 label_perturbation=None, perturbation_ratio=0.1, perturbation_noise_std=0.1, random_seed=42):
        """
        Initialize LiteralDataset with optional label perturbation.
        
        Args:
            dataset_dir: Path to dataset directory
            ent_idx: Entity index mapping
            normalization: Normalization type ("z-norm", "min-max", etc.)
            sampling_ratio: Fraction of training data to use
            selected_attributes: List of specific attributes to use
            label_perturbation: Type of label perturbation ("gaussian", "uniform", "label_flip", None)
            perturbation_ratio: Fraction of labels to perturb (0.0 to 1.0)
            perturbation_noise_std: Standard deviation for gaussian noise or range for uniform noise
            random_seed: Random seed for reproducible perturbations
        """
        self.dataset_dir = os.path.join(dataset_dir, "literals")
        self.normalization_type = normalization
        self.normalization_params = {}
        self.sampling_ratio = sampling_ratio 
        self.selected_attributes = selected_attributes  # List of attribute names to filter
        
        # Label perturbation parameters
        self.label_perturbation = label_perturbation
        self.perturbation_ratio = perturbation_ratio
        self.perturbation_noise_std = perturbation_noise_std
        self.random_seed = random_seed
        self.perturbed_indices = set()  # Track which labels were perturbed

        self.file_paths = {
            s: os.path.join(self.dataset_dir, f"{s}.txt")
            for s in ("train", "val", "test")
        }

        self.entity_to_idx = (
            ent_idx
            if isinstance(ent_idx, dict)
            else {v: i for i, v in ent_idx["entity"].items()}
        )
        self.num_entities = len(self.entity_to_idx)

        self._load_data()

    def _apply_label_perturbation(self, df):
        """
        Apply random perturbations to normalized training labels.
        
        Args:
            df: DataFrame with training data (must have 'tail_norm' column)
            
        Returns:
            DataFrame with perturbed normalized labels
        """
        if self.label_perturbation is None or self.perturbation_ratio <= 0:
            print("No label perturbation applied.")
            return df
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Create a copy to avoid modifying original data
        df_perturbed = df.copy()
        
        # Determine which labels to perturb
        num_samples = len(df_perturbed)
        num_to_perturb = int(num_samples * self.perturbation_ratio)
        
        if num_to_perturb == 0:
            print(f"Perturbation ratio {self.perturbation_ratio} too small, no labels perturbed.")
            return df_perturbed
        
        # Randomly select indices to perturb
        perturb_indices = np.random.choice(num_samples, size=num_to_perturb, replace=False)
        self.perturbed_indices = set(perturb_indices)
        
        # Work with normalized values
        original_normalized = df_perturbed.loc[perturb_indices, 'tail_norm'].values
        
        if self.label_perturbation == "gaussian":
            # Add Gaussian noise to normalized values
            noise = np.random.normal(0, self.perturbation_noise_std, size=len(perturb_indices))
            perturbed_normalized = original_normalized + noise
            
        elif self.label_perturbation == "uniform":
            # Add uniform noise to normalized values
            noise = np.random.uniform(-self.perturbation_noise_std, self.perturbation_noise_std, 
                                    size=len(perturb_indices))
            perturbed_normalized = original_normalized + noise
            
        elif self.label_perturbation == "label_flip":
            # Flip normalized labels within each relation
            perturbed_normalized = []
            for idx in perturb_indices:
                relation = df_perturbed.loc[idx, 'relation']
                relation_data = df_perturbed[df_perturbed['relation'] == relation]['tail_norm']
                
                # Sample a random normalized value from the same relation
                if len(relation_data) > 1:
                    other_values = relation_data[relation_data.index != idx].values
                    if len(other_values) > 0:
                        perturbed_val = np.random.choice(other_values)
                    else:
                        perturbed_val = original_normalized[perturb_indices.tolist().index(idx)]
                else:
                    # If only one value for this relation, add small noise
                    perturbed_val = original_normalized[perturb_indices.tolist().index(idx)] + \
                                  np.random.normal(0, 0.01)
                perturbed_normalized.append(perturbed_val)
            perturbed_normalized = np.array(perturbed_normalized)
            
        elif self.label_perturbation == "scaled_noise":
            # Scale normalized values by noise factors
            noise_factors = np.random.normal(1.0, self.perturbation_noise_std, size=len(perturb_indices))
            perturbed_normalized = original_normalized * noise_factors
            
        elif self.label_perturbation == "dropout":
            # Set some normalized values to zero (simulating missing data)
            perturbed_normalized = original_normalized.copy()
            dropout_mask = np.random.random(len(perturb_indices)) < self.perturbation_noise_std
            perturbed_normalized[dropout_mask] = 0.0
            
        elif self.label_perturbation == "quantization":
            # Quantize normalized values (simulating reduced precision)
            # perturbation_noise_std controls number of quantization levels
            num_levels = max(2, int(1.0 / self.perturbation_noise_std))
            perturbed_normalized = np.round(original_normalized * num_levels) / num_levels
            
        else:
            raise ValueError(f"Unknown label perturbation type: {self.label_perturbation}. "
                           f"Supported: 'gaussian', 'uniform', 'label_flip', 'scaled_noise', 'dropout', 'quantization'")
        
        # Apply perturbations to normalized values
        df_perturbed.loc[perturb_indices, 'tail_norm'] = perturbed_normalized
        
        # Calculate perturbation statistics
        absolute_changes = np.abs(perturbed_normalized - original_normalized)
        relative_changes = absolute_changes / (np.abs(original_normalized) + 1e-8)
        
        print(f"Applied {self.label_perturbation} perturbation to {num_to_perturb}/{num_samples} normalized labels "
              f"({self.perturbation_ratio*100:.1f}%)")
        print(f"  Perturbation stats on normalized values:")
        print(f"    Mean absolute change: {np.mean(absolute_changes):.4f}")
        print(f"    Max absolute change: {np.max(absolute_changes):.4f}")
        print(f"    Mean relative change: {np.mean(relative_changes):.4f}")
        print(f"    Max relative change: {np.max(relative_changes):.4f}")
        print(f"    Normalized value range after perturbation: [{np.min(perturbed_normalized):.3f}, {np.max(perturbed_normalized):.3f}]")
        
        # Store perturbation info for analysis
        self.perturbation_stats = {
            'num_perturbed': num_to_perturb,
            'perturbation_type': self.label_perturbation,
            'mean_absolute_change': np.mean(absolute_changes),
            'max_absolute_change': np.max(absolute_changes),
            'mean_relative_change': np.mean(relative_changes),
            'max_relative_change': np.max(relative_changes),
            'perturbed_indices': list(perturb_indices),
            'original_norm_range': [np.min(original_normalized), np.max(original_normalized)],
            'perturbed_norm_range': [np.min(perturbed_normalized), np.max(perturbed_normalized)]
        }
        
        return df_perturbed

    def _load_data(self):
        # Load mapping from train
        train_path = self.file_paths["train"]
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Data file not found at {train_path}")
        train_df = pd.read_csv(
            train_path, sep="\t", header=None, names=["head", "relation", "tail"]
        )
        train_df = train_df[train_df["head"].isin(self.entity_to_idx)]
        
        # Filter by selected attributes if provided
        if self.selected_attributes is not None:
            original_count = len(train_df)
            train_df = train_df[train_df["relation"].isin(self.selected_attributes)]
            filtered_count = len(train_df)
            print(f"Filtered dataset from {original_count} to {filtered_count} triples using {len(self.selected_attributes)} selected attributes.")
            print(f"Selected attributes: {self.selected_attributes}")
            
            if filtered_count == 0:
                raise ValueError(f"No triples found for selected attributes: {self.selected_attributes}")
        
        self.data_property_to_idx = {
            rel: idx for idx, rel in enumerate(train_df["relation"].unique())
        }

        # reduce the train set for ablations using sampling ratio
        # keeps the sampling_ratio * 100 % of full training set in the train_df

        if self.sampling_ratio is not None:
            if not (0 < self.sampling_ratio <= 1):
                raise ValueError("Fraction must be between 0 and 1.")

            train_df = (
                train_df.groupby("relation", group_keys=False)
                .sample(frac=self.sampling_ratio, random_state=42)
    )
            print(f"Training Literal Embedding model with {self.sampling_ratio*100:.1f}% of the train set.")

        self.num_data_properties = len(self.data_property_to_idx)

        train_df["head_idx"] = train_df["head"].map(self.entity_to_idx)
        train_df["rel_idx"] = train_df["relation"].map(self.data_property_to_idx)
        
        # Apply normalization FIRST, then perturbation
        train_df = self._apply_normalization(train_df)
        train_df = self._apply_label_perturbation(train_df)

        self.triples = torch.tensor(
            train_df[["head_idx", "rel_idx"]].values, dtype=torch.long
        )
        self.tails = torch.tensor(train_df["tail"].values, dtype=torch.float32)
        self.tails_norm = torch.tensor(
            train_df["tail_norm"].values, dtype=torch.float32
        )

    def _apply_normalization(self, df):
        """Applies normalization to the tail values based on the specified type."""
        if self.normalization_type == "z-norm":
            stats = df.groupby("relation")["tail"].agg(["mean", "std"])
            self.normalization_params = stats.to_dict(orient="index")
            df["tail_norm"] = df.groupby("relation")["tail"].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            self.normalization_params["type"] = "z-norm"

        elif self.normalization_type == "min-max":
            stats = df.groupby("relation")["tail"].agg(["min", "max"])
            self.normalization_params = stats.to_dict(orient="index")
            df["tail_norm"] = df.groupby("relation")["tail"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            self.normalization_params["type"] = "min-max"

        elif self.normalization_type == "log":
            # Apply log transformation (add small epsilon to avoid log(0))
            epsilon = 1e-6
            df["tail_norm"] = df["tail"].apply(lambda x: np.log(max(x, epsilon)))
            self.normalization_params["type"] = "log"
            self.normalization_params["epsilon"] = epsilon
            print(f"Applied log transformation with epsilon={epsilon}")

        elif self.normalization_type == "log-z-norm":
            # Apply log transformation followed by z-normalization
            epsilon = 1e-6
            df["tail_log"] = df["tail"].apply(lambda x: np.log(max(x, epsilon)))
            stats = df.groupby("relation")["tail_log"].agg(["mean", "std"])
            self.normalization_params = stats.to_dict(orient="index")
            df["tail_norm"] = df.groupby("relation")["tail_log"].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            self.normalization_params["type"] = "log-z-norm"
            self.normalization_params["epsilon"] = epsilon
            print(f"Applied log transformation + z-normalization with epsilon={epsilon}")

        elif self.normalization_type == "log-min-max":
            # Apply log transformation followed by min-max normalization
            epsilon = 1e-6
            df["tail_log"] = df["tail"].apply(lambda x: np.log(max(x, epsilon)))
            stats = df.groupby("relation")["tail_log"].agg(["min", "max"])
            self.normalization_params = stats.to_dict(orient="index")
            df["tail_norm"] = df.groupby("relation")["tail_log"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            self.normalization_params["type"] = "log-min-max"
            self.normalization_params["epsilon"] = epsilon
            print(f"Applied log transformation + min-max normalization with epsilon={epsilon}")

        else:
            print(" No normalization applied.")
            df["tail_norm"] = df["tail"]
            if self.normalization_type is None:
                self.normalization_params = {}
                self.normalization_params["type"] = None


        return df

    def __getitem__(self, index):
        return self.triples[index], self.tails_norm[index]

    def __len__(self):
        return len(self.triples)

    def get_df(self, split="test", norm=False):
        file_path = self.file_paths.get(split)
        if file_path is None or not os.path.isfile(file_path):
            raise ValueError(
                f"Invalid data split or path does not lead to a file: {file_path}"
            )

        df = pd.read_csv(
            file_path, sep="\t", header=None, names=["head", "relation", "tail"]
        )
        df = df[df["head"].isin(self.entity_to_idx)]
        
        # Filter by selected attributes if provided
        if self.selected_attributes is not None:
            original_count = len(df)
            df = df[df["relation"].isin(self.selected_attributes)]
            print(f"Filtered {split} set from {original_count} to {len(df)} triples using selected attributes.")

        df["head_idx"] = df["head"].map(self.entity_to_idx)
        df["rel_idx"] = df["relation"].map(self.data_property_to_idx)
        
        # Filter out relations not in training set (important when using selected_attributes)
        df = df.dropna(subset=['rel_idx'])

        if norm:
            df = self._apply_normalization(df)

        return df

    def get_batch(self, entity_indices, multi_regression=False, random_seed=None):
        # Combine data and labels into one tensor of shape [n, d+1]
        combined = torch.cat([self.triples, self.tails_norm.unsqueeze(1)], dim=1)

        # Filter by whether first column matches values in entity_indices
        mask = torch.isin(combined[:, 0], entity_indices)
        filtered = combined[mask]

        if random_seed is not None:
            # Use the provided random seed to shuffle the filtered triples
            torch.manual_seed(random_seed)
            shuffled = filtered[torch.randperm(filtered.size(0))]

            # Group by first column (entity indices) and select the first row of each group
            values = shuffled[:, 0]
            unique_vals, inverse_indices = torch.unique(values, return_inverse=True)
            first_indices = torch.stack(
                [
                    (inverse_indices == i).nonzero(as_tuple=True)[0][0]
                    for i in range(len(unique_vals))
                ]
            )
            selected_triples = shuffled[first_indices]
        else:
            # Select all triples for each entity
            selected_triples = filtered

        # Extract entity indices, relation indices, and labels
        ent_ids = selected_triples[:, 0].long()
        rel_ids = selected_triples[:, 1].long()
        labels = selected_triples[:, 2].float()

        if multi_regression:
            y_true = torch.full(
                (len(ent_ids), self.num_data_properties), -9.0, dtype=torch.float32
            )
            y_true[torch.arange(len(ent_ids)), rel_ids] = labels
        else:
            y_true = labels

        return ent_ids, rel_ids, y_true


    def get_ea_encoding(self):
        ea = torch.zeros(
            self.num_entities, self.num_data_properties, dtype=torch.float32
        )
        e = self.triples[:, 0]
        a = self.triples[:, 1]
        ea[e, a] = 1.0
        return ea
    
    def get_available_attributes(self):
        """Get list of all available attributes in the training data"""
        train_path = self.file_paths["train"]
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Data file not found at {train_path}")
        
        train_df = pd.read_csv(
            train_path, sep="\t", header=None, names=["head", "relation", "tail"]
        )
        return sorted(train_df["relation"].unique().tolist())
    
    def get_attribute_stats(self):
        """Get statistics about attributes in the dataset"""
        train_path = self.file_paths["train"]
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Data file not found at {train_path}")
        
        train_df = pd.read_csv(
            train_path, sep="\t", header=None, names=["head", "relation", "tail"]
        )
        train_df = train_df[train_df["head"].isin(self.entity_to_idx)]
        
        # If filtering by selected attributes, show both original and filtered stats
        stats = {}
        if self.selected_attributes is not None:
            # Original stats
            original_stats = train_df["relation"].value_counts()
            stats["original"] = original_stats
            
            # Filtered stats
            filtered_df = train_df[train_df["relation"].isin(self.selected_attributes)]
            filtered_stats = filtered_df["relation"].value_counts()
            stats["filtered"] = filtered_stats
        else:
            stats["all"] = train_df["relation"].value_counts()
        
        return stats
    
    def get_perturbation_info(self):
        """Get information about applied label perturbations"""
        if hasattr(self, 'perturbation_stats'):
            return self.perturbation_stats
        else:
            return {"message": "No perturbations were applied"}
    
    @staticmethod
    def denormalize( preds_norm, attributes, normalization_params) -> np.ndarray:
        """Denormalizes the predictions based on the normalization type.

        Args:
        preds_norm (np.ndarray): Normalized predictions to be denormalized.
        attributes (list): List of attributes corresponding to the predictions.
        normalization_params (dict): Dictionary containing normalization parameters for each attribute.

        Returns:
            np.ndarray: Denormalized predictions.

        """
        if normalization_params["type"] == "z-norm":
            # Extract means and stds only if z-norm is used
            means = np.array([normalization_params[i]["mean"] for i in attributes])
            stds = np.array([normalization_params[i]["std"] for i in attributes])
            return preds_norm * stds + means

        elif normalization_params["type"] == "min-max":
            # Extract mins and maxs only if min-max is used
            mins = np.array([normalization_params[i]["min"] for i in attributes])
            maxs = np.array([normalization_params[i]["max"] for i in attributes])
            return preds_norm * (maxs - mins) + mins

        elif normalization_params["type"] == "log":
            # Reverse log transformation: exp(log_values)
            return np.exp(preds_norm)

        elif normalization_params["type"] == "log-z-norm":
            # Reverse z-norm first, then reverse log transformation
            means = np.array([normalization_params[i]["mean"] for i in attributes])
            stds = np.array([normalization_params[i]["std"] for i in attributes])
            log_values = preds_norm * stds + means
            return np.exp(log_values)

        elif normalization_params["type"] == "log-min-max":
            # Reverse min-max first, then reverse log transformation
            mins = np.array([normalization_params[i]["min"] for i in attributes])
            maxs = np.array([normalization_params[i]["max"] for i in attributes])
            log_values = preds_norm * (maxs - mins) + mins
            return np.exp(log_values)

        elif normalization_params["type"] is None:
            return  preds_norm  # No normalization applied, return as is

        else:
            raise ValueError(
                f"Unsupported normalization type: {normalization_params['type']}. "
                f"Supported types: 'z-norm', 'min-max', 'log', 'log-z-norm', 'log-min-max', or None."
            )
#!/usr/bin/env python3
"""
Simple Disjoint Dataset Creator

A minimal script with a single method to create entity-disjoint train-test splits
for literal embedding datasets.

Usage:
    from create_disjoint_simple import create_disjoint_dataset
    create_disjoint_dataset("KGs/FB15k-237", "KGs/FB15k-237_disjoint")
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Create entity-disjoint train-test split for literal datasets.")
    parser.add_argument("--input_dir", type=str, default="KGs/FB15k-237", help="Input directory containing literals folder")
    parser.add_argument("--output_dir", type=str, default="KGs/FB15k-237_disjoint", help="Output directory for disjoint split")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="Fraction of entities for test set")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def create_disjoint_dataset(input_dir, output_dir, test_ratio=0.3, random_seed=42):
    """
    Create entity-disjoint train-test split from literal dataset.
    
    Args:
        input_dir (str): Input directory containing literals folder with train/test/val files
        output_dir (str): Output directory where disjoint split will be saved
        test_ratio (float): Fraction of entities for test set (default: 0.3)
        random_seed (int): Random seed for reproducibility (default: 42)
    
    Returns:
        dict: Statistics about the created split
    """
    np.random.seed(random_seed)
    
    # Load all data files
    input_path = Path(input_dir)
    literals_dir = input_path / "literals"
    
    if not literals_dir.exists():
        raise FileNotFoundError(f"Literals directory not found: {literals_dir}")
    
    # Find and load all literal files
    file_patterns = ["train.txt", "test.txt", "val.txt", "valid.txt"]
    dfs = []
    
    for pattern in file_patterns:
        file_path = literals_dir / pattern
        if file_path.exists():
            df = pd.read_csv(file_path, sep="\t", header=None, names=["head", "relation", "tail"])
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No literal files found in {literals_dir}")
    
    # Combine all data
    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Get entities and attributes
    all_entities = all_df['head'].unique()
    all_attributes = all_df['relation'].unique()
    
    # Create attribute-entity mapping
    attribute_entities = {}
    for attr in all_attributes:
        attribute_entities[attr] = all_df[all_df['relation'] == attr]['head'].unique()
    
    # Classify entities by attribute count
    entity_attr_count = all_df.groupby('head')['relation'].nunique()
    multi_attr_entities = entity_attr_count[entity_attr_count > 1].index.tolist()
    single_attr_entities = entity_attr_count[entity_attr_count == 1].index.tolist()
    
    # Shuffle entities
    np.random.shuffle(multi_attr_entities)
    np.random.shuffle(single_attr_entities)
    
    # Calculate target sizes
    total_entities = len(all_entities)
    test_size = int(total_entities * test_ratio)
    
    # Initialize entity sets
    test_entities = set()
    train_entities = set()
    
    # Ensure each attribute has representation in both train and test
    for attr in all_attributes:
        attr_entities_list = list(attribute_entities[attr])
        
        # Prefer multi-attribute entities for coverage
        multi_for_attr = [e for e in attr_entities_list if e in multi_attr_entities]
        single_for_attr = [e for e in attr_entities_list if e in single_attr_entities]
        
        # Add one entity to test if not covered yet
        if len(test_entities) < test_size:
            if multi_for_attr and multi_for_attr[0] not in train_entities:
                test_entities.add(multi_for_attr[0])
            elif single_for_attr and single_for_attr[0] not in train_entities:
                test_entities.add(single_for_attr[0])
        
        # Add one entity to train if not covered yet
        remaining_multi = [e for e in multi_for_attr if e not in test_entities]
        remaining_single = [e for e in single_for_attr if e not in test_entities]
        
        if remaining_multi:
            train_entities.add(remaining_multi[0])
        elif remaining_single:
            train_entities.add(remaining_single[0])
    
    # Fill remaining test entities
    remaining_entities = [e for e in all_entities if e not in train_entities and e not in test_entities]
    np.random.shuffle(remaining_entities)
    
    while len(test_entities) < test_size and remaining_entities:
        test_entities.add(remaining_entities.pop())
    
    # Add rest to train
    train_entities.update(remaining_entities)
    
    # Verify no overlap
    overlap = train_entities.intersection(test_entities)
    if overlap:
        raise ValueError(f"Entity overlap detected: {len(overlap)} entities")
    
    # Create train and test dataframes
    train_df = all_df[all_df['head'].isin(train_entities)].copy()
    test_df = all_df[all_df['head'].isin(test_entities)].copy()
    
    # Save files
    output_path = Path(output_dir)
    literals_output_dir = output_path / "literals"
    literals_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output_path = literals_output_dir / "train.txt"
    test_output_path = literals_output_dir / "test.txt"
    
    train_df[['head', 'relation', 'tail']].to_csv(train_output_path, sep='\t', header=False, index=False)
    test_df[['head', 'relation', 'tail']].to_csv(test_output_path, sep='\t', header=False, index=False)
    
    # Return statistics
    return {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'total_triples': len(all_df),
        'train_triples': len(train_df),
        'test_triples': len(test_df),
        'train_entities': len(train_entities),
        'test_entities': len(test_entities),
        'attributes': len(all_attributes),
        'entity_overlap': len(overlap),
        'files_created': [str(train_output_path), str(test_output_path)]
    }

if __name__ == "__main__":
    args = parse_args()
    result = create_disjoint_dataset(
        args.input_dir,
        args.output_dir,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    print(f"Disjoint dataset created successfully!")
    print(f"Input directory: {result['input_dir']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Total triples: {result['total_triples']}")
    print(f"Train triples: {result['train_triples']}")
    print(f"Test triples: {result['test_triples']}")
    print(f"Train entities: {result['train_entities']}")
    print(f"Test entities: {result['test_entities']}")
    print(f"Attributes: {result['attributes']}")
    print(f"Entity overlap: {result['entity_overlap']}")
    print(f"Files created: {', '.join(result['files_created'])}")
    print(f"✅ Disjoint dataset creation completed!")
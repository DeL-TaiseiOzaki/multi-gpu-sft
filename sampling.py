import argparse
import json
import random
from pathlib import Path

def sample_jsonl(input_file, output_file, sample_size):
    """Sample random lines from JSONL file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if sample_size > len(lines):
        raise ValueError(f"Sample size {sample_size} exceeds file length {len(lines)}")
        
    sampled_lines = random.sample(lines, sample_size)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random sampling from JSONL file')
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('output_file', help='Output JSONL file')
    parser.add_argument('-n', '--num_samples', type=int, required=True, help='Number of lines to sample')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    if args.seed:
        random.seed(args.seed)
        
    sample_jsonl(args.input_file, args.output_file, args.num_samples)
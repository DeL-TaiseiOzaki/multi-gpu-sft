import argparse
import json
from pathlib import Path

def merge_jsonl_files(input_files, output_file):
    """
    Merge multiple JSONL files in the specified order
    
    Args:
        input_files (list): List of input JSONL file paths
        output_file (str): Output JSONL file path
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge JSONL files in specified order')
    parser.add_argument('input_files', nargs='+', help='Input JSONL files')
    parser.add_argument('-o', '--output', required=True, help='Output JSONL file')
    
    args = parser.parse_args()
    merge_jsonl_files(args.input_files, args.output)
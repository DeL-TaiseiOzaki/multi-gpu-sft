import json

def process_jsonl(input_file, output_file, column_name, process_function):
    """
    Process a JSONL file line by line, applying a specific function to a given column.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to the output JSONL file.
        column_name (str): The column name to process.
        process_function (callable): A function to apply to the specified column.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)  # Parse the JSON line
                if column_name in data:
                    data[column_name] = process_function(data[column_name])  # Apply the processing function
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')  # Write the processed line to the output file
            except json.JSONDecodeError:
                print(f"Error decoding line: {line.strip()}")
            except Exception as e:
                print(f"Error processing line: {line.strip()} - {e}")

def remove_before_instruction_and_delimiter(value):
    """
    Removes text before "指示文" and the first delimiter found after it, 
    if "指示文" is within the first 30 characters.
    """
    if isinstance(value, str):
        index = value.find("指示文")
        if 0 <= index <= 30:
            delimiters = ["："]
            first_delimiter_index = float('inf')
            found_delimiter = ""
            
            for delimiter in delimiters:
                delimiter_index = value.find(delimiter, index + len("指示文"))
                if delimiter_index != -1 and delimiter_index < first_delimiter_index:
                    first_delimiter_index = delimiter_index
                    found_delimiter = delimiter
            
            if first_delimiter_index != float('inf'):
                return value[first_delimiter_index + len(found_delimiter):]
            else:
                return value[index + len("指示文"):]
    return value

# Example usage
if __name__ == "__main__":
    # Input and output file paths
    input_jsonl = "gen_dataset/generated_sets.jsonl"
    output_jsonl = "gen_dataset/generated_sets2.jsonl"

    # Column to process
    target_column = "instruction"

    # Run the processing function
    process_jsonl(input_jsonl, output_jsonl, target_column, remove_before_instruction_and_delimiter)
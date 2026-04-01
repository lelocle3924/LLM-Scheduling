import argparse
import json
import os
import re

def convert_txt_to_json(input_path, output_path=None):
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".json"
        
    with open(input_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if not lines:
        print(f"File {input_path} is empty.")
        return
        
    try:
        header = lines[0].split()
        num_jobs = int(header[0])
        num_machines = int(header[1])
        
        jobs = []
        
        for line_idx in range(1, num_jobs + 1):
            if line_idx >= len(lines):
                break
                
            job_tokens = [int(x) for x in lines[line_idx].split()]
            job = []
            
            for i in range(0, len(job_tokens), 2):
                machine = job_tokens[i]
                processing_time = job_tokens[i+1]
                operation = [{
                    "machine": machine,
                    "processing": processing_time
                }]
                job.append(operation)
                
            jobs.append(job)
    except (IndexError, ValueError) as e:
        print(f"Error parsing file {input_path}: {e}")
        return

    data = {
        "machines": num_machines,
        "jobs": jobs
    }

    json_str = json.dumps(data, indent=2)
    # Format the operation object so it's cleanly on 1 line
    json_str = re.sub(
        r'\[\s*\{\s*"machine":\s*(\d+),\s*"processing":\s*(\d+)\s*\}\s*\]',
        r'[{"machine": \1,"processing": \2}]',
        json_str
    )

    with open(output_path, 'w') as f:
        f.write(json_str)
        
    print(f"Successfully converted {input_path} -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert FJSP instance from .txt to .json")
    parser.add_argument("input", help="Input .txt file or directory containing .txt files")
    parser.add_argument("-o", "--output", help="Output .json file (only used if input is a file)", default=None)
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            if filename.endswith(".txt"):
                input_path = os.path.join(args.input, filename)
                convert_txt_to_json(input_path)
    else:
        convert_txt_to_json(args.input, args.output)

def main2():
    for i in range(20):
        convert_txt_to_json(f"mt{i}.txt")

if __name__ == "__main__":
    main2()

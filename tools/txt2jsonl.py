import os
import errno
import jsonlines
import argparse

def ValidateFilepath(filepath: str) -> str:
    """
    Validate if the provided filepath exists.

    Args:
        filepath [str]: the file path.

    Raises:
        FileNotFoundError if the filepath does not exist.

    Returns:
        The filepath if it is existed.
    """
    if not os.path.isdir(filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)
    return filepath

print("Converting txt files into jsonl files...")

parser = argparse.ArgumentParser(description="Convert a novel from txt format to jsonl format.")

parser.add_argument("--input", default="", type=str, help="The path to the input txt dir.")
parser.add_argument("--output", default="", type=str, help="The path to the output jsonl file.")

args = parser.parse_args()


novel_folder = ValidateFilepath(args.input)
output_file = args.output

novel_id =1

for filename in os.listdir(novel_folder):
    if filename.endswith(".txt"):
        novel_path = os.path.join(novel_folder, filename)

        with open(novel_path, "r", encoding="utf-8") as f:
            novel_content = f.read().replace("\n", " ")

        novel_json = {"meta": {"ID": f'{novel_id:05d}'}, "text": novel_content}

        with jsonlines.open(output_file, mode="a") as writer:
            writer.write(novel_json)
        novel_id += 1

print("Conversion done, output path: ", args.output)

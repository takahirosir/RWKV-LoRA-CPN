import os
import jsonlines

novel_folder = "/root/autodl-tmp/txt"
output_file = "/root/autodl-tmp/jsonl/text.jsonl"

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

print("转换完成！")

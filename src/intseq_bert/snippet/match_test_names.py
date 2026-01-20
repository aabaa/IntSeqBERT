
import json
import os

def main():
    test_split_path = "data/oeis/splits/std/test.txt"
    jsonl_path = "data/oeis/data.jsonl"
    num_items = 200

    if not os.path.exists(test_split_path):
        print(f"Error: {test_split_path} not found")
        return

    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found")
        return

    # Read top N IDs
    target_ids = []
    with open(test_split_path, "r") as f:
        for _ in range(num_items):
            line = f.readline()
            if not line: break
            target_ids.append(line.strip())

    target_set = set(target_ids)
    names = {}

    # Scan JSONL
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data["oeis_id"] in target_set:
                    names[data["oeis_id"]] = data["name"]
                    if len(names) == len(target_set):
                        break
            except json.JSONDecodeError:
                continue

    # Print results
    print(f"# Top {num_items} Sequences in Test Split\n")
    for oid in target_ids:
        name = names.get(oid, "Not Found")
        print(f"- **{oid}**: {name}")

if __name__ == "__main__":
    main()

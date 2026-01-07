import json
from collections import Counter
from pathlib import Path

def main():
    file_path = Path("data/oeis/data_final.jsonl")
    
    # カウントしたい注目タグ
    target_tags = ["core", "easy", "nice", "word", "cons", "fini", "dead", "dumb", "unkn", "random"]
    
    tag_counts = Counter()
    total_seqs = 0
    
    print("Counting tags...")
    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                kws = data.get("keywords", "")
                
                # キーワードリストの正規化
                if isinstance(kws, str):
                    tags = [t.strip() for t in kws.split(",")]
                elif isinstance(kws, list):
                    tags = kws
                else:
                    tags = []
                
                tag_counts.update(tags)
                total_seqs += 1
            except:
                continue

    print(f"\nTotal Sequences: {total_seqs:,}")
    print("-" * 30)
    print("Tag Frequency:")
    for tag in target_tags:
        count = tag_counts[tag]
        ratio = (count / total_seqs) * 100
        print(f"{tag:<10}: {count:>7,} ({ratio:>5.1f}%)")
    
    print("-" * 30)
    print("Top 20 Most Common Tags:")
    for tag, count in tag_counts.most_common(20):
        print(f"{tag:<10}: {count:>7,}")

if __name__ == "__main__":
    main()
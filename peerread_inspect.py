from pathlib import Path

root = Path("PeerRead")

if not root.exists():
    print("PeerRead repo not found.")
    print("Run: git clone https://github.com/allenai/PeerRead.git")
    exit()

print("\n===== PeerRead Repo Found =====")
print("Path:", root.resolve())

print("\n===== Top-level files/folders =====")
for item in root.iterdir():
    print(item)

print("\n===== Train / Dev / Test folders found =====")
for split in ["train", "dev", "test"]:
    matches = list(root.rglob(split))
    print(f"\n{split.upper()} folders:", len(matches))
    for m in matches[:20]:
        print(" -", m)

print("\n===== Sample files =====")
files = [f for f in root.rglob("*") if f.is_file()]
for f in files[:50]:
    print(f)

print("\nTotal files found:", len(files))
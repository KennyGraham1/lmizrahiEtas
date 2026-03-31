import os

root_dir = "etas"
params_to_add = "from __future__ import annotations\n"

count = 0
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".py"):
            filepath = os.path.join(dirpath, filename)
            with open(filepath, "r") as f:
                content = f.read()
            
            if "from __future__ import annotations" not in content:
                with open(filepath, "w") as f:
                    f.write(params_to_add + content)
                print(f"Patched {filepath}")
                count += 1
            else:
                print(f"Skipping {filepath} (already present)")

print(f"Total patched files: {count}")

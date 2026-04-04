import os
import sys
import shutil
from huggingface_hub import hf_hub_download

def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'fineweb10B')
    dest = os.path.join(local_dir, fname)
    if not os.path.exists(dest):
        os.makedirs(local_dir, exist_ok=True)
        cached_path = hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                                      repo_type="dataset")
        shutil.copy(cached_path, dest)

get("fineweb_val_%06d.bin" % 0)
num_chunks = 103
if len(sys.argv) >= 2:
    num_chunks = int(sys.argv[1])
for i in range(1, num_chunks+1):
    print(f"Downloading chunk {i}")
    get("fineweb_train_%06d.bin" % i)
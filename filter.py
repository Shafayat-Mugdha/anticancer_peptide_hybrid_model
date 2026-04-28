import os
import glob
import numpy as np
from Bio import SeqIO

# -------------------------
# 1️⃣ Load sequences from FASTA
# -------------------------
fasta_file = r"E:\Anticancer\code\anticancer model\acp740\new paper\acp740.txt"
records = list(SeqIO.parse(fasta_file, "fasta"))

sequence_ids = [record.id.split("|")[0] for record in records]  # remove "|1"
print(f"✅ Loaded {len(sequence_ids)} sequence IDs from FASTA")

# -------------------------
# 2️⃣ Locate PSSM files
# -------------------------
pssm_dir = r"E:\Anticancer\code\dataset\pssm_output\acp740"
pssm_files = glob.glob(os.path.join(pssm_dir, "*.pssm"))
pssm_file_ids = [os.path.splitext(os.path.basename(f))[0] for f in pssm_files]
print(f"✅ Found {len(pssm_files)} PSSM files")

# -------------------------
# 3️⃣ Create keep_idx for sequences that have PSSM
# -------------------------
keep_idx = []
for i, seq_id in enumerate(sequence_ids):
    mapped_pssm_id = seq_id.replace("ACP_", "seq_")
    if mapped_pssm_id in pssm_file_ids:
        keep_idx.append(i)

print(f"✅ {len(keep_idx)} sequences have corresponding PSSM files")

# -------------------------
# 4️⃣ Load all .npy files
# -------------------------
sequences_padded = np.load("sequences_padded.npy")
physicochemical = np.load("physicochemical.npy")
labels = np.load("labels.npy")

# -------------------------
# 5️⃣ Filter arrays based on keep_idx
# -------------------------
sequences_padded_filtered = sequences_padded[keep_idx]
physicochemical_filtered = physicochemical[keep_idx]
labels_filtered = labels[keep_idx]

# -------------------------
# 6️⃣ Save filtered arrays
# -------------------------
np.save("sequences_padded_filtered.npy", sequences_padded_filtered)
np.save("physicochemical_filtered.npy", physicochemical_filtered)
np.save("labels_filtered.npy", labels_filtered)

print("✅ All arrays filtered and saved:")
print("sequences_padded_filtered.npy:", sequences_padded_filtered.shape)
print("physicochemical_filtered.npy:", physicochemical_filtered.shape)
print("labels_filtered.npy:", labels_filtered.shape)

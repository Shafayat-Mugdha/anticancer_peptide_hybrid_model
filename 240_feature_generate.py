import numpy as np
import pandas as pd
import os
import glob
import re
from Bio import SeqIO

# ⚙️ Configuration for ACP240
FASTA_240 = "acp240.txt"
PSSM_DIR_240 = r"E:\Anticancer\code\dataset\pssm_output\acp240" 
# নিশ্চিত হোন এই ফোল্ডারে ACP240 এর PSSM ফাইলগুলো আছে
PHYS_CSV_240 = "physicochemical_combined_clean_240.csv" 
PSSM_EXPECTED_ROWS = 30 

print("🚀 Processing ACP240 for Independent Testing...")

# 1. Load Sequences & Labels
records = list(SeqIO.parse(FASTA_240, "fasta"))
sequences = [str(rec.seq) for rec in records]
labels = np.array([int(rec.description.split("|")[1]) for rec in records], dtype=np.int32)

# 2. Sequence Padding (740 এর সাথে লেন্থ মিল রাখতে হবে, ধরুন ৯৭)
aa_dict = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
aa_dict['X'] = 0
max_len = 97 # ACP740 এ যা ছিল তাই দিন

seq_padded = []
for seq in sequences:
    encoded = [aa_dict.get(aa, 0) for aa in seq]
    encoded += [0]*(max_len - len(seq))
    seq_padded.append(encoded[:max_len])

# 3. PSSM Processing
pssm_list = []
# এখানে আমরা ধরে নিচ্ছি আপনার PSSM ফাইলগুলোর নাম seq_1.pssm স্টাইলে আছে
for i in range(1, len(records) + 1):
    pssm_file = os.path.join(PSSM_DIR_240, f"seq_{i}.pssm")
    if os.path.exists(pssm_file):
        with open(pssm_file) as f:
            lines = f.readlines()
        matrix = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 22 and parts[0].isdigit():
                matrix.append([float(x) for x in parts[2:22]])
        
        matrix = 1 / (1 + np.exp(-np.array(matrix, dtype=np.float32))) # Sigmoid normalization
        if len(matrix) < PSSM_EXPECTED_ROWS:
            matrix = np.vstack([matrix, np.zeros((PSSM_EXPECTED_ROWS - len(matrix), 20))])
        else:
            matrix = matrix[:PSSM_EXPECTED_ROWS, :]
        pssm_list.append(matrix)
    else:
        print(f"⚠️ Missing PSSM for seq_{i}")

# 4. Physicochemical (Assuming you have this CSV ready for 240)
df_phys = pd.read_csv(PHYS_CSV_240)
phys_features = df_phys.values.astype(np.float32)

# 5. Save everything
np.save("sequences_padded_240.npy", np.array(seq_padded, dtype=np.int32))
np.save("pssm_features_240.npy", np.array(pssm_list, dtype=np.float32))
np.save("physicochemical_240.npy", phys_features)
np.save("labels_240.npy", labels)

print(f"✅ Done! Created features for {len(labels)} samples.")
import os
import shutil

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

source_folder = "../data/tcga_brca/mrna" # methylation, mirna
destination_folder = "../data/tcga_brca/mrna_txt" # methylation_txt, mirna_txt

# Create destination if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.endswith(".txt"):
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)
        shutil.move(src_path, dst_path)

print("Done moving .txt files.")

os.makedirs(destination_folder, exist_ok=True)
#TSV is for RNA:
for filename in os.listdir(source_folder):
    if filename.endswith(".tsv"):
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)
        shutil.move(src_path, dst_path)

print("Done moving .tsv files.")

#################################################################################

input_folder = "../data/tcga_brca/methylation_txt" # methylation_txt, mirna_txt
output_folder = "../data/tcga_brca/mrna_txt_transpose" # methylation_txt_transpose, mirna_txt_transpose

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    in_path = os.path.join(input_folder, filename)
    out_path = os.path.join(output_folder, filename)

    # Read file (adjust sep if needed: "\t" for TSV)
    df = pd.read_csv(in_path, sep="\t", header=None)

    # Transpose
    df_t = df.T

    # Save
    df_t.to_csv(out_path, sep="\t", header=False, index=False)

print("All files transposed.")

#This is for RNA:
for filename in os.listdir(input_folder):
    in_path = os.path.join(input_folder, filename)
    out_path = os.path.join(output_folder, filename)

    # Read file (adjust sep if needed: "\t" for TSV)
    df = pd.read_csv(in_path, sep=None, header=None)

    # Transpose
    df_t = df.T

    # Save
    df_t.to_csv(out_path, sep="\t", header=False, index=False)

print("All files transposed.")

#################################################################################

input_folder = "../data/tcga_brca/methylation_txt_transpose"

def process_file(filename):
    file_path = os.path.join(input_folder, filename)
    patient_id = filename[:12]  # first 12 chars of filename
    try:
        df = pd.read_csv(file_path, sep="\t", header=None, nrows=2)
        if df.shape[0] < 2:
            return None
        values = df.iloc[1].values  # second row
        # Combine patient_id and values into one DataFrame row
        row_df = pd.DataFrame([values])
        row_df.insert(0, "patient_id", patient_id)
        return row_df
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

# Run in parallel
results = Parallel(n_jobs=-1)(
    delayed(process_file)(f) for f in os.listdir(input_folder)
)

# Remove None results
results = [r for r in results if r is not None]

# Concatenate all rows into one DataFrame
final_df = pd.concat(results, ignore_index=True)

# Save
final_df.to_csv("../data/tcga_brca/methylation_txt_transpose/methyl.csv")

print("Done. Shape:", final_df.shape)

#################################################################################

input_folder = "../data/tcga_brca/mirna_txt_transpose"

def process_file(filename):
    file_path = os.path.join(input_folder, filename)
    try:
        df = pd.read_csv(file_path, sep="\t", header=None)
        if df.shape[0] < 2:
            return None
        return filename[:12], df.iloc[3].values # 4th row
    except:
        return None

results = Parallel(n_jobs=-1)(
    delayed(process_file)(f) for f in os.listdir(input_folder)
)

# Filter None
results = [r for r in results if r is not None]
index = [r[0] for r in results]
rows = [r[1] for r in results]

final_df = pd.DataFrame(rows, index=index)

# Save
final_df.to_csv("../data/tcga_brca/mirna_txt_transpose/mirna.csv")

print("Done. Shape:", final_df.shape)

#################################################################################

input_folder = "../data/tcga_brca/mrna_txt_transpose"

def process_file(filename):
    file_path = os.path.join(input_folder, filename)
    try:
        df = pd.read_csv(file_path, sep=None, header=None, nrows=2)
        if df.shape[0] < 2 or df.shape[1] < 4:
            return None
        return filename[:12], df.iloc[0].values
    except:
        return None

results = Parallel(n_jobs=-1)(
    delayed(process_file)(f) for f in os.listdir(input_folder)
)

results = [r for r in results if r is not None]

index = [r[0] for r in results]
values = [r[1] for r in results]

final_df = pd.DataFrame(values, index=index)


# Save
final_df.to_csv("../data/tcga_brca/mrna_txt_transpose/mrna.csv")

print("Done. Shape:", final_df.shape)

#################################################################################

df = pd.read_csv("../data/tcga_brca/mrna_txt_transpose/mrna.csv", header=None)

all_rows = []
gene_ids = None

for i in range(1, df.shape[0]):
    row = df.iloc[i]   # entire row as string

    # Extract patient_id
    patient_id = row[0]

    # unstranded counts
    gene_counts = row.iloc[7:].apply(lambda x: x.split("\t")[3])

    values_list = gene_counts.values
    values_str = [str(x) for x in values_list]
    gene_str = ",".join(values_str)
    row_str = patient_id + "," + gene_str

    all_rows.append(row_str)

# Convert to DataFrame
df_final = pd.DataFrame(all_rows)

df_split = df_final.iloc[:, 0].str.split(',', expand=True)

# Rename first column as patient_id
df_split.rename(columns={0: 'patient_id'}, inplace=True)

# Convert remaining columns to numeric if needed
for col in df_split.columns[1:]:
    df_split[col] = pd.to_numeric(df_split[col], errors='coerce')

print("Done. New shape:", df_split.shape)

# Save to CSV
df_split.to_csv("../data/tcga_brca/mrna_txt_transpose/mrna_counts.csv", index=False)

################################################################################# most variant genes

df = pd.read_csv("../data/tcga_brca/mrna_txt_transpose/mrna_counts.csv", header=None)

# Assume first column is patient ID
patient_id = df.iloc[:, 0]          # grab first column
gene_data = df.iloc[:, 1:]          # all remaining columns

# Compute variance for each column
variances = gene_data.var(axis=0)

# Select top 2000 columns by variance
top2000_cols = variances.sort_values(ascending=False).head(2000).index

# Create reduced dataframe
df_reduced = pd.concat([patient_id, gene_data[top2000_cols]], axis=1)

print("Reduced shape:", df_reduced.shape)

# Save
df_reduced.to_csv("../data/tcga_brca/mrna_txt_transpose/mrna_counts_most_variant.csv",  index=False, header=False)

print("Done! Shape:", df_reduced.shape)

################################################################################## most variant methyl features

# Read CSV
df = pd.read_csv("../data/tcga_brca/methylation_txt_transpose/methyl.csv", header=None)
out_file = "../data/tcga_brca/methylation_txt_transpose/methyl_most_variant.csv"


print(df.iloc[:5, :5])

# Separate patient_id
patient_id = df.iloc[1:, 1]

# Convert all remaining columns to numeric (VERY IMPORTANT)
data = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')

# Replace NaN (from bad values) with 0 or column mean
data = data.fillna(0)

# Convert to numpy (now safe)
data_np = data.to_numpy(dtype=np.float32)

# Compute variance
variances = np.var(data_np, axis=0)

# Top 2000 indices
top_idx = np.argpartition(-variances, 2000)[:2000]
top_idx = top_idx[np.argsort(-variances[top_idx])]

# Select data
data_reduced = data_np[:, top_idx]

# Back to DataFrame
#df_reduced = pd.concat([patient_id.reset_index(drop=True), pd.DataFrame(data_reduced)], axis=1)
df_reduced = pd.DataFrame(data_reduced)
# Add patient_id
df_reduced.insert(0, "patient_id", patient_id.values)

# Save
df_reduced.to_csv(out_file, index=False, header=False)

print("Done! Shape:", df_reduced.shape)

print(df_reduced.iloc[:5, :5])

################################################################################## most variant mirna features

# Read file
df = pd.read_csv("../data/tcga_brca/mirna_txt_transpose/mirna.csv")

# Remove rows where ANY column contains "cross-mapped"
df_clean = df[~df.astype(str).apply(lambda x: x.str.contains("cross-mapped", case=False)).any(axis=1)]

df_clean.drop(df_clean.columns[1], axis=1, inplace=True)


print(df_clean.iloc[:5, :5])

# Separate patient_id
patient_id = df_clean.iloc[:, 0]

# Convert all remaining columns to numeric (VERY IMPORTANT)
data = df_clean.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Replace NaN (from bad values) with 0 or column mean
data = data.fillna(0)

# Convert to numpy (now safe)
data_np = data.to_numpy(dtype=np.float32)

# Compute variance
variances = np.var(data_np, axis=0)

# Top 2000 indices
top_idx = np.argpartition(-variances, 2000)[:2000]
top_idx = top_idx[np.argsort(-variances[top_idx])]

# Select data
data_reduced = data_np[:, top_idx]


df_reduced = pd.DataFrame(data_reduced)
# Add patient_id
df_reduced.insert(0, "patient_id", patient_id.values)

# Save
df_reduced.to_csv("../data/tcga_brca/mirna_txt_transpose/mirna_most_variant.csv", index=False)

print(df_reduced.iloc[:5, :5])

print("Done! Shape:", df_reduced.shape)

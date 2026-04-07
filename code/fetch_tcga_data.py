import requests
import pandas as pd
import os
BASE_URL = "https://api.gdc.cancer.gov"

# -----------------------------
# Helper function
# # -----------------------------
def query_gdc(filters, fields, size=500):
    url = f"{BASE_URL}/files"
    params = {
        "filters": filters,
        "fields": ",".join(fields),
        "format": "JSON",
        "size": size
    }
    response = requests.post(url, json=params)
    return response.json()["data"]["hits"]


# -----------------------------
# Common filters
# -----------------------------
def build_filter(data_category, data_type=None, workflow_type=None, platform=None):
    content = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "data_category", "value": [data_category]}}
    ]

    if data_type:
        content.append({"op": "in", "content": {"field": "data_type", "value": [data_type]}})
    if workflow_type:
        content.append({"op": "in", "content": {"field": "analysis.workflow_type", "value": [workflow_type]}})
    if platform:
        content.append({"op": "in", "content": {"field": "platform", "value": [platform]}})

    return {"op": "and", "content": content}


# -----------------------------
# Fields to retrieve
# -----------------------------
FIELDS = [
    "file_id",
    "file_name",
    "cases.submitter_id",
    "cases.samples.sample_type"
]


# -----------------------------
# 1. mRNA
# -----------------------------
mrna_hits = query_gdc(
    build_filter(
        data_category="Transcriptome Profiling",
        data_type="Gene Expression Quantification"#,
        #workflow_type="HTSeq - Counts"
    ),
    FIELDS
)

# -----------------------------
# 2. miRNA
# -----------------------------
mirna_hits = query_gdc(
    build_filter(
        data_category="Transcriptome Profiling",
        data_type="miRNA Expression Quantification"
    ),
    FIELDS
)

# -----------------------------
# 3. DNA methylation
# -----------------------------
meth_hits = query_gdc(
    build_filter(
        data_category="DNA Methylation",
        platform="Illumina Human Methylation 450"
    ),
    FIELDS
)


# -----------------------------
# Convert to DataFrame
# -----------------------------
def hits_to_df(hits, omic_name):
    rows = []
    for h in hits:
        if "cases" in h and len(h["cases"]) > 0:
            case = h["cases"][0]
            rows.append({
                "patient_id": case["submitter_id"][:12],
                "sample_type": case["samples"][0]["sample_type"],
                f"{omic_name}_file_id": h["file_id"]
            })
    return pd.DataFrame(rows)


df_mrna = hits_to_df(mrna_hits, "mrna")
df_mirna = hits_to_df(mirna_hits, "mirna")
df_meth = hits_to_df(meth_hits, "meth")


# -----------------------------
# Keep only Primary Tumor
# -----------------------------
# df_mrna = df_mrna[df_mrna["sample_type"] == "Primary Tumor"]
# df_mirna = df_mirna[df_mirna["sample_type"] == "Primary Tumor"]
# df_meth = df_meth[df_meth["sample_type"] == "Primary Tumor"]

print("mrna:", df_mrna.shape)
print("mirna:", df_mirna.shape)
print("meth:", df_meth.shape)
# -----------------------------
# Merge all omics
# -----------------------------
df_multi = df_mrna.merge(df_mirna, on="patient_id", how="inner") \
                  .merge(df_meth, on="patient_id", how="inner")


# -----------------------------
# 4. Clinical (tumor stage)
# -----------------------------
clinical_url = f"{BASE_URL}/cases"

clinical_params = {
    "filters": {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": ["TCGA-BRCA"]
        }
    },
    "fields": "submitter_id,diagnoses.ajcc_pathologic_stage,diagnoses.ajcc_clinical_stage",
   # "fields": "submitter_id,diagnoses.tumor_stage",
    "format": "JSON",
    "size": 2000
}

clinical = requests.post(clinical_url, json=clinical_params).json()["data"]["hits"]

clinical_rows = []

for c in clinical:
    stage = None

    diagnoses = c.get("diagnoses", [])
    if diagnoses:
        d = diagnoses[0]

        stage = (
                d.get("ajcc_pathologic_stage") or
                d.get("ajcc_clinical_stage") or
                d.get("tumor_stage")
        )

    clinical_rows.append({
        "patient_id": c["submitter_id"][:12],
        "tumor_stage": stage
    })

df_clinical = pd.DataFrame(clinical_rows)


# -----------------------------
# Merge with omics
# -----------------------------
df_final = df_multi.merge(df_clinical, on="patient_id", how="left")

df_final = df_final[
    df_final["tumor_stage"].notna() &
    (df_final["tumor_stage"] != "Stage X")
]

valid_stages = [
    "Stage I", "Stage IA", "Stage IB",
    "Stage II", "Stage IIA", "Stage IIB",
    "Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC",
    "Stage IV"
]

df_final = df_final[df_final["tumor_stage"].isin(valid_stages)]

df_final["stage_main"] = df_final["tumor_stage"].str.extract(r"(Stage [IVX]+)")

stage_map = {
    "Stage I": 1,
    "Stage II": 2,
    "Stage III": 3,
    "Stage IV": 4
}

df_final["stage_numeric"] = df_final["stage_main"].map(stage_map)

print(df_final["stage_main"].value_counts())

# -----------------------------
# Save result
# -----------------------------
df_final.to_csv("tcga_brca_multiomics_mapping.csv", index=False)

print("Done! Saved tcga_brca_multiomics_mapping.csv")
print(df_final.head())

########################################## save omic files


# Path to your mapping file
mapping_file = "tcga_brca_multiomics_mapping.csv"
df_map = pd.read_csv(mapping_file)

# Create output directories
# for omic in ["mrna", "mirna", "methylation"]:
#     os.makedirs('../data/tcga_brca/'+omic, exist_ok=True)

# GDC API endpoint
GDC_FILES_API = "https://api.gdc.cancer.gov/files"

# Function to fetch files by patient_id and data_type
def fetch_gdc_files(patient_ids, data_category, output_dir):
    """
    patient_ids: list of patient barcodes (first 12 characters)
    data_category: 'Transcriptome Profiling', 'DNA Methylation', etc.
    output_dir: folder to save files
    """
    # Query parameters
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": patient_ids}},
            {"op": "in", "content": {"field": "files.data_category", "value": [data_category]}},
            {"op": "in", "content": {"field": "files.experimental_strategy",
                                     "value": ["RNA-Seq", "miRNA-Seq", "Methylation array"]}}
        ]
    }

    params = {
        "filters": str(filters).replace("'", '"'),  # API requires double quotes
        "format": "JSON",
        "fields": "file_id,file_name,cases.submitter_id",
        "size": 5000
    }

    response = requests.get(GDC_FILES_API, params=params)
    data = response.json()["data"]["hits"]

    # Download each file
    for f in data:
        file_id = f["file_id"]
        patient = f["cases"][0]["submitter_id"][:12]
        file_name = f["file_name"]
        out_path = os.path.join(output_dir, f"{patient}_{file_name}")

        download_url = f"https://api.gdc.cancer.gov/data/{file_id}"
        r = requests.get(download_url, stream=True)
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    print(f"Downloaded {len(data)} files for {data_category}.")


# Get unique patient IDs
patient_ids = df_map["patient_id"].str[:12].unique().tolist()

# Fetch each omic
fetch_gdc_files(patient_ids, "Transcriptome Profiling", "../data/tcga_brca/mrna")
fetch_gdc_files(patient_ids, "Transcriptome Profiling", "../data/tcga_brca/mirna")


#######################################

def fetch_methylation_files(patient_ids, output_dir):
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": patient_ids}},
            {"op": "in", "content": {"field": "files.data_category", "value": ["DNA Methylation"]}},
            {"op": "in", "content": {"field": "files.experimental_strategy", "value": ["Methylation Array"]}}
        ]
    }
    params = {
        "filters": str(filters).replace("'", '"'),
        "format": "JSON",
        "fields": "file_id,file_name,cases.submitter_id",
        "size": 5000
    }

    response = requests.get(GDC_FILES_API, params=params)
    data = response.json()["data"]["hits"]
    print(len(data))

    for f in data:
        file_id = f["file_id"]
        patient = f["cases"][0]["submitter_id"][:12]
        file_name = f["file_name"]
        out_path = os.path.join(output_dir, f"{patient}_{file_name}")
        download_url = f"https://api.gdc.cancer.gov/data/{file_id}"
        r = requests.get(download_url, stream=True)
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)

    print(f"Downloaded {len(data)} methylation files.")

patient_ids = df_map["patient_id"].str[:12].unique().tolist()
fetch_methylation_files(patient_ids,  "../data/tcga_brca/methylation")

####################################################

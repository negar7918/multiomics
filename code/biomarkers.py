biomarkers = {'hsa-mir-488': 154, 'hsa-mir-3622a': 150, 'hsa-mir-577': 147, 'hsa-mir-190b': 145, 'hsa-mir-499': 143, 'hsa-mir-129-1': 137, 'hsa-mir-216a': 134, 'hsa-mir-663': 134, 'hsa-mir-135b': 132, 'hsa-mir-518f': 129, 'hsa-mir-944': 128, 'hsa-mir-551a': 127, 'hsa-mir-522': 127, 'hsa-mir-512-2': 126, 'hsa-mir-30a': 125, 'hsa-mir-548o': 124, 'hsa-mir-934': 122,
              'hsa-mir-138-2': 121, 'hsa-mir-376a-2': 121, 'hsa-mir-3662': 120, 'hsa-mir-133b': 119, 'hsa-mir-320e': 119, 'hsa-mir-516a-2': 118, 'hsa-mir-618': 114, 'hsa-mir-1295': 114, 'hsa-mir-105-1': 114, 'hsa-mir-204': 112, 'hsa-mir-1179': 112, 'hsa-mir-1294': 110, 'hsa-mir-3613': 109, 'hsa-mir-143': 109, 'hsa-mir-135a-2': 107, 'hsa-mir-3922': 106,
              'hsa-mir-9-3': 104, 'hsa-mir-1262': 104, 'hsa-mir-1276': 103, 'hsa-mir-580': 102, 'hsa-mir-935': 102, 'hsa-mir-31': 101, 'hsa-mir-765': 101, 'hsa-mir-1-2': 101, 'hsa-mir-3619': 101, 'hsa-mir-4286': 100, 'hsa-mir-145': 100, 'hsa-mir-589': 100, 'hsa-mir-584': 100,
              'hsa-mir-362': 99, 'hsa-mir-1538': 99, 'hsa-mir-18b': 98,
              'hsa-mir-33b': 97, 'hsa-mir-17': 97, 'hsa-mir-3667': 97, 'hsa-mir-449b': 97, 'hsa-mir-129-2': 97, 'hsa-mir-2116': 97, 'hsa-mir-3065': 97, 'hsa-mir-744': 97, 'hsa-mir-146a': 94, 'hsa-mir-10a': 94, 'hsa-mir-520b': 94, 'hsa-mir-636': 93, 'hsa-mir-570': 93, 'hsa-mir-92a-2': 91, 'hsa-mir-3614': 91, 'hsa-mir-548b': 91, 'hsa-mir-383': 91, 'hsa-mir-3620': 90, 'hsa-mir-519a-1': 89, 'hsa-mir-301b': 89, 'hsa-mir-3157': 89, 'hsa-mir-24-2': 89, 'hsa-mir-561': 88, 'hsa-mir-181a-2': 88, 'hsa-mir-585': 88, 'hsa-mir-3117': 87, 'hsa-mir-548s': 87, 'hsa-mir-154': 85, 'hsa-mir-1185-2': 85, 'hsa-mir-525': 85, 'hsa-mir-1245': 85, 'hsa-mir-545': 85, 'hsa-mir-329-2': 84, 'hsa-mir-378': 84, 'hsa-mir-34b': 84, 'hsa-mir-130b': 83, 'hsa-mir-429': 82, 'hsa-mir-504': 81, 'hsa-mir-548k': 81, 'hsa-mir-330': 81, 'hsa-mir-135a-1': 81, 'hsa-mir-642a': 81, 'hsa-mir-3150b': 79, 'hsa-mir-877': 79, 'hsa-mir-517b': 79, 'hsa-mir-3188': 78, 'hsa-mir-95': 78, 'hsa-mir-30e': 78, 'hsa-mir-3187': 78, 'hsa-mir-655': 77, 'hsa-mir-149': 77, 'hsa-mir-607': 77, 'hsa-mir-3074': 77, 'hsa-mir-3176': 76, 'hsa-mir-1258': 76, 'hsa-mir-592': 75, 'hsa-mir-9-2': 75, 'hsa-mir-1304': 75, 'hsa-let-7a-1': 75, 'hsa-mir-3940': 75, 'hsa-mir-486': 74, 'hsa-mir-100': 74, 'hsa-mir-378c': 73, 'hsa-mir-520a': 73, 'ISL2|64843': 72, 'hsa-mir-767': 72, 'hsa-mir-551b': 72, 'hsa-let-7a-2': 72, 'hsa-mir-579': 72, 'hsa-mir-376a-1': 72, 'hsa-mir-320d-2': 72, 'hsa-mir-939': 72, 'hsa-mir-1255a': 72, 'hsa-mir-126': 71, 'hsa-mir-155': 71, 'hsa-mir-3680': 71, 'hsa-mir-548t': 71}

import matplotlib.pyplot as plt
import numpy as np

# plt.figure(figsize=(50, 130))
# plt.margins(y=0)
# plt.yticks(fontsize=70)
# plt.xticks(fontsize=50)
# plt.barh(list(biomarkers.keys())[::-1], list(biomarkers.values())[::-1])
# for k in list(biomarkers.keys()):
#     if k in ['hsa-mir-145', 'hsa-mir-155', 'hsa-mir-429', 'hsa-mir-100', 'hsa-mir-204', 'hsa-mir-488', 'hsa-mir-577', 'hsa-mir-190b', 'hsa-mir-129-1', 'hsa-mir-216a', 'hsa-mir-135b', 'hsa-mir-944', 'hsa-mir-522', 'hsa-mir-30a', 'hsa-mir-934', 'hsa-mir-135a-2', 'hsa-let-7a-1',
#              'hsa-mir-3662', 'hsa-mir-133b', 'hsa-mir-9-3', 'hsa-mir-1262', 'hsa-mir-31', 'hsa-mir-589', 'hsa-mir-18b', 'hsa-mir-17', 'hsa-mir-146a', 'hsa-mir-126',
#              ]:
#         plt.barh(k, biomarkers[k], color='red')
# plt.xlabel("Frequency", fontsize=50)
# plt.tight_layout()
# plt.savefig('./brca_biomarkers.pdf')
##########################################
# plt.figure(figsize=(30, 10))
# plt.margins(y=.05, x=.01)
# plt.yticks(fontsize=30)
# highlight = {
#     'hsa-mir-145', 'hsa-mir-155', 'hsa-mir-429', 'hsa-mir-100',
#     'hsa-mir-204', 'hsa-mir-488', 'hsa-mir-577', 'hsa-mir-190b',
#     'hsa-mir-129-1', 'hsa-mir-216a', 'hsa-mir-135b', 'hsa-mir-944',
#     'hsa-mir-522', 'hsa-mir-30a', 'hsa-mir-934', 'hsa-mir-135a-2',
#     'hsa-let-7a-1', 'hsa-mir-3662', 'hsa-mir-133b', 'hsa-mir-9-3',
#     'hsa-mir-1262', 'hsa-mir-31', 'hsa-mir-589', 'hsa-mir-18b',
#     'hsa-mir-17', 'hsa-mir-146a', 'hsa-mir-126'
# }
#
# x = list(biomarkers.keys())
# y = list(biomarkers.values())
#
# colors = ['red' if k in highlight else 'blue' for k in x]
#
# plt.scatter(x, y, marker='^', s=400, c=colors)
# plt.xticks([])
# plt.ylabel("Importance", fontsize=40)
# plt.xlabel("Ranking", fontsize=40)
# plt.tight_layout()
# plt.savefig('./brca_biomarkers_rank.pdf')
##########################################

import pandas as pd

# Your highlight set
highlight = [
    'hsa-mir-145', 'hsa-mir-155', 'hsa-mir-429', 'hsa-mir-100',
    'hsa-mir-204', 'hsa-mir-488', 'hsa-mir-577', 'hsa-mir-190b',
    'hsa-mir-129-1', 'hsa-mir-216a', 'hsa-mir-135b', 'hsa-mir-944',
    'hsa-mir-522', 'hsa-mir-30a', 'hsa-mir-934', 'hsa-mir-135a-2',
    'hsa-let-7a-1', 'hsa-mir-3662', 'hsa-mir-133b', 'hsa-mir-9-3',
    'hsa-mir-1262', 'hsa-mir-31', 'hsa-mir-589', 'hsa-mir-18b',
    'hsa-mir-17', 'hsa-mir-146a', 'hsa-mir-126'
]

# Convert biomarkers dict to DataFrame
df = pd.DataFrame([(k, biomarkers[k]) for k in highlight], columns=['miRNA', 'Importance'])

# Add highlight column
df['miRNA'].apply(lambda x: biomarkers[x])

# Sort by importance descending
df_sorted = df.sort_values(by='Importance', ascending=False)

plt.subplot(121)

cell_text = []
for row in range(len(df_sorted)):
    cell_text.append(df_sorted.iloc[row])

plt.figure(figsize=(3, len(df_sorted)/2))
tbl = plt.table(cellText=cell_text, colLabels=df_sorted.columns, loc='center')

for (row, col), cell in tbl.get_celld().items():
    if row > 0 and col == 0:
        cell.get_text().set_color('red')
    if col == 1:
        cell.get_text().set_ha('center')

plt.axis('off')
plt.savefig('./brca_biomarkers_sota.pdf')

#############################################

highlight = {
    'hsa-mir-145', 'hsa-mir-155', 'hsa-mir-429', 'hsa-mir-100',
    'hsa-mir-204', 'hsa-mir-488', 'hsa-mir-577', 'hsa-mir-190b',
    'hsa-mir-129-1', 'hsa-mir-216a', 'hsa-mir-135b', 'hsa-mir-944',
    'hsa-mir-522', 'hsa-mir-30a', 'hsa-mir-934', 'hsa-mir-135a-2',
    'hsa-let-7a-1', 'hsa-mir-3662', 'hsa-mir-133b', 'hsa-mir-9-3',
    'hsa-mir-1262', 'hsa-mir-31', 'hsa-mir-589', 'hsa-mir-18b',
    'hsa-mir-17', 'hsa-mir-146a', 'hsa-mir-126'
}

from collections import defaultdict

for k in highlight:
    biomarkers.pop(k, None)   # safe: no KeyError

grouped = defaultdict(list)

for k, v in biomarkers.items():
    grouped[v].append(k)

first_elements = {k: v[0] for k, v in grouped.items() if v}

# Convert biomarkers dict to DataFrame
df = pd.DataFrame([(k, first_elements[k]) for k in first_elements.keys()], columns=['Importance', 'miRNA'])

# Sort by importance descending
df_sorted = df.sort_values(by='Importance', ascending=False)

df_swapped = df_sorted.iloc[:, [1, 0]]

plt.subplot(121)

cell_text = []
for row in range(len(df_swapped)):
    cell_text.append(df_swapped.iloc[row])

plt.figure(figsize=(3, len(df_sorted)/2))
tbl = plt.table(cellText=cell_text, colLabels=df_swapped.columns, loc='center')

tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

for (row, col), cell in tbl.get_celld().items():
    cell.PAD = .005
    if row > 0 and row < 27 and col == 0:
        cell.get_text().set_color('blue')
        cell.get_text().set_ha('center')
    if row >= 27:
        cell.set_visible(False)
    if col == 1:
        cell.set_width(0.5)
        cell.get_text().set_ha('center')

plt.axis('off')
plt.savefig('./brca_biomarkers_new.pdf')

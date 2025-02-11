from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import KMeans
import multiomics.code.evaluation as evaluation


def plot_tsne(data, label, name, method):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    g = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df,
        legend="full",
        alpha=0.7
    )
    for t, l in zip(g.legend_.texts, ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]):
        t.set_text(l)
    plt.title("t-SNE on "+method, fontsize=20)
    plt.xlabel("tsne-2d-one", fontsize=20)
    plt.ylabel("tsne-2d-two", fontsize=20)
    plt.savefig('../plots/BRCA/'+name+'_tsne.png')
    plt.close()


def plot_corr(data, label, path, method):
    c1, c2, c3, c4, c5 = [], [], [], [], []
    plt.figure(figsize=(10, 4))
    for i in range(len(data[:, 0])):
        d = data[i, :-32]
        if label[i] == 0:
            c1.append(d)
        elif label[i] == 1:
            c2.append(d)
        elif label[i] == 2:
            c3.append(d)
        elif label[i] == 3:
            c4.append(d)
        else:
            c5.append(d)

    fig2, axes2 = plt.subplots(5, 3, figsize=(6, 8), gridspec_kw={'width_ratios': [2, 2, 2]})

    tmp1 = np.array(c1)
    mrna = pd.DataFrame(tmp1[:, :32])
    sns.heatmap(mrna.corr(), ax=axes2[0, 0], cmap='viridis', cbar=False)
    axes2[0, 0].set_title('mRNA')
    axes2[0, 0].set_ylabel('Luminal A', fontsize=8, labelpad=20)
    axes2[0, 0].set_xticks([])
    meth = pd.DataFrame(tmp1[:, 32:64])
    sns.heatmap(meth.corr(), ax=axes2[0, 1], cmap='viridis', cbar=False)
    axes2[0, 1].set_title('DNAMeth')
    axes2[0, 1].set_xticks([])
    axes2[0, 1].set_yticks([])
    mirna = pd.DataFrame(tmp1[:, 64:])
    sns.heatmap(mirna.corr(), ax=axes2[0, 2], cmap='viridis', cbar=False)
    axes2[0, 2].set_title('miRNA')
    axes2[0, 2].set_xticks([])
    axes2[0, 2].set_yticks([])

    tmp2 = np.array(c2)
    mrna = pd.DataFrame(tmp2[:, :32])
    sns.heatmap(mrna.corr(), ax=axes2[1, 0], cmap='viridis', cbar=False)
    axes2[1, 0].set_ylabel('Luminal B', fontsize=8, labelpad=20)
    axes2[1, 0].set_xticks([])
    meth = pd.DataFrame(tmp2[:, 32:64])
    sns.heatmap(meth.corr(), ax=axes2[1, 1], cmap='viridis', cbar=False)
    axes2[1, 1].set_xticks([])
    axes2[1, 1].set_yticks([])
    mirna = pd.DataFrame(tmp2[:, 64:])
    sns.heatmap(mirna.corr(), ax=axes2[1, 2], cmap='viridis', cbar=False)
    axes2[1, 2].set_xticks([])
    axes2[1, 2].set_yticks([])

    tmp3 = np.array(c3)
    mrna = pd.DataFrame(tmp3[:, :32])
    sns.heatmap(mrna.corr(), ax=axes2[2, 0], cmap='viridis', cbar=False)
    axes2[2, 0].set_ylabel('Normal-like', fontsize=8, labelpad=20)
    axes2[2, 0].set_xticks([])
    meth = pd.DataFrame(tmp3[:, 32:64])
    sns.heatmap(meth.corr(), ax=axes2[2, 1], cmap='viridis', cbar=False)
    axes2[2, 1].set_xticks([])
    axes2[2, 1].set_yticks([])
    mirna = pd.DataFrame(tmp3[:, 64:])
    sns.heatmap(mirna.corr(), ax=axes2[2, 2], cmap='viridis', cbar=False)
    axes2[2, 2].set_xticks([])
    axes2[2, 2].set_yticks([])

    tmp4 = np.array(c4)
    mrna = pd.DataFrame(tmp4[:, :32])
    sns.heatmap(mrna.corr(), ax=axes2[3, 0], cmap='viridis', cbar=False)
    axes2[3, 0].set_ylabel('Basal-like', fontsize=8, labelpad=20)
    axes2[3, 0].set_xticks([])
    meth = pd.DataFrame(tmp4[:, 32:64])
    sns.heatmap(meth.corr(), ax=axes2[3, 1], cmap='viridis', cbar=False)
    axes2[3, 1].set_xticks([])
    axes2[3, 1].set_yticks([])
    mirna = pd.DataFrame(tmp4[:, 64:])
    sns.heatmap(mirna.corr(), ax=axes2[3, 2], cmap='viridis', cbar=False)
    axes2[3, 2].set_xticks([])
    axes2[3, 2].set_yticks([])

    tmp5 = np.array(c5)
    mrna = pd.DataFrame(tmp5[:, :32])
    sns.heatmap(mrna.corr(), ax=axes2[4, 0], cmap='viridis', cbar=False)
    axes2[4, 0].set_ylabel('HER2-enriched', fontsize=8, labelpad=20)
    meth = pd.DataFrame(tmp5[:, 32:64])
    sns.heatmap(meth.corr(), ax=axes2[4, 1], cmap='viridis', cbar=False)
    axes2[4, 1].set_yticks([])
    mirna = pd.DataFrame(tmp5[:, 64:])
    sns.heatmap(mirna.corr(), ax=axes2[4, 2], cmap='viridis', cbar=False)
    axes2[4, 2].set_yticks([])

    fig2.suptitle('correlation of embeddings of ' + method, fontsize=15)
    plt.savefig(path + '_vae_corr.pdf', dpi=300)
    plt.close()

def plot_with_path(data, label, path, method):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    g1 = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df,
        legend="full",
        alpha=0.7
    )
    if 'brca' in path:
        for t, l in zip(g1.legend_.texts, ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]):
            t.set_text(l)
    else:
        for t, l in zip(g1.legend_.texts, ["type I", "type II", "type III", "type IV"]):
            t.set_text(l)
    plt.title("t-SNE on "+method, fontsize=20)
    plt.savefig(path+'_tsne.png')
    plt.close()

    embedding = umap.UMAP(random_state=42).fit_transform(data)
    df = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    df['umap-2d-one'] = embedding[:, 0]
    df['umap-2d-two'] = embedding[:, 1]

    g2= sns.scatterplot(
        x="umap-2d-one", y="umap-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df,
        legend="full",
        alpha=0.8
    )
    if 'brca' in path:
        for t, l in zip(g2.legend_.texts, ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]):
            t.set_text(l)
    else:
        for t, l in zip(g2.legend_.texts, ["type I", "type II", "type III", "type IV"]):
            t.set_text(l)

    plt.title("UMAP on "+method, fontsize=20)
    plt.savefig(path+ '_umap.png')
    plt.close()

    # c1, c2, c3, c4, c5 = [], [], [], [], []
    # plt.figure(figsize=(10, 4))
    # fig, axes = plt.subplots(2, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 2, 2]})
    # for i in range(len(data[:, 0])):
    #     d = data[i, :-32]
    #     if label[i] == 0:
    #         c1.append(d)
    #     elif label[i] == 1:
    #         c2.append(d)
    #     elif label[i] == 2:
    #         c3.append(d)
    #     elif label[i] == 3:
    #         c4.append(d)
    #     else:
    #         c5.append(d)
    # pos, names = np.linspace(0, 96, 4), ['mRNA', 'DNAMeth', 'miRNA']
    # position = pos[:-1] + np.diff(pos) / 2
    # axes[0, 0].scatter(np.arange(len(d)), np.mean(c1, axis=0), color='blue', alpha=0.5)
    # axes[0, 0].set_xticks(position)
    # axes[0, 0].set_xticklabels(names)
    # axes[0, 1].scatter(np.arange(len(d)), np.mean(c2, axis=0), color='#FF00FF', alpha=0.5) # magenta
    # axes[0, 1].set_xticks(position)
    # axes[0, 1].set_xticklabels(names)
    # axes[0, 2].scatter(np.arange(len(d)), np.mean(c3, axis=0), color='red', alpha=0.5)
    # axes[0, 2].set_xticks(position)
    # axes[0, 2].set_xticklabels(names)
    # axes[1, 0].scatter(np.arange(len(d)), np.mean(c4, axis=0), color='#808000', alpha=0.5) # olive green
    # axes[1, 0].set_xticks(position)
    # axes[1, 0].set_xticklabels(names)
    # axes[1, 1].scatter(np.arange(len(d)), np.mean(c5, axis=0), color='#00FF66', alpha=0.5) # phosphorous
    # axes[1, 1].set_xticks(position)
    # axes[1, 1].set_xticklabels(names)
    # axes[1][2].set_visible(False)
    # fig.suptitle('mean embeddings of '+method, fontsize=20)
    # plt.savefig(path + '_vae.png')
    # plt.close()

def plot_sim(data, path, name):

    import torch.nn.functional as F
    import torch
    # Compute cosine similarity matrix
    d = torch.from_numpy(data)
    similarity_matrix = F.cosine_similarity(d.unsqueeze(1), d.unsqueeze(0), dim=2)
    # Convert to NumPy for visualization
    sim_matrix_np = similarity_matrix.cpu().numpy()

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix_np, cmap="crest")
    plt.title("Cosine Similarity Heatmap Of " + name )
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.savefig(path)
    plt.close()



    # plt.figure(figsize=(10, 4))
    # for i in range(len(data[:,0])):
    #     if label[i] == 0:
    #         c = 'red'
    #     elif label[i] == 1:
    #         c = 'green'
    #     elif label[i] == 2:
    #         c = 'blue'
    #     elif label[i] == 3:
    #         c = 'yellow'
    #     else:
    #         c = 'black'
    #     plt.scatter(np.arange(len(data[i])), data[i], color=c, alpha=0.5)
    # plt.title(method, fontsize=20)
    # plt.savefig(path + '_vae.png')
    # plt.close()


def plot_umap_seq(rmrna, data, label, path, name, method):
    embedding = umap.UMAP(random_state=42).fit_transform(data)
    df = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    chunk = 25
    df['umap-2d-one'] = embedding[83:90, 0]
    df['umap-2d-two'] = embedding[83:90, 1]

    g= sns.scatterplot(
        x="umap-2d-one", y="umap-2d-two",
        hue=label[83:90].flatten(),
        palette=sns.color_palette("hls", len(np.unique(label[83:90]))),
        data=df,
        legend="full",
        alpha=0.7
    )
    for t, l in zip(g.legend_.texts, ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]):
        t.set_text(l)

    for i, txt in enumerate(label[83:90].tolist()):
        g.text(df['umap-2d-one'][i], df['umap-2d-two'][i], i+83)
    plt.title("UMAP on "+method, fontsize=20)
    plt.xlabel("umap-2d-one", fontsize=20)
    plt.ylabel("umap-2d-two", fontsize=20)
    plt.savefig(path + name + '_umap_seq.png')
    plt.close()
    # fig = plt.figure()
    # diff_genes = np.argsort(rmrna[2] - rmrna[19]) # 2, 19: close points (1, 18)
    # diff = np.sort(rmrna[2] - rmrna[19])
    # #plt.plot(np.arange(1000), diff, label='diff samples 1 and 18 w.r.t. genes')
    # diff_em = np.sort(data[2] - data[19])
    # #plt.plot(np.arange(32), diff_em, label='diff samples 1 and 18 w.r.t. embedding')
    #
    # diff_genes = np.argsort(rmrna[9] - rmrna[22])  # 9, 22: far points (8, 21)
    # diff = np.sort(rmrna[9] - rmrna[22]) #* 11
    # plt.plot(np.arange(1000), diff, label='diff samples 8 and 21 w.r.t. genes')
    # diff_em = np.sort(data[9] - data[22]) #* 11
    # plt.plot(np.arange(32), diff_em, label='diff samples 8 and 21 w.r.t. embedding')
    #
    # diff_genes = np.argsort(rmrna[16] - rmrna[22])  # 9, 22: far points (15, 21)
    # diff = np.sort(rmrna[16] - rmrna[22])  # * 11
    # plt.plot(np.arange(1000), diff, label='diff samples 15 and 21 w.r.t. genes')
    # diff_em = np.sort(data[16] - data[22])  # * 11
    # plt.plot(np.arange(32), diff_em, label='diff samples 15 and 21 w.r.t. embedding')
    #
    # fig.legend()
    # fig.set_figwidth(20)
    # plt.savefig(path + name + '_umap_seq2.png')


def plot_umap(data, label, name, method):
    embedding = umap.UMAP(random_state=42).fit_transform(data)
    df = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    df['umap-2d-one'] = embedding[:, 0]
    df['umap-2d-two'] = embedding[:, 1]

    g= sns.scatterplot(
        x="umap-2d-one", y="umap-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df,
        legend="full",
        alpha=0.8
    )
    for t, l in zip(g.legend_.texts, ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]):
        t.set_text(l)
    #plt.title("UMAP on "+method, fontsize=20)
    #plt.xlabel("umap-2d-one", fontsize=20)
    #plt.ylabel("umap-2d-two", fontsize=20)
    plt.savefig('../plots/BRCA/' + name + '_umap.png')
    plt.close()


def plot_umap_kidney(data, label, name, method, mirna):
    embedding = umap.UMAP(random_state=42).fit_transform(data)
    df = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    df['umap-2d-one'] = embedding[:, 0]
    df['umap-2d-two'] = embedding[:, 1]

    g1= sns.scatterplot(
        x="umap-2d-one", y="umap-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df,
        legend="full",
        alpha=0.5
    )
    for t, l in zip(g1.legend_.texts, ["KIRP", "KIRC"]):
        t.set_text(l)

    g1.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # from https://www.nature.com/articles/s41598-020-71997-6/tables/1 we have:
    # 175th feature in miRNA is hsa-mir-122 which is a biomarker for kidney cancer
    # 97th feature in miRNA is hsa-mir-155 which is a biomarker for kidney cancer
    # df2 = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two']) # 58, 54
    # df2['umap-2d-one'] = mirna[:, 56]
    # df2['umap-2d-two'] = mirna[:, 52]
    # g2 = sns.scatterplot(
    #     x="umap-2d-one", y="umap-2d-two",
    #     hue=label,
    #     palette=sns.color_palette("hls", len(np.unique(label))),
    #     data=df2,
    #     legend="full",
    #     alpha=0.5
    # )
    # for t, l in zip(g2.legend_.texts, ["KIRP", "KIRC"]):
    #     t.set_text(l)
    # g2.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    plt.title("UMAP on kidney cancers using "+method, fontsize=15)
    plt.tight_layout()
    plt.savefig('../plots/kidney/' + name + '_umap.png')
    plt.close()


def plot_umap_coad(data, label, name, method, mrna, view):
    embedding = umap.UMAP(random_state=42).fit_transform(data)
    df = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    df['umap-2d-one'] = embedding[:, 0]
    df['umap-2d-two'] = embedding[:, 1]

    g1 = sns.scatterplot(
        x="umap-2d-one", y="umap-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df,
        legend="full",
        alpha=0.5
    )
    for t, l in zip(g1.legend_.texts, ["CIN", "GS", "MSI", "POLE"]):
        t.set_text(l)

    for i, txt in enumerate(label.tolist()):
        g1.text(df['umap-2d-one'][i], df['umap-2d-two'][i], i)

    names = np.loadtxt('../../dataset/COAD/COAD_mRNA.csv', delimiter=',', dtype=str)
    genes = names[1:, 0]
    top_genes = {}
    for i, txt in enumerate(label.tolist()):
        ind = np.argpartition(mrna[i], -20)[-20:]
        top_genes[i] = genes[ind]

    if view == 1:
        view = "final"
    elif view == 2:
        view = "mrna"
    elif view == 3:
        view = "methyl"
    else:
        view = "mirna"
    with open('../plots/COAD/top_genes_{}.txt'.format(view), 'w') as f:
        for key in top_genes.keys():
            f.write("%s,%s\n" % (key, top_genes[key]))

    #g1.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    plt.title("UMAP on colon cancers using " + method, fontsize=15)
    #plt.tight_layout()
    plt.xlabel("umap-2d-one", fontsize=20)
    plt.ylabel("umap-2d-two", fontsize=20)
    plt.savefig('../plots/COAD/' + name + '_umap.png')
    plt.close()


def plot_umap_kirc(data, label, name, method):
    label = np.array(label[:, 0], dtype=int)

    embedding = umap.UMAP(random_state=42).fit_transform(data)
    df = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    df['umap-2d-one'] = embedding[:, 0]
    df['umap-2d-two'] = embedding[:, 1]

    g1= sns.scatterplot(
        x="umap-2d-one", y="umap-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df,
        legend="full",
        alpha=0.5
    )
    for t, l in zip(g1.legend_.texts, ["0", "1"]):
        t.set_text(l)

    g1.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    plt.title("UMAP on KIRC using "+method, fontsize=15)
    plt.tight_layout()
    plt.savefig('../plots/KIRC/' + name + '_umap.png')
    plt.close()


def plot_umap_luad(data, label, name, method, mrna):
    label = np.array(label[:, 2], dtype=int)

    embedding = umap.UMAP(random_state=42).fit_transform(data)
    df = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    df['umap-2d-one'] = embedding[:, 0]
    df['umap-2d-two'] = embedding[:, 1]

    g1= sns.scatterplot(
        x="umap-2d-one", y="umap-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df,
        legend="full",
        alpha=0.5
    )
    for t, l in zip(g1.legend_.texts, ["0", "1"]):
        t.set_text(l)

    g1.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    embedding2 = umap.UMAP(random_state=42).fit_transform(mrna)
    df2 = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    df2['umap-2d-one'] = embedding2[:, 0]
    df2['umap-2d-two'] = embedding2[:, 1]

    g2 = sns.scatterplot(
        x="umap-2d-one", y="umap-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df2,
        legend="full",
        alpha=0.5
    )
    for t, l in zip(g1.legend_.texts, ["0", "1"]):
        t.set_text(l)

    #g2.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    plt.title("UMAP on LUAD using "+method, fontsize=15)
    plt.tight_layout()
    plt.savefig('../plots/LUAD/' + name + '_umap.png')
    plt.close()


def plot_umap_with_stage(data, label, name, method):
    # LIHC
    label = label.T
    stage = label[0]
    l = label[1]
    stages = {'Stage I':10 , 'Stage II':15, 'Stage III':20, 'Stage IIIA':25, 'Stage IIIB':30,
              'Stage IIIC':35, 'Stage IV':40, 'Stage IVA':45, 'Stage IVB':50}
    sizes = []
    colors = []
    i = 0
    for s in stage:
        sizes = np.append(sizes, stages[s])
        if l[i] == 0:
            c = 'red'
        else:
            c = 'blue'
        colors = np.append(colors, c)
        i += 1

    embedding = umap.UMAP(random_state=42).fit_transform(data)
    df = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    df['umap-2d-one'] = embedding[:, 0]
    df['umap-2d-two'] = embedding[:, 1]

    fig, axes = plt.subplots(2, 1)

    g1= sns.scatterplot(
        ax=axes[0],
        x="umap-2d-one", y="umap-2d-two",
        hue=l,
        palette=sns.color_palette("hls", len(np.unique(l))),
        data=df,
        legend="full",
        alpha=0.5
    )
    for t, l in zip(g1.legend_.texts, ["1", "11"]):
        t.set_text(l)

    g1.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    g2 = axes[1].scatter(df['umap-2d-one'], df['umap-2d-two'] , c=colors, s=sizes, alpha=.4)
    # for t, l in zip(g2.legend_.texts, ['Stage I', 'Stage II', 'Stage III', 'Stage IIIA', 'Stage IIIB',
    #           'Stage IIIC', 'Stage IV', 'Stage IVA', 'Stage IVB']):
    #     t.set_text(l)

    # g2.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    plt.title("UMAP on LIHC using "+method, fontsize=15)
    plt.tight_layout()
    plt.savefig('../plots/LIHC/' + name + '_umap.png')
    plt.close()


def plot_tsne_genes(data, gene_labels, name, method):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    g = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        data=df,
        legend="full",
        alpha=0.7
    )
    for i, txt in enumerate(gene_labels.tolist()):
        if txt in [#'SOX11', 'AMY1A', 'SLC6A15', 'FABP7', 'SLC6A14', 'SLC6A2', 'FGFBP1', 'DSG1', 'UGT8', 'ANKRD45', 'PI3'
                   #, 'SERPINB5', 'COL11A2', 'ARHGEF4', 'SOX10',
                   'KRT5', 'KRT14', 'KRT17', 'EGFR', 'KIT', 'CRYAB', 'CAV1', 'CAV2', 'MSN', 'ITGB4',  # high on basel
                    'ESR1', 'PGR', 'ERBB2']: # low on basel
            g.text(df['tsne-2d-one'][i], df['tsne-2d-two'][i], txt)
    plt.title("t-SNE on genes "+method, fontsize=20)
    plt.xlabel("tsne-2d-one", fontsize=20)
    plt.ylabel("tsne-2d-two", fontsize=20)
    plt.savefig('../plots/BRCA/'+name+'_tsne_genes.png')
    plt.close()


def plot_umap_genes(data, gene_labels, name, method):
    embedding = umap.UMAP(random_state=42).fit_transform(data)
    df = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    df['umap-2d-one'] = embedding[:, 0]
    df['umap-2d-two'] = embedding[:, 1]

    g = sns.scatterplot(
        x="umap-2d-one", y="umap-2d-two",
        data=df,
        legend="full",
        alpha=0.7
    )
    for i, txt in enumerate(gene_labels.tolist()):
        if txt in [#'SOX11', 'AMY1A', 'SLC6A15', 'FABP7', 'SLC6A14', 'SLC6A2', 'FGFBP1', 'DSG1', 'UGT8', 'ANKRD45', 'PI3'
                   #, 'SERPINB5', 'COL11A2', 'ARHGEF4', 'SOX10',
                   'KRT5', 'KRT14', 'KRT17', 'EGFR', 'KIT', 'CRYAB', 'CAV1', 'CAV2', 'MSN', 'ITGB4',  # high on basel
                    'ESR1', 'PGR', 'ERBB2']: # low on basel
            g.text(df['umap-2d-one'][i], df['umap-2d-two'][i], txt)
    plt.title("UMAP on genes "+method, fontsize=20)
    plt.xlabel("umap-2d-one", fontsize=20)
    plt.ylabel("umap-2d-two", fontsize=20)
    plt.savefig('../plots/BRCA/' + name + '_umap_genes.png')
    plt.close()


def plot(method, dir_prior, disease, final_embedding, view1_specific_em_new, view2_specific_em_new,
         view3_specific_em_new, view1_data_new, view2_data_new, view3_data_new, y_true, n_clusters):
    np.random.seed(42)

    if dir_prior:
        method = "Dirichlet " + method
    if disease == 'lihc':
        plot_umap_with_stage(final_embedding, y_true, method + "_final_em", method)
        plot_umap_with_stage(view1_specific_em_new.detach().numpy(), y_true, method + "_mRNA_em", method)
        plot_umap_with_stage(view2_specific_em_new.detach().numpy(), y_true, method + "_DNAMeth_em", method)
        plot_umap_with_stage(view3_specific_em_new.detach().numpy(), y_true, method + "_miRNA_em", method)
        import umap
        embedding = umap.UMAP(random_state=42).fit_transform(final_embedding)
        km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        y_pred = km.fit_predict(embedding)
        truth = np.array(y_true[:, 1].flatten(), dtype=int)
        nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth, y_pred)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))
    elif disease == 'brca2':
        plot_umap(final_embedding, y_true, method + "_final_em", method)
        plot_umap(view1_specific_em_new.detach().numpy(), y_true, method + "_mRNA_em", method)
        plot_umap(view2_specific_em_new.detach().numpy(), y_true, method + "_DNAMeth_em", method)
        plot_umap(view3_specific_em_new.detach().numpy(), y_true, method + "_miRNA_em", method)
        km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        import umap
        embedding = umap.UMAP(random_state=1).fit_transform(final_embedding)
        y_pred = km.fit_predict(embedding)
        truth = np.asarray(y_true, dtype=int)
        nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth, y_pred)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))
    elif disease == 'coad':
        plot_umap_coad(final_embedding, y_true, method + "_final_em", method, view1_data_new, 1)
        plot_umap_coad(view1_specific_em_new.detach().numpy(), y_true, method + "_mRNA_em", method, view1_data_new, 2)
        plot_umap_coad(view2_specific_em_new.detach().numpy(), y_true, method + "_DNAMeth_em", method, view1_data_new, 3)
        plot_umap_coad(view3_specific_em_new.detach().numpy(), y_true, method + "_miRNA_em", method, view1_data_new, 4)
        truth = np.asarray(y_true, dtype=int)
        import umap
        embedding = umap.UMAP(random_state=42).fit_transform(final_embedding)
        km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        y_pred = km.fit_predict(embedding)
        nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth, y_pred)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))
    elif disease == '5cancer':
        plot_umap_coad(final_embedding, y_true, method + "_final_em", method, view1_data_new, 1)
        plot_umap_coad(view1_specific_em_new.detach().numpy(), y_true, method + "_mRNA_em", method, view1_data_new, 2)
        plot_umap_coad(view2_specific_em_new.detach().numpy(), y_true, method + "_DNAMeth_em", method, view1_data_new, 3)
        plot_umap_coad(view3_specific_em_new.detach().numpy(), y_true, method + "_miRNA_em", method, view1_data_new, 4)
        truth = np.asarray(y_true, dtype=int)
        km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        y_pred = km.fit_predict(final_embedding)
        nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth, y_pred)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))
    elif disease == 'kirc':
        plot_umap_kirc(final_embedding, y_true, method + "_final_em", method)
        plot_umap_kirc(view1_specific_em_new.detach().numpy(), y_true, method + "_mRNA_em", method)
        plot_umap_kirc(view2_specific_em_new.detach().numpy(), y_true, method + "_DNAMeth_em", method)
        plot_umap_kirc(view3_specific_em_new.detach().numpy(), y_true, method + "_miRNA_em", method)
        km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        y_pred = km.fit_predict(final_embedding)
        truth = np.array(y_true[:, 1].flatten(), dtype=int)
        nmi_, ari_, f_score_, acc_, v_ = evaluation.evaluate(truth, y_pred)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f  <==|' % (
            nmi_, ari_, f_score_, acc_, v_))
    elif disease == 'kidney':
        plot_umap_kidney(final_embedding, y_true, method + "_final_em", method,
                              view3_data_new.detach().numpy())
        plot_umap_kidney(view1_specific_em_new.detach().numpy(), y_true, method + "_mRNA_em", method,
                              view3_data_new.detach().numpy())
        plot_umap_kidney(view2_specific_em_new.detach().numpy(), y_true, method + "_DNAMeth_em", method,
                              view3_data_new.detach().numpy())
        plot_umap_kidney(view3_specific_em_new.detach().numpy(), y_true, method + "_miRNA_em", method,
                              view3_data_new.detach().numpy())
        import umap
        embedding = umap.UMAP(random_state=1).fit_transform(final_embedding)
        km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        y_pred = km.fit_predict(embedding)
        truth = np.array(y_true, dtype=int)
        nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth, y_pred)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))
    elif disease == 'luad':
        plot_umap_luad(final_embedding, y_true, method + "_final_em", method, view1_data_new.detach().numpy())
        plot_umap_luad(view1_specific_em_new.detach().numpy(), y_true, method + "_mRNA_em", method,
                            view1_data_new.detach().numpy())
        plot_umap_luad(view2_specific_em_new.detach().numpy(), y_true, method + "_DNAMeth_em", method,
                            view1_data_new.detach().numpy())
        plot_umap_luad(view3_specific_em_new.detach().numpy(), y_true, method + "_miRNA_em", method,
                            view1_data_new.detach().numpy())
        import umap
        embedding = umap.UMAP(random_state=42).fit_transform(final_embedding)
        km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        y_pred = km.fit_predict(embedding)
        truth = np.array(y_true[:, 1].flatten(), dtype=int)
        nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth, y_pred)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))
    else:
        truth = y_true.flatten()
        plot_tsne(final_embedding, truth, method + "_final_em", method)
        plot_umap(final_embedding, truth, method + "_final_em", method)
        plot_tsne(view1_specific_em_new.detach().numpy(), truth, method + "_mRNA_em", method)
        plot_umap(view1_specific_em_new.detach().numpy(), truth, method + "_mRNA_em", method)
        plot_tsne(view2_specific_em_new.detach().numpy(), truth, method + "_DNAMeth_em", method)
        plot_umap(view2_specific_em_new.detach().numpy(), truth, method + "_DNAMeth_em", method)
        plot_tsne(view3_specific_em_new.detach().numpy(), truth, method + "_miRNA_em", method)
        plot_umap(view3_specific_em_new.detach().numpy(), truth, method + "_miRNA_em", method)
        np.save('../../final_embedding.npy', final_embedding)
        np.savetxt('../../dataset/final_embedding.txt', final_embedding)
        km = KMeans(n_clusters=n_clusters, random_state=42)
        y_pred = km.fit_predict(final_embedding)
        nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth, y_pred)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))




# # Set the random seed for reproducibility
# np.random.seed(42)
#
# # Generate some data
# data = np.random.rand(100, 2)
#
# # Run KMeans with a fixed random state
# kmeans = KMeans(n_clusters=3, random_state=42)
# labels = kmeans.fit_predict(data)
#
# # Print out the labels to ensure they are consistent
# print(labels)

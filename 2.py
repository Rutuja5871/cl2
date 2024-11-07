import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

genes = [f'gene_{i}' for i in range(1, 101)]
conditions = ['Control', 'Treatment']
samples = [f'sample_{i}' for i in range(1, 11)]

data = np.random.poisson(lam=20, size=(100, 10))

data[:5, 5:10] += 15

df = pd.DataFrame(data, index=genes, columns=samples)

metadata = pd.DataFrame({
    'sample': samples,
    'condition': ['Control'] * 5 + ['Treatment'] * 5
})

df_norm = df.div(df.sum(axis=0), axis=1) * 1e6
df_log = np.log2(df_norm + 1)

def differential_expression(df_log, metadata):
    results = []
    for gene in df_log.index:
        y = df_log.loc[gene].values
        X = pd.get_dummies(metadata['condition'], drop_first=True)
        X = sm.add_constant(X.astype(float))
        
        model = sm.OLS(y, X).fit()
        p_value = model.pvalues[1]
        results.append({'gene': gene, 'p_value': p_value})
        
    results_df = pd.DataFrame(results)
    results_df['adjusted_p_value'] = sm.stats.multipletests(results_df['p_value'], method='fdr_bh')[1]
    return results_df

results_df = differential_expression(df_log, metadata)

deg = results_df[results_df['adjusted_p_value'] < 0.05]

annotations = {
    'gene_1': 'Pathway A',
    'gene_2': 'Pathway B',
    'gene_3': 'Pathway C',
    'gene_4': 'Pathway D',
    'gene_5': 'Pathway E',
}

deg['annotation'] = deg['gene'].map(annotations).fillna('Unknown')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='gene', y='adjusted_p_value', hue='annotation', data=deg)
plt.axhline(y=0.05, color='r', linestyle='--')
plt.xlabel('Genes')
plt.ylabel('Adjusted P-Value')
plt.title('Differentially Expressed Genes')
plt.xticks(rotation=90)
plt.legend(title="Annotations")
plt.tight_layout()
plt.show()

deg.to_csv('differentially_expressed_genes.csv', index=False)

report = """
RNA-Seq Data Analysis Report
============================
Differentially Expressed Genes:
{}

Functional Annotations:
{}

Potential Biological Interpretations:
- The genes such as gene_1, gene_2, etc., are involved in pathways A, B, etc.
These pathways are important for understanding the effect of the treatment condition.
""".format(deg[['gene', 'adjusted_p_value']], deg[['gene', 'annotation']])

with open('RNASeq_Analysis_Report.txt', 'w') as f:
    f.write(report)

print("Analysis complete. Results saved to 'differentially_expressed_genes.csv' and 'RNASeq_Analysis_Report.txt'.")

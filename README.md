# 🧬 Integration of multiple gene expression with GeneCompete 🏆

GeneCompete is a tool to combine heterogeneous gene expression datasets to order gene importance.

## Web-Application
The easy-to-use web-based platform can be accessed through 👉
https://genecompete.streamlit.app/ 👈
> [!NOTE]
> We suggest using Python function for large datasets.

## Python functions

We proposed two python function.
- GeneCompete_Union [^1]
- GeneCompete_Intersect [^2]

Input | Description
 ------------ | ------------- 
table | **Gene expression data**: Multiple files where the first column is gene name. These data can be prepared by any tools.
name | **Column name**: The interested value that will be used as competing score (in the example is logFC).
method | **Ranking Method**: Select 'Win-loss', 'Massey', 'Colley', 'Keener', 'Elo', 'Markov', 'PageRank', or 'BiPagerank'
reg | **Regulation cases**: 'Up-regulation' or 'Down-regulation'
FC | **logFC threshold**: The large number of genes can consume computational time. Before ranking, datasets are filtered with *logFC > FC* in case of up-regulation and *logFC < -FC* for down-regulation.

- Installation
```!git clone https://github.com/panisajan/GeneCompete```
- Load data
```
import pandas as pd
dat1 = pd.read_csv("sample_data/dat1.csv", index_col=0)
dat2 = pd.read_csv("sample_data/dat2.csv", index_col=0)
dat3 = pd.read_csv("sample_data/dat3.csv", index_col=0)
dat4 = pd.read_csv("sample_data/dat4.csv", index_col=0)
dat1.head()
```
<img src='figure/Fig1.png' width="600">

## Example 1 (Union strategy):
```
from GeneCompete_Union import*

my_data = [dat1, dat3, dat4]
my_methods = ['Win-loss', 'Massey', 'BiPagerank']

score = GeneCompete_Union(table = my_data, name = 'logFC', method = my_methods, reg = 'Down-regulation', FC = 1)
score.head()
```
<img src='figure/Fig2.png'>

## Example 2 (Intersect strategy):
```
from GeneCompete_Intersect import*

my_data = [dat1, dat2, dat3, dat4]
my_methods = ['Win-loss', 'Keener', 'PageRank']

score = GeneCompete_Intersect(table = my_data, name = 'logFC', method = my_methods, reg = 'Up-regulation', FC = None)
score.head()
```
<img src='figure/Fig3.png'>

## Further Reading
[^1]: aa
[^2]: aa


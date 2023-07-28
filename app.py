import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
from rdkit.Chem.TemplateAlign import AlignMolToTemplate2D
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem import AllChem
import mols2grid
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import seaborn as sns
from ipywidgets import interact
from itertools import product
import useful_rdkit_utils as uru
from tqdm.auto import tqdm
import numpy as np

sns.set(rc={'figure.figsize': (10, 10)})
sns.set_style('whitegrid')
sns.set_context('talk')

#Code adapted from Pat Walters https://github.com/PatWalters/practical_cheminformatics_tutorials/blob/main/sar_analysis/free_wilson.ipynb

st.title('Free Wilson Analysis')
st.sidebar.header("Configuration")
input_filename = st.sidebar.file_uploader('Upload CSV File here. ')
if not input_filename:
    input_filename = "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/data/CHEMBL313_sel.smi"

data_load_state = st.sidebar.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
df = pd.read_csv(input_filename)
df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.subheader('Peek at 5 rows data')
st.write(df.head())
# print(df.select_dtypes('float64').columns)

# st.text('Autodetecting columns with numbers')
st.sidebar.selectbox('Please choose the column you want to run with free wilson', df.select_dtypes('float64').columns)
st.sidebar.selectbox('Please choose the SMILES column you want to run with free wilson', df.select_dtypes('object').columns)

# core_smiles = "c1ccc(C2CC3CCC(C2)N3)cc1"
# core_mol = Chem.MolFromSmiles(core_smiles)

core_smiles = st.sidebar.text_input('Input your Core SMILES here', placeholder="c1ccc(C2CC3CCC(C2)N3)cc1")
if not core_smiles:
    core_smiles = "c1ccc(C2CC3CCC(C2)N3)cc1"

try:
    core_mol = Chem.MolFromSmiles(core_smiles)
except:
    st.text('ERROR: Your SMILES string is not valid')
    st.error('This is an error')

Compute2DCoords(core_mol)
for mol in df.mol:
    AllChem.GenerateDepictionMatching2DStructure(mol,core_mol)
# raw_html = mols2grid.display(df_result, mapping={"smiles": "SMILES"})._repr_html_()

if st.button('Run Free Wilson'):
    st.subheader('All the molecules')
    raw_html= mols2grid.display(df,mol_col='mol',use_coords=True, prerender=True, substruct_highlight=False)._repr_html_()
    components.html(raw_html, width=900, height=550, scrolling=True)

    st.subheader('Find the cores in the data')
    match, miss = RGroupDecompose(core_mol,df.mol.values,asSmiles=True)
    print(f"{len(miss)} molecules were not matched")
    rgroup_df = pd.DataFrame(match)
    st.write(rgroup_df.head())
    core_df = pd.DataFrame({"mol" : [Chem.MolFromSmiles(x) for x in rgroup_df.Core.unique()]})
    cores_html = mols2grid.display(core_df,mol_col="mol")._repr_html_()
    components.html(cores_html, width=900, height=300, scrolling=True)

    st.subheader('Find R Groups and run Ridge regression')
    unique_list = []
    for r in rgroup_df.columns[1:]:
        num_rgroups = len(rgroup_df[r].unique())
        print(r,num_rgroups)
        unique_list.append(rgroup_df[r].unique())
    total_possible_products = np.prod([len(x) for x in unique_list])
    st.text(f"{total_possible_products:,} products possible")
    enc = OneHotEncoder(categories=unique_list,sparse=False)
    one_hot_mat = enc.fit_transform(rgroup_df.values[:,1:])
    train_X, test_X, train_y, test_y = train_test_split(one_hot_mat,df.pIC50)
    ridge = Ridge()
    ridge.fit(train_X,train_y)
    pred = ridge.predict(test_X)

    res_df = pd.DataFrame({'Exp_pIC50' : test_y, "Pred_pIC50": pred})
    r2 = r2_score(test_y,pred)
    fgrid = sns.lmplot(x='Exp_pIC50', y="Pred_pIC50", data=res_df)
    ax = fgrid.axes[0,0]
    ax.text(6,9,f"$R^2$={r2:.2f}");
    st.pyplot(fgrid.fig)

    def clear_sss_matches(mol_in):
        mol_in.__sssAtoms = []

    class RGroupAligner:
        def __init__(self):
            self.ref_mol = Chem.MolFromSmarts("[#0]-[!#0]")
            Compute2DCoords(self.ref_mol)
            _ = self.ref_mol.GetConformer(0)
            
        def align(self,mol_in):
            Compute2DCoords(mol_in)
            _ = mol_in.GetConformer(0)
            AlignMolToTemplate2D(mol_in,self.ref_mol,clearConfs=True)
            clear_sss_matches(mol_in)

    uru.rd_shut_the_hell_up()
    #alread_made_SMILES = set([Chem.MolToSmiles(x) for x in df.mol])
    rg_df_dict = {}
    rgroup_aligner = RGroupAligner()
    start = 0
    rgroup_names = rgroup_df.columns[1:]
    for rg,name in zip(enc.categories_,rgroup_names):
        rg_mol_list = [Chem.MolFromSmiles(x) for x in rg]
        _ = [rgroup_aligner.align(x) for x in rg_mol_list]
        coef_list = ridge.coef_[start:start+len(rg)]
        start += len(rg)
        rg_df = pd.DataFrame({"mol": rg_mol_list, "coef": coef_list})
        rg_df.sort_values("coef",inplace=True)
        rg_df_dict[name] = rg_df
    
    r_group_render = st.selectbox('Choose R Group', rg_df_dict.keys())#, on_change=display_mols)
    rs_html= mols2grid.display(rg_df_dict[r_group_render],mol_col="mol",
                    use_coords=True, prerender=True, substruct_highlight=False,
                    subset=["img","coef"],
                    transform={"coef" : lambda a: f"{a:.2f}"},
                    style={"coef": lambda b: "color: red" if b < 0 else "color: green" if b > 0 else "color: black"})._repr_html_()
    components.html(rs_html, width=900, height=500, scrolling=True)

    full_model = Ridge()
    full_model.fit(one_hot_mat,df.pIC50)
    already_made_smiles = set([Chem.MolToSmiles(x) for x in df.mol])

    st.subheader('Enumerating the products')
    progress_text = "Enumeration in progress"
    my_bar = st.progress(0, text=progress_text)

    uru.rd_shut_the_hell_up()
    prod_list = []
    for i,p in enumerate(product(*enc.categories)):
        my_bar.progress(i / total_possible_products, text=f"Enumeration in progress, total products = {total_possible_products}")
        core_smiles = rgroup_df.Core.values[0]
        smi = (".".join(p))
        mol = Chem.MolFromSmiles(smi+"."+core_smiles)
        prod = Chem.molzip(mol)
        prod = Chem.RemoveAllHs(prod)
        prod_smi = Chem.MolToSmiles(prod)
        if prod_smi not in already_made_smiles:
            desc = enc.transform([p])                           
            prod_pred_ic50 = full_model.predict(desc)[0]
            prod_list.append([prod_smi,prod_pred_ic50])
        if i == 5000:
            break

    prod_df = pd.DataFrame(prod_list,columns=["SMILES","Pred_pIC50"])
    prod_df.sort_values("Pred_pIC50",ascending=False,inplace=True)

    best_df = prod_df.head(100).copy()
    best_df['mol'] = best_df.SMILES.apply(Chem.MolFromSmiles)
    for mol in best_df.mol:
        AllChem.GenerateDepictionMatching2DStructure(mol,core_mol)
    prods_html = mols2grid.display(best_df,mol_col='mol',use_coords=True, prerender=True, substruct_highlight=False,
                    transform={"Pred_pIC50" : lambda x: f"{x:.1f}"},
                    subset=["img","Pred_pIC50"])._repr_html_()

    components.html(prods_html, width=900, height=500, scrolling=True)


    st.subheader('Download option')
    st.text('peek at data')
    st.write(prod_df.head())
    st.download_button(
    "Press to Download the full enumerated dataframe",
    prod_df.to_csv(index=False).encode('utf-8'),
    "results.csv",
    "text/csv",
    key='download-csv'
    )
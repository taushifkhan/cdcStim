import streamlit as st
import pandas as pd
import scipy.stats as sT
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import numpy as np
import base64
import seaborn as sns

import altair as alt
alt.data_transformers.disable_max_rows()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Transcriptomics of cDC with stimulation and adjuvent + stimulation")

st.markdown("""
* version = 0.1 [16 Feb 2023]
* Latest : addition of FC list (downloadable) and DEG figure in Tab : Fold Change w.r.t Base
* contact : taushif.khan@jax.org
""")

dataLink = {
    "normData":'data/CPM_TMM_adjcount.csv.zip',
    "fc_sample":'data/fc_individual.csv.zip',
    "sampleInfo":'data/sampleListPCAannot.csv.zip',
    "degStim_adju":'data/DEG_stim_adjuvent.csv.zip',
    "modularTranscript":'data/DC_moduleChange.csv.zip',
    "dcModulesDef":'data/dcNormModules.csv.zip',
}

def plotResponsiveness(fc_individual, cutoff):
    fc_filtered = fc_individual[abs(fc_individual)>=cutoff]
    fcResponsiveGene = fc_filtered.fillna(0).astype(bool).sum().to_frame().rename({0:'TotalResponsive'},axis=1)
    fcResponsiveGene['upReg_all'] = fc_individual[fc_individual<=-1* cutoff].fillna(0).astype(bool).sum()
    fcResponsiveGene['dwnReg_all'] = fc_individual[fc_individual>=cutoff].fillna(0).astype(bool).sum()
    fcResponsiveGene['sampleName'] = [i.split(".")[0] for i in fcResponsiveGene.index]
    fcResponsiveGene['condition'] = ["_".join(i.split(".")[1:]) for i in fcResponsiveGene.index]
    
    regX = fcResponsiveGene[['upReg_all','dwnReg_all','sampleName','condition']].\
    melt(id_vars=['sampleName','condition']).rename({'variable':'state','value':'geneCount'},axis=1)

    regX.loc[regX[regX.state=="dwnReg_all"].index,'geneCount'] = -1 * regX.loc[regX[regX.state=="dwnReg_all"].index,'geneCount']
    return regX, fc_filtered
    

def get_table_download_link(df,fname):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    # rfile = """f<a href="data:file/csv;base64,{b64}" download="{}_{}.csv">Download csv file</a>""".format(fname,df.shape[0])
    return f'<a href="data:file/csv;base64,{b64}" download="{fname}_{df.shape[0]}.csv">Download csv file</a>'

@st.cache(suppress_st_warning=True)
def loadData():
    normCount = pd.read_csv(dataLink["normData"],compression="zip").rename({"Unnamed: 0":'gene'},axis=1).set_index("gene")
    fc_Sample = pd.read_csv(dataLink["fc_sample"],compression="zip").rename({"Unnamed: 0":'gene'},axis=1).set_index("gene")
    sampleInfo = pd.read_csv(dataLink["sampleInfo"],compression="zip")
    degStim_adju = pd.read_csv(dataLink["degStim_adju"],compression="zip")
    dcmodules = pd.read_csv(dataLink["modularTranscript"],compression="zip")
    dcModuleDef = pd.read_csv(dataLink["dcModulesDef"],compression="zip")
    return normCount, fc_Sample, sampleInfo, degStim_adju, dcmodules, dcModuleDef


normCount, fc_Sample, sampleInfo, degStim_adju, dcmodules,dcModuleDef = loadData()

norm_expression, grpDEG, foldchanges, degCompare, moduleTranscriptomics = st.tabs(["Norm Expr", "Group DEG","Fold Changes w.r.t Base",\
     "DEG : Stim Vs Adj+Stim","Module Transciptomics"])

with norm_expression:
    st.subheader("PCA plot for sample and stiimulation with normalized expression")
    pca1 = alt.Chart(sampleInfo, width=350, height=400).mark_point(filled=True,size=180).encode(
        x=alt.X('PC1',axis=alt.Axis(title="PC1 [32.86%]",titleFontSize=14)),
        y=alt.Y('PC2',axis=alt.Axis(title="PC2 [7.07%]",titleFontSize=14)),
        color=alt.Color('condition'),
        shape=alt.Shape('Age_Group'),
        tooltip=['sname','Age','Gender','Donor_Source']
    ).properties(title="PCA on experssion of {} genes; {} samples".format(normCount.shape[0], normCount.shape[1]))

    st.altair_chart(pca1, use_container_width=True)
    st.caption("Hover-over each point to know more about sample phenotype and treatment condition")

    st.markdown("""
    PCA plot shows clustering of normalized gene expression of 17758 genes for  samples (n=10) treated in 8 different condition 
    (7 stimulation; 1 medium). Data points are marked with shapes indicating age group (Old and Young) and colored with different stimulation 
    condition. Proliferation (or change in gene expression) profile of each samples teated with stimulation condition can be oberved to be clustering together.
    """)

with grpDEG:
    st.markdown("""
    ### Treating sample groups as cohorts for stimulation

    #### DEGs in Old Vs Young 
    Batch adjusted normalised expression profile for each stimulation condition used to extract DEGs among Old and Young samples. Each 
    sample group has 5 samples (data points), we used edgeR and limma pakage in R to estimate Log fold change for each gene.
    ##### Previous version filter with FDR #####
    FDR of 0.05 has been set to filter genes that were significant difference in two age groups.

    In brief, we observed very low number of gene with differential expression profile among samples grouped by Age groups. From all 8 stimulation, we found 89 DEGs 
    genes that observed to have different experssion with a FDR of 0.05. The low number of DEG indicates that samples grouped with Age Group (Old/Young) do not have any 
    major difference in DC proliferation. This is true across all stimulaiton condition. Another probable cause might be the higer sample heterogenity, hence grouping with
    a Age phenotype, DEGs technique can not pick up genes with resulting in cDC differentiation.

    ##### Filtering with P-value : less stringent #####
    1. select stimulaiton condition 
    2. Use the slider to choose P-value 
    3. If immune geens re present select radio 
    4. Download DEGs with normalized expression profile 
    5. See Heatmap of z-score of normalized expresison value 
    6. Vizualize normalized expressin of selected gene as boxplot  
    """)
    @st.cache(suppress_st_warning=True)
    def dataDEG_ageGroup():
        deg_age = pd.read_csv("data/AgeGroup_DEG_FC.csv.gz",compression="gzip").set_index("Unnamed: 0")
        pval_age= pd.read_csv("data/AgeGroup_DEG_Pval.csv.gz",compression="gzip").set_index("Unnamed: 0")
        immuneGenes = pd.read_csv("data/immoGenes.csv.gz",compression="gzip")
        return deg_age, pval_age, immuneGenes
    
    deg_age, pval_age, immuneGenes = dataDEG_ageGroup()
    # st.caption("Table : DEG table for each stimulation tested for Age Groups (significant @ FDR < 0.05)")
    # st.write(deg_age)
    st.subheader("Select a stimulaiton to see DEGs and sample expression")
    deg1_stim = st.selectbox("Select stimulation condition: ", list(deg_age.columns))
    df_stim1_pval = pval_age[deg1_stim+"_pval"].dropna()
    histfig, hax = plt.subplots(figsize=(5,3))
    # df_stim1_pval.plot(kind="hist", ax= hax)
    hax.hist(df_stim1_pval.values, 20, cumulative=True)
    plt.xlabel("P-value")
    plt.ylabel("Number of DEGs Filtered")
    st.pyplot(histfig)

    pval_filter = st.slider("Select P-value cut-off", min_value=df_stim1_pval.min()+0.001,max_value=df_stim1_pval.max(),value=0.05, step=0.001)
    deg_age_stimfilter = deg_age.loc[df_stim1_pval[df_stim1_pval<=pval_filter].index][deg1_stim]
    if deg_age_stimfilter.shape[0]>3:
        st.write("Total DEG [for Age Groups] in stim {} with Pvalue <= {} : {}".format(deg1_stim, pval_filter, deg_age_stimfilter.shape[0]))
        _immSubset = list(set(deg_age_stimfilter.index).intersection(set(immuneGenes.Symbol)))
        if len(_immSubset):
            immsubsetDF = deg_age_stimfilter.loc[_immSubset]
            st.write("Select DEG has {} Immune Genes ".format(len(_immSubset)))
        else:
            st.write("No immune related genes found")

        _sampleSelect = sampleInfo[sampleInfo.condition==deg1_stim].sort_values(by='Age',ascending=False).set_index('sname')
        df_count_PT = normCount[_sampleSelect.index].loc[deg_age_stimfilter.index]
        st.subheader("Download expression profile:")
        flname = "normExpr_{}_{}_{}.csv".format(deg1_stim, pval_filter, df_count_PT.shape[0])
        filelink = get_table_download_link(df_count_PT,flname)
        st.markdown(filelink,unsafe_allow_html=True)

        st.caption("Heat map for selected DEG : {} [samples from Old (left) to Young (right)]".format(deg1_stim))
        _allName = "ALL [{}]".format(df_count_PT.shape[0])
        _immName = "Immune [{}]".format(len(_immSubset))
        glistview = st.radio("Select ALL/ Immune related genes", (_allName,_immName))

        if ((glistview == _immName) & (len(_immSubset)>3)):
            hm_ageG = sns.clustermap(df_count_PT.loc[_immSubset].apply(sT.zscore,axis=1),cmap='RdBu_r',figsize=(6,10),vmin=-1.5,vmax=1.5,col_cluster=False,
                    cbar_kws={'label':'normalized experssion (z-score)'},cbar_pos=[1.,0.5,0.05,0.1])
            st.pyplot(hm_ageG)
        else:
            hm_ageG = sns.clustermap(df_count_PT.apply(sT.zscore,axis=1),cmap='RdBu_r',figsize=(6,10),vmin=-1.5,vmax=1.5,col_cluster=False,
                    cbar_kws={'label':'normalized experssion (z-score)'},cbar_pos=[1.,0.5,0.05,0.1])
            st.pyplot(hm_ageG)
       # boxplot for each gene
        df_count = df_count_PT.unstack().reset_index().rename({0:'normExpr_val','Unnamed: 0':'gene','level_0':'sampleName'},axis=1)
        df_count = df_count.set_index('sampleName').join(_sampleSelect[['Age','Age_Group']]).reset_index()
        deggene_select = st.selectbox("Gene from DEGs to see as a boxplot on Normalized expression", df_count.gene.unique())
        gxpr_stim, gbox_stim = plt.subplots()
        PROPS = {
       'boxprops':{'edgecolor':'None'},
            }
        sns.boxplot(x='Age_Group',y='normExpr_val',data=df_count[df_count.gene==deggene_select],\
                palette=['#bdbdbd','#636363'], ax= gbox_stim, **PROPS)
        sns.stripplot(x='Age_Group',y='normExpr_val',data=df_count[df_count.gene==deggene_select],\
            legend=False, color="k")
        plt.ylabel("Expression profile of {}".format(deggene_select),fontdict={'size':14})
        plt.tight_layout()
        st.pyplot(gxpr_stim)
    else:
        st.write("No DEG FOUND increase the P-value")

    

with foldchanges:
    st.markdown("""
    ## Background ##
    10 samples were treated with 8 conditions. cDC from all samples were treated with stimulations (R848, cGAMP and SDRNA_NP) and with 
    addition of adjuvent ASP1 along with no-stimulation ("medium") and only adjuvent ("ASP1"). Sample-wise fold changes were estimated
    by substracting normalized expression of each gene after stimulation with respect to base response ("medium").

    ### Responsive genes: ###
    With a selected cut-off value, a gene response can be assigned as up (FC> cutoff) or down [FC < -1 x cutoff] regulated. This will select
    a set of genes that have considerable differene (cut-off) from base as aresult of stimulation. Usually, a |FC| of more than 1 (log scale)
    considers to be ideal cut-off. 
    
    **Tune the slider below to see change in responsive genes with different threshold**
    """)
    cutoff_select = st.slider("change in absolue fold change (|FC|)",value=1.0, max_value=2.5, min_value= 0.5)
    regX, filterdeFC = plotResponsiveness(fc_Sample, cutoff=cutoff_select)
    respGFilgure = plt.figure(figsize=(5,2))
    simorder = ['ASP1','R848','ASP1_R848','SDRNA_NP','ASP1_SDRNA_NP','cGAMP','ASP1_cGAMP']
    g = sns.barplot(y='condition',x='geneCount',data=regX,hue='state',dodge=False,\
                order=simorder,width=0.8,palette=['#e6550d','#3182bd'])
    g.legend_.remove()
    sns.stripplot(y='condition',x='geneCount',data=regX,hue='state',dodge=False,\
                order=simorder,legend=False,color='k')
    plt.title("Reponsive genes at FC >= |{}|".format(cutoff_select))
    st.pyplot(respGFilgure)

    st.markdown("Above bar plot shows responsive gene count (x-axis) per sample as upregulated (> cutoff ; orange color) & down \
    regulates (< -cutoff blue color) for each stimulation condition.")

    def tsum(kgrp):
        return (abs(kgrp.geneCount.values[0])+abs(kgrp.geneCount.values[0]))

    respGCol = regX.groupby(['sampleName','condition']).apply(tsum).reset_index().pivot(index='sampleName',columns='condition',values=0)
    respGCol = respGCol[simorder]
    st.caption("Table : Number of responsive genes per sample -stimulation with |FC| = {}".format(cutoff_select))
    st.table(respGCol)

    st.subheader("Download responsive genes FC for a stimulation")
    stimcondition = st.selectbox("Choose a stimulation to get all DEG estimated per sample:",simorder)
    _sampleNmaes = sampleInfo[sampleInfo.condition==stimcondition].sname.unique()
    df_fcselelcted = filterdeFC[_sampleNmaes]
    nz_samplecutoff = df_fcselelcted.fillna(0).astype(bool).sum(axis=1)
    minsampleSel = st.slider("Responsive gene in minimum number of sample:",min_value=1,max_value=10,value=7)
    filteredGlist = nz_samplecutoff[nz_samplecutoff>=minsampleSel].index
    df_fcselelcted_mingene = df_fcselelcted.loc[filteredGlist]
    st.markdown("""
    ### Filter parameters:
    * Stimulation: {}
    * Minimum number of samples with a responsive gene={}
    * Total filtered gene set : {}""".format(stimcondition, minsampleSel, df_fcselelcted_mingene.shape))
    filenName_download = "sampleWiseDEG_{}_{}_{}.csv".format(stimcondition,minsampleSel,df_fcselelcted_mingene.shape[0])
    st.write("Clink the link to download selected file :{}".format(filenName_download))
    filelink = get_table_download_link(df_fcselelcted_mingene,filenName_download)
    st.markdown(filelink,unsafe_allow_html=True)

    st.write(df_fcselelcted_mingene)

    st.subheader("Heatmap of selected DEG and medium")
    st.write("Samples were ordered from Old to Young.")
    _sampleselect = sampleInfo[(sampleInfo.condition=="medium")|\
        (sampleInfo.condition==stimcondition)].sort_values(by=['condition','Age_Group']).sname.unique()
    nexpr_selected = normCount[_sampleselect].loc[df_fcselelcted_mingene.index].fillna(0).apply(sT.zscore,axis=1)

    degplot_stimFC = sns.clustermap(nexpr_selected,cmap='RdBu_r',figsize=(12,14),vmin=-2,vmax=2,col_cluster=False,
                cbar_kws={'label':'normalized experssion (z-score)'},cbar_pos=[1.,0.5,0.05,0.1])
    st.pyplot(degplot_stimFC)
    st.write("DEG heatmap ")
    sampleInfoAccess = sampleInfo[['sID','Age','Age_Group']].drop_duplicates(subset="sID").sort_values(by='Age_Group')
    st.write(sampleInfoAccess)

with degCompare:
    st.markdown("""
    ### Effect of addjuvent aided stimulation
    Compared fold changes (compared w.r.t base stimulation; previous tab) each stimulation condition (cGAMP; R848; SDRNA_NP) to that of 
    in combination of ASP1 as adjuvent. The statistics was performed as in a T-test and a P-value of < 0.05 is considered to be significant.
    Correction for multiple testing was also performed using Benjamini Hochberg correction, however, filtering was only done on P-value to be 
    inclusive.

    **Use the slider to select a P-value cut-off [recomended 0.05]**
    """)
    
    pvalSlider  = st.slider("P-value to select DEGs for stim Vs. ASP1+stim:",\
                            min_value=0.01, max_value=0.1,value=0.05,step=0.01)
    
    degStim_adju_sig = degStim_adju[degStim_adju.pval<=pvalSlider]

    set_compo = degStim_adju_sig.groupby('compare')['gene'].apply(set)
    setFigure, ax = plt.subplots()
    venn3(set_compo.values,set_compo.keys(), ax = ax)
    st.pyplot(setFigure)
    st.caption("Over lap of DEGs in Stimulation Vs ASP1+Stimulation condition with selected P-value cutoff of: {}".format(pvalSlider))
    
    st.subheader("Download selected DEGs")
    stimselect = st.selectbox("Choose stimulation with above P-value:",list(set_compo.keys()))
    selectedDEGs = degStim_adju_sig[degStim_adju_sig["compare"]==stimselect].sort_values(by="pval")
    st.text("Number of GEGs for {} with P-value < {} = {}\nUse the link below to get selcted DEG list as CSV file".\
            format(stimselect ,pvalSlider, selectedDEGs.shape[0]))
    filelink = get_table_download_link(selectedDEGs,"{}_{}.csv".format(stimselect,pvalSlider))
    st.markdown(filelink,unsafe_allow_html=True)

    stim1 = stimselect.split("ASP1")[0][:-1]
    stim2 = "ASP1"+stimselect.split("ASP1")[1]

    st.subheader("Heat map of top DEGs {} Vs {}".format(stim1, stim2))
    # get sample organised
    sampleInfo_tmp = sampleInfo[(sampleInfo.condition.str.contains(stim1))|(sampleInfo.condition=="ASP1")|\
        (sampleInfo.condition=="medium")]
    sampleInfo_tmp['condition'] = pd.Categorical(sampleInfo_tmp['condition'], ["medium", "ASP1", stim1,stim2])
    sampleInfo_tmp = sampleInfo_tmp.sort_values(by=['condition','Age'])
    # import ipdb; ipdb.set_trace();
    ngenes = st.slider("select number of genes in the heatmap", min_value=20,max_value=selectedDEGs.shape[0],value=50)
    x = normCount[sampleInfo_tmp.sname.values].loc[selectedDEGs.gene.values[:ngenes]].fillna(0)
    x = x.apply(sT.zscore,axis=1)
    deg_hm = sns.clustermap(x,cmap='RdBu_r',figsize=(12,14),vmin=-2,vmax=2,col_cluster=False,
                cbar_kws={'label':'normalized experssion (z-score)'},cbar_pos=[1.,0.5,0.05,0.1])
    st.pyplot(deg_hm)
    st.caption("Normalized expression profile (z-score) of top {} DEGs, color coded from red (high expressed) to blue (low expressed)\
        across samples. Samples are organised for each condition in ascending order of Age, with younger to older sample.")

    st.subheader("Compare expression of gene across stimulation")
    st.write("Box plot for stimulation wise expression profile of selected gene in {} and {}".format(stim1, stim2))
    geneSelected = st.selectbox("choose gene",list(selectedDEGs.gene.values))
    xGene = normCount[sampleInfo_tmp.sname.values].loc[geneSelected].to_frame().reset_index()
    xGene['sampleName'] = [i.split(".")[0] for i in xGene['index'].values]
    xGene['stimulation'] = ["_".join(i.split(".")[1:]) for i in xGene['index'].values]

    gxpr, gbox = plt.subplots()
    PROPS = {
    'boxprops':{'edgecolor':'None'},
    }
    sns.boxplot(x='stimulation',y=geneSelected,data=xGene,order=['medium','ASP1',stim1,stim2],
            palette=['#bdbdbd','#636363','#9ecae1','#3182bd'], ax= gbox, **PROPS)
    sns.stripplot(x='stimulation',y=geneSelected,data=xGene,order=['medium','ASP1',stim1,stim2],\
        legend=False, color="k")
    plt.ylabel("Expression profile of {}".format(geneSelected),fontdict={'size':14})
    plt.tight_layout()
    st.pyplot(gxpr)

with moduleTranscriptomics:
    st.markdown("""
    ### Chnage in Transcriptomics Module profile ###
    Dendritic Cell (DC) specific module definitions were adopted from [1]. Briefly,  Banchereau et.al, built on [2] a modular framework 
    of transcriptional module for human DCs, to understand its proliferation (or response) upon vaccine challenge in vitro. Here, for sample-wise 
    fold changes (Tab 2) we overlay the modular annotation of about 2000 genes for responsive genes (|FC| > 1). When a gene response was recorded
    more than > 1 fold change (FC) (Log scale) as compare to medium we annotate the response as up-regulation. A FC < -1 were labeled as down-regulated.

    Gene modules response were aggregated into modules, which is a set of genes with similar transcription profile as recored by [1].

    Reference:

    [1] Banchereau R, et al. Transcriptional specialization of human dendritic cell subsets in response to microbial vaccines.
    Nat Commun. 2014 Oct 22;5:5283. 
    doi: 10.1038/ncomms6283. PMID: 25335753; PMCID: PMC4206838.

    [2] Chaussabel D. et al. A modular analysis framework for blood genomics studies: application to systemic lupus erythematosus. 
    Immunity 29, 150–164 (2008).
    """)

    st.subheader("Modules variance across sample-stimulation")
    st.markdown("""
    To analyze focused modules, we estimate module variance across samples-stimulation which can be filtered with a quantile distribution.
    Use the slider below to adjust the desired variance with 25th to 99th percentile i.e. from least variance to max variance in modules across stimulation.
    The scale on the slider indicate qauntile. With slider at 0.9, we filter the modules that have variance across sample-srimulation more than 90 quartile. 
    Hence, filtering the modules that have maximum difference in response profile across stimulation conditions.

    The resultant heatmap, indicate collective response (median) of samples in a stimualtion for respective modules (y-axis). The sample wise varaiation
    is describbed by standard deviation (std) indicated by size of the spot. Larger inter sample variation for a module in a stimulation, smaller is the size of the spot. 
    Largest spots represnts, stimualtion condition with least sample wise deviation.  
    """)
    modChangePT_median = dcmodules.groupby(["ModName","stimulation"])['pct_change'].agg(['median','std']).reset_index()
    stimVar = modChangePT_median.pivot(index='ModName',columns='stimulation',values='median').var(axis=1)
    selelctVarQuantile = st.slider("Filter for Module with variance across",min_value=0.1,max_value=0.99,value=0.90)
    moduleSelect = stimVar[stimVar>=stimVar.quantile(selelctVarQuantile)].index
    mod_df = modChangePT_median[modChangePT_median.ModName.isin(moduleSelect)]
    xmap = sns.clustermap(mod_df.pivot(index='ModName',columns='stimulation',values='median').fillna(0),figsize=(4,4))

    cmap = ['#92c5de','#0571b0','#f7f7f7','#ca0020','#d7191c']
    varMap = alt.Chart(mod_df[~mod_df.ModName.str.contains("Ribosomal")],width=550).mark_point(filled=True).encode(
        x=alt.X('stimulation', sort=['ASP1','R848','ASP1_R848','SDRNA_NP','ASP1_SDRNA_NP','cGAMP','ASP1_cGAMP']),
        y=alt.Y('ModName',sort=list(xmap.data2d.index),axis=alt.Axis(labelFontSize=14,labelLimit=1000,title='')),
        color=alt.Color('median',scale=alt.Scale(range=cmap,domain=[-0.7,-0.2,0,0.2,0.7],clamp=True,)),
        size=alt.Size('std',scale=alt.Scale(reverse=True),legend=alt.Legend(values=[0.05,0.2,0.5])),
        tooltip=['ModName','median','std']
    )

    st.altair_chart(varMap, use_container_width=False)

    st.caption("Median response per stimulation condition in Up/Down regulation for listed modules")

    st.subheader("Subject wise response in selected modules")
    modStimselect = st.selectbox("choose stimulation :",['cGAMP','R848','SDRNA_NP'])

    dx_asp1 = dcmodules[((dcmodules.stimulation.str.contains(modStimselect))|(dcmodules.stimulation=="ASP1"))\
                       &(dcmodules.ModName.isin(moduleSelect))].reset_index().drop('index',axis=1)
    dx_asp1['stimulation'] = pd.Categorical(dx_asp1['stimulation'],['ASP1',modStimselect,'ASP1_'+modStimselect])
    dx_asp1 = dx_asp1.sort_values(by='stimulation').reset_index().drop(['index','Unnamed: 0'],axis=1).reset_index()
    dx_asp1 = dx_asp1.set_index('sampleID').join(sampleInfo.set_index("sID")[['Age','Age_Group']]).\
        drop_duplicates(subset="index").sort_values(by=['stimulation','Age'])
    dx_asp1 = dx_asp1[~dx_asp1.ModName.str.contains("Ribosomal")]

    sampleInfo_selected = sampleInfo.set_index("sname").loc[dx_asp1.sampleName.unique()].reset_index()
    sampleOrderPlot = alt.Chart(sampleInfo_selected,height=20,width=560).mark_rect().encode(
        x = alt.X('sname',sort=list(sampleInfo_selected.sID.values),axis= alt.Axis(labelFontSize=14,)),
        color=alt.Color('Age_Group',scale=alt.Scale(domain=['Young','Old'],range=['#756bb1','#c51b8a']))
    )

    modChangePlot = alt.Chart(dx_asp1,width=560).mark_point(filled=True).encode(
        x=alt.X('sampleName', sort=dx_asp1.sampleName.values,axis=alt.Axis(ticks=False,labels=False,title='')),
        y=alt.Y('ModName',sort=list(xmap.data2d.index),axis=alt.Axis(labelFontSize=14,labelLimit=1000,title='')),
        color=alt.Color('pct_change',scale=alt.Scale(range=cmap,domain=[-0.7,-0.2,0,0.2,0.7],clamp=True,)),
        size=alt.Size('pct_overlap',scale=alt.Scale(reverse=False),legend=alt.Legend(values=[0.1,0.3,0.7])),
        tooltip=['ModName','sampleName','Age_Group','Age','pct_change','pct_overlap']
        )

    modPlot = alt.vconcat(modChangePlot,sampleOrderPlot).properties(spacing=-5)
    st.altair_chart(modPlot)

    st.subheader("Genes expression profile in a Module:")
    getmoduleName = st.selectbox("Choose a module to look into gene expression", list(dx_asp1.ModName.unique()))
    moduleName = getmoduleName.split("_")[0]
    fcTMP = fc_Sample.loc[dcModuleDef[dcModuleDef.module==moduleName].genes][dx_asp1.sampleName.unique()]
    st.write(fcTMP)
    st.write("For module response we filterout any response less than |FC| <1")
    fcSelect_rep = fcTMP[abs(fcTMP)>=1]
    modDX = sns.clustermap(fcSelect_rep.fillna(0),cmap="RdBu_r",vmin=-3,vmax=3,figsize=(12,4),col_cluster=False)
    fcSelect_repX = fcSelect_rep.unstack().reset_index().rename({'level_0':'sampleName','Unnamed: 0':'gene',0:'FC'},axis=1).dropna()

    genesInModplot = alt.Chart(fcSelect_repX,width=560).mark_rect().encode(
        x = alt.X("sampleName",sort=dx_asp1.sampleName.values, axis=alt.Axis(ticks=False,labels=False,title='')),
        y = alt.Y("gene",sort=list(modDX.data2d.index), axis=alt.Axis(labelFontSize=14,)),
        color = alt.Color("FC",scale=alt.Scale(range=cmap,domain=[-5.5,-2.5,0,2.5,5.5],clamp=False,)),
        tooltip = ['FC','sampleName','gene']
    )

    annotGenemodPlot = alt.vconcat(genesInModplot,sampleOrderPlot).properties(spacing=-5)
    st.altair_chart(annotGenemodPlot,use_container_width=True)
    st.caption("Fold change w.r.t Base (medium response for each subject for all genes in the selected module")

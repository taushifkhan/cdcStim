import streamlit as st
import pandas as pd
import scipy.stats as sT
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import numpy as np
import base64
import seaborn as sns
import upsetplot
from upsetplot import from_contents

import altair as alt
alt.data_transformers.disable_max_rows()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Transcriptomics of cDC with stimulation and adjuvent + stimulation")

st.markdown("""
* version = June 2023 [v1.2]
* Latest : 20 samples and 6 treatment 
* contact : taushif.khan@jax.org
""")

colorcode = {'medium':'#737373','ASP1':'#dd1c77','R848':'#9ecae1','ASP1-R848':'#3182bd',
             'cGAMP':'#bcbddc','ASP1-cGAMP':'#6a51a3'}
PROPS = {'boxprops':{'edgecolor':'None'},}

dataLink = {
    "normData":'data_7June2023/humancDC_normCount_SourceGenderCorr.pkl.gzip',
    # "fc_sample":'data/fc_individual.csv.zip',
    "sampleInfo":'data_7June2023/humancDC_sampleInfo_PCA.csv.gz',
    # "degStim_adju":'data/DEG_stim_adjuvent.csv.zip',
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
    normCount = pd.read_pickle(dataLink["normData"],compression="gzip")
    # fc_Sample = pd.read_csv(dataLink["fc_sample"],compression="zip").rename({"Unnamed: 0":'gene'},axis=1).set_index("gene")
    sampleInfo = pd.read_csv(dataLink["sampleInfo"],compression="gzip").set_index("sname")
    # degStim_adju = pd.read_csv(dataLink["degStim_adju"],compression="zip")
    # dcmodules = pd.read_csv(dataLink["modularTranscript"],compression="zip")
    # dcModuleDef = pd.read_csv(dataLink["dcModulesDef"],compression="zip")
    # return normCount, fc_Sample, sampleInfo, degStim_adju, dcmodules, dcModuleDef
    return normCount, sampleInfo


# normCount, fc_Sample, sampleInfo, degStim_adju, dcmodules,dcModuleDef = loadData()
normCount, sampleInfo = loadData()

# norm_expression, grpDEG, foldchanges, degCompare, moduleTranscriptomics = st.tabs(["Norm Expr", "Group DEG","Fold Changes w.r.t Base",\
    #  "DEG : Stim Vs Adj+Stim","Module Transciptomics"])

norm_expression, deg_treatment, deg_ageGroup, downloadData = st.tabs(["Sample Brief","DEG_Treatments","DEG_AgeGroup", "Download"])

with norm_expression:
    st.subheader("PCA plot for sample and stiimulation with normalized expression")
    pca1 = alt.Chart(sampleInfo.reset_index()).mark_point(filled=True,size=180).encode(
        x=alt.X('PC1'),
        y=alt.Y('PC2'),
        color=alt.Color('Treatments',scale=alt.Scale(domain=list(colorcode.keys()), range=list(colorcode.values()))),
        shape=alt.Shape('Age_Group'),
        tooltip=['Donor_ID','Age_Group','Donor_ID','Seq_Run','Exp_No','Seq_Run','Donor_Source']
        ).properties(title="PCA on experssion of {} genes; {} samples".format(normCount.shape[0], normCount.shape[1])).interactive()

    st.altair_chart(pca1, use_container_width=True)
    st.caption("Hover-over each point to know more about sample phenotype and treatment condition")

    st.markdown("""
    * PCA plot shows clustering of normalized gene expression of 23326 genes for  samples (n=20) treated in 6 different condition \
            (5 stimulation; 1 medium). Data points are marked with shapes indicating age group (Old and Young) and colored with different stimulation \
            condition. 
    * Normalization (edgeR: calcNormFactors with TMM method) was done over batch corrected samples (Combat-Seq; 4 batches).
    """)

    # sequencing depth
    st.subheader("Sequencing depth of each samples and Treatment condition")
    fig,ax = plt.subplots(1,2,figsize=(6,3),sharey=True,width_ratios=[0.7,0.3])

    PROP = {'boxprops':{'edgecolor': None}}
    sns.boxplot(x='Donor_ID',y='totalCount',data=sampleInfo,ax=ax[0],color='0.6',**PROPS)
    sns.boxplot(x='Treatments',y='totalCount',data=sampleInfo,ax=ax[1],order=['medium','ASP1','R848','ASP1-R848','cGAMP','ASP1-cGAMP'],
            color='0.7',**PROPS)
    plt.tight_layout(w_pad=1)
    ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=90)
    ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=90)
    
    st.pyplot(fig)
    st.markdown("""
    boxplot on left : distribution of total count for each sample for all 6 treatment condition. [Right] distribution of
    sequencing depth (total count) for each treatment over all samples. 
    """)

with deg_treatment:
    st.markdown("""
    DEG for each stimulation condition.
    """)
    treatDEG = pd.read_pickle("data_7June2023/humancDC_treatment_DEG.pkl",compression="gzip")
    fdr = treatDEG.loc[0,'dfs']
    fc = treatDEG.loc[1,'dfs']
    cutoff_threshold = st.selectbox("FDR threshold for signifiance:",[0.005,0.001,0.0005,0.0001,0.00005,0.05])
    significant_dict = {}
    st.write("Number of significant DEG at selcted FDR :", cutoff_threshold)
    for k in fdr.columns:
        filterGenes = fdr[fdr[k]<cutoff_threshold].dropna().index
        significant_dict[k] = filterGenes
        st.write(k , filterGenes.shape[0])

    
    mselect_condition = st.multiselect("Remove conditions to filter significant DEG", list(significant_dict.keys()),list(significant_dict.keys()))
    filter_dict = {key: significant_dict[key] for key in mselect_condition}
    # plot
    ax = plt.figure()
    upsetplot.plot(from_contents(filter_dict),min_subset_size=1,show_counts=True,fig=ax)
    st.pyplot(ax)

    st.subheader("Filter significant DEGs for treatment conditions/ combinations")
    sigcols = st.multiselect("Significant in ", list(fdr.columns), ['ASP1'])
    nonsigcols = st.multiselect("Not significant in", list(set(fdr.columns)-set(sigcols)))

    sigcase = fdr[sigcols][fdr[sigcols]<cutoff_threshold].dropna().index
    tmp     = fdr.drop(sigcols,axis=1).loc[sigcase]
    filter_sig = tmp[tmp[tmp>=cutoff_threshold].fillna(0).astype(bool).sum(axis=1)==len(nonsigcols)].index
    filter_sig = list(set(normCount.index).intersection(set(filter_sig)))

    st.write("Significant in :"," : ".join(sigcols),sigcase.shape, "not in", ":".join(nonsigcols), "Gene Found :", len(filter_sig), len(nonsigcols))

    if len(filter_sig) >=1:
        flName_treatSig = "DEG_FC_{}.csv".format("_".join(sigcols))
        fc_sig = fc.loc[filter_sig][sigcols]
        degTreat_link = get_table_download_link(fc_sig.reset_index(), flName_treatSig)
        st.write("Download DEG LIST HERE: ", fc_sig.shape)
        st.markdown(degTreat_link, unsafe_allow_html=True)

        if len(filter_sig) >= 5:
            _samplesfor_heatmap = sampleInfo[(sampleInfo.Treatments.isin(sigcols))|(sampleInfo.Treatments=='medium')].sort_values(by=['Treatments','Age_Group']).index
            data_hm = normCount.loc[filter_sig][_samplesfor_heatmap]
            sns.set(font_scale=1.2)
            hm_treat = sns.clustermap(data_hm,cmap="coolwarm",col_cluster=False,z_score=0,vmin=-1,vmax=1,
               cbar_pos=[1,0.6,0.03,0.2],cbar_kws={"label":'norm expression (z-score)'},figsize=(20,20))
            st.pyplot(hm_treat)
        else:
            st.write("Not enough genes for heatmap [min deg >=5]")

        geneSelect = st.selectbox("Select Gene for normExpression:", filter_sig)
        bplotdData = normCount.loc[geneSelect].to_frame().join(sampleInfo[['Treatments','Age_Group']])
        gxpr_treat, gbox_treat = plt.subplots(1,1,figsize=(6,4))
        sns.boxplot(x='Treatments',y=geneSelect,data=bplotdData,order=list(colorcode.keys()),\
                                    palette=list(colorcode.values()),**PROPS,ax=gbox_treat)
        plt.ylabel("Expression profile of {}".format(geneSelect),fontdict={'size':14})
        st.pyplot(gxpr_treat)

    st.write("## Knowledge driven functional enrichment")
    kd_deg = pd.read_pickle("data_7June2023/KD_TreatmentPathway.pkl.gzip",compression="gzip")
    db_select = st.multiselect("Select one or multiple dataset", list(kd_deg.dataset.unique()),['KEGG_2021_Human'])
    sig_thresold = st.selectbox("FDR q-value for enrichment: ", [0.005,0.001, 0.05, 0.01],)
    kd_select_long = kd_deg[(kd_deg.dataset.isin(db_select))&(kd_deg['FDR q-val']<sig_thresold)]
    kd_select = kd_select_long.pivot_table(index='Term',columns='sampleInfo',values="NES")
    st.write("Number of pathways enriched in {} for FDR [<{}] : {}".format(db_select, sig_thresold, kd_select.shape[0]))
    if kd_select.shape[0] > 3:
        sns.set(font_scale=1)
        kd_heatmap = sns.clustermap(kd_select.fillna(0),cmap="Reds",figsize=(3.5,5),col_cluster=False,
                                    cbar_kws={'label':'Norm. Enrichment Score','orientation':'horizontal'},cbar_pos=[0.3,0.9,0.35,0.03])
        st.pyplot(kd_heatmap)
    else:
        st.write(kd_deg)
    if kd_select.shape[0] > 0:
        kd_fileName = "treatment_pathwatDEG_{}_{}.csv".format('_'.join(db_select), sig_thresold)
        kdile_link = get_table_download_link(kd_select_long.reset_index(), kd_fileName)
        st.write("Download selected enrichment from here:")
        st.markdown(kdile_link,unsafe_allow_html=True)

    st.subheader("Look into a pathway")
    selected_pathway = st.selectbox("Select a pathway from filtered list: ", list(kd_select.index))
    geneList = set(';'.join(kd_select_long[kd_select_long.Term== selected_pathway].Lead_genes.values).split(";"))

    annot = sampleInfo.sort_values(by=['Treatments','Age_Group'])
    data_path = normCount.loc[geneList][annot.index].copy()
    glistHM = sns.clustermap(data_path.apply(sT.zscore,axis=1),cmap="viridis",figsize=(12,6),col_cluster=False,
                                    cbar_kws={'label':'Norm. expression','orientation':'horizontal'},cbar_pos=[0.3,1.1,0.35,0.03])
    st.pyplot(glistHM)

    st.write("## Data driven functional enrichment from Transcriptional Modules")
    dd_MT = pd.read_pickle("data_7June2023/DD_MTenrichment_treatment.pkl")
    sel_mt = st.selectbox("MT definition :", dd_MT.dataset.unique())
    mt_sig = dd_MT[dd_MT.dataset==sel_mt].pivot_table(index='Term',columns='sampleName',values="NES")
    mt_heatmap = sns.clustermap(mt_sig.fillna(0),vmin=-2,vmax=2, cmap="coolwarm",figsize=(4,5),col_cluster=False,
                                cbar_kws={'label':'NES','orientation':'horizontal'},cbar_pos=[0.3,1.1,0.35,0.03]) 
    st.pyplot(mt_heatmap)
    st.subheader("Look into a Module ")
    selected_module = st.selectbox("Select a modules from filtered list: ", list(mt_sig.index))
    geneList_module = set(';'.join(dd_MT[dd_MT.Term== selected_module].Lead_genes.values).split(";"))

    data_path_MT = normCount.loc[geneList_module][annot.index].copy()
    glistHM_module= sns.clustermap(data_path_MT.apply(sT.zscore,axis=1),cmap="viridis",figsize=(12,6),col_cluster=False,
                                    cbar_kws={'label':'Norm. expression','orientation':'horizontal'},cbar_pos=[0.3,1.1,0.35,0.03])
    st.pyplot(glistHM_module)

with deg_ageGroup:
    st.markdown("""
    Comparing for Age group (Old , Young). DEG w.r.t medium for five threatment conditions fron each age group was generated.
    Filter a subset based on significance (or non significance).
    """)
    deg_AG = pd.read_pickle("data_7June2023/humancDC_DEG_AgeGroupTreatment.pkl.gzip",compression="gzip")

    fdrAG = deg_AG.loc['fdr','dfs']
    fcAG = deg_AG.loc['fc','dfs']

    cutoff_threshold_AG = st.selectbox("FDR threshold for signifiance of Age Group:",[0.005,0.001,0.0005,0.0001,0.00005,0.05])
    significant_dict_AG = {}
    coutStat = []
    st.write("Number of significant DEG at selcted FDR  [@ |FC| > 1.5]:", cutoff_threshold)
    for k in fdrAG.columns:
        filterGenes = fdrAG[fdrAG[k]<cutoff_threshold_AG].dropna().index
        significant_dict[k] = filterGenes
        fc_tmp = fcAG[k].loc[filterGenes]
        fc_tmpsig = fc_tmp[abs(fc_tmp)>1.5].index
        significant_dict_AG[k] = fc_tmpsig
        coutStat.append([k[:3], k[4:] , fc_tmpsig.shape[0]])
    
    coutStat = pd.DataFrame(coutStat,columns=['AgeGroup','Treatment','DEG_count'])
    st.write(coutStat.pivot_table(index='Treatment',columns='AgeGroup',values='DEG_count'))

    age_conditionSel = st.multiselect("Choose keys to compare in upset plot",list(significant_dict_AG.keys()), ['Old_ASP1','Yng_ASP1'])
    dictplot = {k: significant_dict_AG[k] for k in age_conditionSel}
    ax_AG = plt.figure(figsize=(4,2))
    upsetplot.plot(from_contents(dictplot),min_subset_size=1,show_counts=True, fig =ax_AG)
    st.pyplot(ax_AG)

    st.subheader("Filter significant DEGs for treatment conditions/ combinations for Age Groups")
    sigcols_Ag = st.multiselect("Significant in [Age Group]", list(fdrAG.columns), ['Old_ASP1','Yng_ASP1'])
    nonsigcols_Ag = st.multiselect("Not significant in [Age Group]", list(set(fdrAG.columns)-set(sigcols_Ag)))

    sigcase_ag = fdrAG[sigcols_Ag][fdrAG[sigcols_Ag]<cutoff_threshold_AG].dropna().index
    tmp_ag     = fdrAG.drop(sigcols_Ag,axis=1).loc[sigcase_ag]
    filter_sig_ag = tmp_ag[tmp_ag[tmp_ag>=cutoff_threshold_AG].fillna(0).astype(bool).sum(axis=1)==len(nonsigcols_Ag)].index
    filter_sig_ag = list(set(normCount.index).intersection(set(filter_sig_ag)))

    st.write("Significant in :"," : ".join(sigcols_Ag),sigcase_ag.shape, "not in", ":".join(nonsigcols_Ag), "Gene Found :", len(filter_sig_ag), len(nonsigcols_Ag))

    if len(filter_sig_ag) >=1:
        flName_treatSig_AG = "AG_DEG_FC_{}.csv".format("_".join(sigcols_Ag))
        fc_sig_AG = fcAG.loc[filter_sig_ag][sigcols_Ag]
        degTreat_link_AG = get_table_download_link(fc_sig_AG.reset_index(), flName_treatSig_AG)
        st.write("Download DEG LIST HERE: ", fc_sig_AG.shape)
        st.markdown(degTreat_link_AG, unsafe_allow_html=True)

        # f_treatment = [i[4:] for i in filter_sig_ag]
        # if len(filter_sig_ag) >= 5:
        #     _samplesfor_heatmap = sampleInfo[(sampleInfo.Treatments.isin(f_treatment))|(sampleInfo.Treatments=='medium')].sort_values(by=['Treatments','Age_Group']).index
        #     data_hm = normCount.loc[filter_sig_ag][_samplesfor_heatmap]
        #     sns.set(font_scale=1.2)
        #     hm_treat = sns.clustermap(data_hm,cmap="coolwarm",col_cluster=False,z_score=0,vmin=-1,vmax=1,
        #        cbar_pos=[1,0.6,0.03,0.2],cbar_kws={"label":'norm expression (z-score)'},figsize=(20,20))
        #     st.pyplot(hm_treat)
        # else:
        #     st.write("Not enough genes for heatmap [min deg >=5]")

        geneSelect_AG = st.selectbox("Select Gene for normExpression:", filter_sig_ag)
        bplotdData_AG = normCount.loc[geneSelect_AG].to_frame().join(sampleInfo[['Treatments','Age_Group']])
        gxpr_AG, gbox_AG = plt.subplots(1,1,figsize=(6,4))
        sns.boxplot(x='Treatments',y=geneSelect_AG,data=bplotdData_AG,\
                                    hue='Age_Group',**PROPS,ax=gbox_AG)
        plt.ylabel("Expression profile of {}".format(geneSelect_AG),fontdict={'size':14})
        st.pyplot(gxpr_AG)

        st.subheader("Gene enrichment analysis with known pathway annotation")
        kd_ag_pathway =  pd.read_pickle("data_7June2023/KD_AgeGrpTreatmentPathway.pkl.gzip",compression="gzip")
        ag_pathwayselect = st.selectbox("choose dataset for enrichment study:",list(kd_ag_pathway.dataset.unique()))
        ag_pathwaySig = st.selectbox("pathway significance with FDR:", [0.005,0.001,0.05,0.01])
        kd_ag_select = kd_ag_pathway[(kd_ag_pathway['dataset'] ==ag_pathwayselect) & (kd_ag_pathway['FDR q-val']<ag_pathwaySig)]
        dxAG_PT = kd_ag_select.pivot_table(index='Term',columns='sampleInfo',values="NES")
        sns.set(font_scale=1)
        ag_KD_pathwatHeatmap = sns.clustermap(dxAG_PT.fillna(0),figsize=(6,8),cmap="viridis",
                                    cbar_kws={'label':'NES','orientation':'horizontal'},cbar_pos=[0.3,1.1,0.35,0.03])
        st.pyplot(ag_KD_pathwatHeatmap)

        if kd_ag_select.shape[0] > 0:
            ag_kd_fileName = "treatment_pathwatDEG_{}_{}.csv".format(ag_pathwayselect, ag_pathwaySig)
            ag_kdile_link = get_table_download_link(kd_ag_select.reset_index(), ag_kd_fileName)
            st.write("Download selected enrichment from here:")
            st.markdown(ag_kdile_link,unsafe_allow_html=True)

        st.subheader("Look into a pathway from above list")
        selected_pathway_AG = st.selectbox("Select a pathway from filtered list [Age Group]: ", list(dxAG_PT.index))
        geneList_AG = set(';'.join(kd_ag_select[kd_ag_select.Term== selected_pathway_AG].Lead_genes.values).split(";"))

        data_path_AG = normCount.loc[geneList_AG][sampleInfo.index].copy().T
        st.write("genes involved:", ','.join(geneList_AG))
        x = data_path_AG.unstack().reset_index().rename({'level_1':'sampleID',0:'normExpr'},axis=1).set_index('sampleID').join(sampleInfo[['Age_Group','Treatments']])
        x['AG_treat'] = x['Age_Group']+"_"+x['Treatments']
        x_agg  = x.groupby(['AG_treat','geneSymbol'])['normExpr'].agg(['median']).reset_index().pivot_table(index='geneSymbol',columns='AG_treat',values='median')

        glistHM_AG = sns.clustermap(x_agg,cmap="viridis",figsize=(5,8),col_cluster=True,vmin=0.5,vmax=10,
                                        cbar_kws={'label':'median expression profile (normalized)','orientation':'horizontal'},cbar_pos=[0.3,1.1,0.35,0.03])
        st.pyplot(glistHM_AG)



with downloadData:
    ## get all files
    st.subheader("Data used")
    flname = "human_cDC_normExpr.csv"
    filelink = get_table_download_link(normCount.reset_index(),flname)
    st.subheader("Normalized count file: ", )
    st.markdown(filelink,unsafe_allow_html=True)

    st.subheader("Sample metadata file: ", )
    sampleFlnamelink = get_table_download_link(sampleInfo.reset_index(),"human_cDC_sampleInfo.csv")
    st.markdown(sampleFlnamelink,unsafe_allow_html=True)

    st.subheader("Methods and Reference")
    st.markdown("""
    
    1. DC modules: Banchereau, Romain, et al. "Transcriptional specialization of human dendritic cell subsets in response to microbial vaccines."
     Nature communications 5.1 (2014): 5283.

    2. BloodGen 3: Rinchai, D., Roelands, J., Toufiq, M., Hendrickx, W., Altman, M. C., Bedognetti, D., & Chaussabel, D. (2021). BloodGen3Module: \
    blood transcriptional module repertoire analysis and visualization using R. Bioinformatics, 37(16), 2382-2389

    """)
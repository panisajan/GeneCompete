# pip install numpy==1.22.4
# pip install scipy==1.10.1
import streamlit as st
#st.set_option('deprecation.showfileUploaderEncoding', False)
import time
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy import sparse
from scipy.sparse.linalg import eigs
import sknetwork
from sknetwork.ranking import PageRank
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power
from scipy.sparse.linalg import eigs

@st.cache_data
def GeneCompete_Intersect(table,name,method,reg):
    import pandas as pd
    import numpy as np
    from numpy.linalg import inv
    from scipy import sparse
    from scipy.sparse.linalg import eigs
    import sknetwork
    from sknetwork.ranking import PageRank
    from fast_pagerank import pagerank
    from fast_pagerank import pagerank_power
    from scipy.sparse.linalg import eigs

    check_list = list(set(l.index) for l in (table))
    intersect_set = set(table[0].index)
    for s in check_list:
        intersect_set = intersect_set.intersection(s)
    N = len(intersect_set) ## number of games
    
    @st.cache_data
    def winloss_up(data):
        dat_fil = data.loc[list(intersect_set),]
        win = np.transpose(np.sign(np.sign((np.array(dat_fil[name])[None,:] - np.array(dat_fil[name])[:,None])) + 1))
        win[np.diag_indices_from(win)] = 0
        loss = abs(np.sign(win - 1))
        loss[np.diag_indices_from(loss)] = 0
        return win, loss
    
    @st.cache_data
    def winloss_down(data):
        dat_fil = data.loc[list(intersect_set),]
        win = np.transpose(np.sign(np.sign((np.array(dat_fil[name])[:,None] - np.array(dat_fil[name])[None,:])) + 1))
        win[np.diag_indices_from(win)] = 0
        loss = abs(np.sign(win - 1))
        loss[np.diag_indices_from(loss)] = 0
        return win, loss
        
    if reg == 'Up-regulation':
        a = []
        win_combine = loss_combine = np.zeros((N, N))
        for i in range(len(table)):
            a = winloss_up(table[i])
            win_combine = win_combine + a[0]
            loss_combine = loss_combine + a[1]
            
    elif reg == 'Down-regulation':
        a = []
        win_combine = loss_combine = np.zeros((N, N))
        for i in range(len(table)):
            a = winloss_down(table[i])
            win_combine = win_combine + a[0]
            loss_combine = loss_combine + a[1]
    
    test1 = pd.DataFrame(win_combine)
    test1.index = test1.columns = intersect_set

    result = pd.DataFrame(columns=['TeamW', 'TeamL', 'Wscore', 'Lscore'])
    i, j = np.nonzero(win_combine)
    winners = np.where(win_combine[i, j] > win_combine[j, i], i, j)
    losers = np.where(win_combine[i, j] > win_combine[j, i], j, i)
    wscores = np.maximum(win_combine[i, j], win_combine[j, i])
    lscores = np.minimum(win_combine[i, j], win_combine[j, i])
    team_names = test1.columns.values
    winners = team_names[winners]
    losers = team_names[losers]
    result['TeamW'] = winners
    result['TeamL'] = losers
    result['Wscore'] = wscores
    result['Lscore'] = lscores


    from rankit.Table import Table
    data = Table(result, col = ['TeamW', 'TeamL', 'Wscore', 'Lscore'])

    if method == 'Win-loss':
        win = win_combine.sum(axis = 1)
        win_df = pd.DataFrame({'Name':list(intersect_set),'Score(Win)':win})
        win_df = win_df.sort_values(by="Score(Win)", ascending=False)
        win_df['Rank(Win)'] = range(1,len(win_df)+1)
        return win_df

    elif method == 'Massey':
        from rankit.Ranker import MasseyRanker
        ranker = MasseyRanker()
        masseyRank = ranker.rank(data)
        masseyRank.columns = ['Name','Score(Massey)','Rank(Massey)']
        return masseyRank

    elif method == 'Colley':
        from rankit.Ranker import ColleyRanker
        ranker = ColleyRanker()
        colleyRank = ranker.rank(data)
        colleyRank.columns = ['Name','Score(Colley)','Rank(Colley)']
        return colleyRank

    elif method == 'Keener':
        from rankit.Ranker import KeenerRanker
        ranker = KeenerRanker(threshold=1e-4)
        keenerRank = ranker.rank(data)
        keenerRank.columns = ['Name','Score(Keener)','Rank(Keener)']
        return keenerRank
    
    elif method == 'Elo':
        from rankit.Ranker import EloRanker
        eloRanker = EloRanker()
        eloRanker.update(data)
        eloRank = eloRanker.leaderboard()
        eloRank.columns = ['Name','Score(Elo)','Rank(Elo)']
        return eloRank    

    elif method == 'Markov':
        loss = loss_combine.sum(axis = 1)
        N_G = -win_combine
        N_G[np.diag_indices_from(N_G)] = loss
        P2 = np.vstack([N_G,np.ones((1, N))])
        z2 = np.zeros((1, N+1))
        z2[0,N] = 1
        x_qr2 = np.linalg.lstsq(P2, z2.T,rcond=None)[0]
        df_markov = pd.DataFrame(x_qr2)
        df_markov.columns = ['Score(Markov)']
        df_markov['Name'] = list(intersect_set)
        df_markov = df_markov.sort_values(by="Score(Markov)", ascending=False)
        df_markov['Rank(Markov)'] = range(1,len(df_markov)+1)
        return df_markov

    elif method == 'PageRank':
        from rankit.Ranker import MarkovRanker
        #ranker = MarkovRanker(restart=0.3, threshold=1e-4)
        ranker = MarkovRanker(restart=0.85, threshold=1e-4)
        pgRank = ranker.rank(data)
        pgRank.columns = ['Name','Score(PageRank)','Rank(PageRank)']
        return pgRank
    
    elif method == 'BiPageRank':
        A = (np.array(win_combine)).T
        sA = sparse.csr_matrix(A)
        pr_A = pagerank_power(sA, p=0.85)
        B = (np.array(loss_combine)).T
        sB = sparse.csr_matrix(B)
        pr_B = pagerank_power(sB, p=0.85)
        bipagerank = pd.DataFrame(list(pr_A-pr_B), index = intersect_set)
        bipagerank.columns = ['Score(BiPageRank)']
        bipagerank['Name'] = list(intersect_set)
        bipagerank = bipagerank.sort_values(by="Score(BiPageRank)", ascending=False)
        bipagerank['Rank(BiPageRank)'] = range(1,len(bipagerank)+1)
        return bipagerank


@st.cache_data
def GeneCompete_Union(table,name,method,reg,FC):
    import pandas as pd
    import numpy as np
    from numpy.linalg import inv
    from scipy import sparse
    from scipy.sparse.linalg import eigs
    import sknetwork
    from sknetwork.ranking import PageRank
    from fast_pagerank import pagerank
    from fast_pagerank import pagerank_power
    from scipy.sparse.linalg import eigs
    @st.cache_data
    def reorder(dat_matrix,dat_remain):
        a = pd.concat([dat_matrix, dat_remain]).fillna(0)
        a = a.reindex(union_set,columns=union_set)
        return a
    
    @st.cache_data
    def winloss_up(data):
        dat_fil = data.loc[(data.index).intersection(union_set),]
        remain = set([item for item in union_set if not(pd.isnull(item)) == True])-set(data.index)
        matrix_remain = pd.DataFrame(np.zeros((len(remain),len(remain))),index=list(remain),columns=list(remain))
        win = np.transpose(np.sign(np.sign((np.array(dat_fil[name])[None,:] - np.array(dat_fil[name])[:,None])) + 1))
        win[np.diag_indices_from(win)] = 0
        loss = abs(np.sign(win - 1))
        loss[np.diag_indices_from(loss)] = 0
        win_dat = pd.DataFrame(win,index = dat_fil.index, columns = dat_fil.index)
        loss_dat = pd.DataFrame(loss,index = dat_fil.index, columns = dat_fil.index)
        win_all = reorder(win_dat,matrix_remain)
        loss_all = reorder(loss_dat,matrix_remain)
        return win_all, loss_all
    
    @st.cache_data
    def winloss_down(data):
        dat_fil = data.loc[(data.index).intersection(union_set),]
        remain = set([item for item in union_set if not(pd.isnull(item)) == True])-set(data.index)
        matrix_remain = pd.DataFrame(np.zeros((len(remain),len(remain))),index=list(remain),columns=list(remain))
        win = np.transpose(np.sign(np.sign((np.array(dat_fil[name])[:,None] - np.array(dat_fil[name])[None,:])) + 1))
        win[np.diag_indices_from(win)] = 0
        loss = abs(np.sign(win - 1))
        loss[np.diag_indices_from(loss)] = 0
        win_dat = pd.DataFrame(win,index = dat_fil.index, columns = dat_fil.index)
        loss_dat = pd.DataFrame(loss,index = dat_fil.index, columns = dat_fil.index)
        win_all = reorder(win_dat,matrix_remain)
        loss_all = reorder(loss_dat,matrix_remain)
        return win_all, loss_all
    
    all_data1 = []
    for i in range(len(table)):
        all_data1.append(i)

    if reg == 'Up-regulation':
        for i in range(len(table)):
            all_data1[i] = (table[i])[(table[i][name] > FC)]

        check_list = list(set(l.index) for l in (all_data1))
        union_set = set(all_data1[0].index)
        for s in check_list:
            union_set = union_set.union(s)
        N = len(union_set) ## number of games
        a = []
        win_combine = loss_combine = np.zeros((N, N))
        for i in range(len(all_data1)):
            a = winloss_up(all_data1[i])
            win_combine = win_combine + a[0]
            loss_combine = loss_combine + a[1]
        
    
    elif reg == 'Down-regulation':
        for i in range(len(table)):
            all_data1[i] = (table[i])[(table[i][name] < FC)]

        check_list = list(set(l.index) for l in (all_data1))
        union_set = set(all_data1[0].index)
        for s in check_list:
            union_set = union_set.union(s)
        N = len(union_set) ## number of games 
        a = []
        win_combine = loss_combine = np.zeros((N, N))
        for i in range(len(all_data1)):
            a = winloss_down(all_data1[i])
            win_combine = win_combine + a[0]
            loss_combine = loss_combine + a[1]
    
    # N_ij
    N_ij = win_combine + loss_combine # number of games between i  and j
    
    win_combine = np.array(win_combine)
    test1 = pd.DataFrame(win_combine)
    test1.index = test1.columns = union_set
    
    result = pd.DataFrame(columns=['TeamW', 'TeamL', 'Wscore', 'Lscore'])
    i, j = np.nonzero(win_combine)
    winners = np.where(win_combine[i, j] > win_combine[j, i], i, j)
    losers = np.where(win_combine[i, j] > win_combine[j, i], j, i)
    wscores = np.maximum(win_combine[i, j], win_combine[j, i])
    lscores = np.minimum(win_combine[i, j], win_combine[j, i])
    team_names = test1.columns.values
    winners = team_names[winners]
    losers = team_names[losers]
    result['TeamW'] = winners
    result['TeamL'] = losers
    result['Wscore'] = wscores
    result['Lscore'] = lscores

    from rankit.Table import Table
    data = Table(result, col = ['TeamW', 'TeamL', 'Wscore', 'Lscore'])

    if method == 'Win-loss':
        win_perc = test1/N_ij
        win_s = win_perc.sum(axis=1)
        win_df = pd.DataFrame({'Name':list(union_set),'Score(Win)':win_s})
        win_df = win_df.sort_values(by="Score(Win)", ascending=False)
        win_df['Rank(Win)'] = range(1,len(win_df)+1)
        return win_df

    elif method == 'Massey':
        from rankit.Ranker import MasseyRanker
        ranker = MasseyRanker()
        masseyRank = ranker.rank(data)
        masseyRank.columns = ['Name','Score(Massey)','Rank(Massey)']
        return masseyRank

    elif method == 'Colley':
        from rankit.Ranker import ColleyRanker
        ranker = ColleyRanker()
        colleyRank = ranker.rank(data)
        colleyRank.columns = ['Name','Score(Colley)','Rank(Colley)']
        return colleyRank

    elif method == 'Keener':
        from rankit.Ranker import KeenerRanker
        ranker = KeenerRanker(threshold=1e-4)
        keenerRank = ranker.rank(data)
        keenerRank.columns = ['Name','Score(Keener)','Rank(Keener)']
        return keenerRank
    
    elif method == 'Elo':
        from rankit.Ranker import EloRanker
        eloRanker = EloRanker()
        eloRanker.update(data)
        eloRank = eloRanker.leaderboard()
        eloRank.columns = ['Name','Score(Elo)','Rank(Elo)']
        return eloRank

    elif method == 'Markov':
        loss = loss_combine.sum(axis = 1)
        N_G = -win_combine
        N_G[np.diag_indices_from(N_G)] = loss
        P2 = np.vstack([N_G,np.ones((1, N))])
        z2 = np.zeros((1, N+1))
        z2[0,N] = 1
        x_qr2 = np.linalg.lstsq(P2, z2.T,rcond=None)[0]
        df_markov = pd.DataFrame(((x_qr2.T*(np.array(N_ij.sum(axis=1))))).T)
        df_markov.columns = ['Score(Markov)']
        df_markov['Name'] = list(union_set)
        df_markov = df_markov.sort_values(by="Score(Markov)", ascending=False)
        df_markov['Rank(Markov)'] = range(1,len(df_markov)+1)
        return df_markov

    elif method == 'PageRank':
        from rankit.Ranker import MarkovRanker
        #ranker = MarkovRanker(restart=0.3, threshold=1e-4)
        ranker = MarkovRanker(restart=0.85, threshold=1e-4)
        pgRank = ranker.rank(data)
        pgRank.columns = ['Name','Score(PageRank)','Rank(PageRank)']
        return pgRank
    
    elif method == 'BiPageRank':
        A = (np.array(win_combine)).T
        sA = sparse.csr_matrix(A)
        pr_A = pagerank_power(sA, p=0.85)
        B = (np.array(loss_combine)).T
        sB = sparse.csr_matrix(B)
        pr_B = pagerank_power(sB, p=0.85)
        bipagerank = pd.DataFrame(list(pr_A-pr_B), index= union_set)
        bipagerank.columns = ['Score(BiPageRank)']
        bipagerank['Name'] = list(union_set)
        bipagerank = bipagerank.sort_values(by="Score(BiPageRank)", ascending=False)
        bipagerank['Rank(BiPageRank)'] = range(1,len(bipagerank)+1)
        return bipagerank

@st.cache_data
def Match(table,name,strategy,reg,FC = None):
    import pandas as pd
    import numpy as np

    if strategy == 'Intersect':
        check_list = list(set(l.index) for l in (table))
        intersect_set = set(table[0].index)
        for s in check_list:
            intersect_set = intersect_set.intersection(s)
        N = len(intersect_set) ## number of games
        
        @st.cache_data
        def winloss_up(data):
            dat_fil = data.loc[list(intersect_set),]
            win = np.transpose(np.sign(np.sign((np.array(dat_fil[name])[None,:] - np.array(dat_fil[name])[:,None])) + 1))
            win[np.diag_indices_from(win)] = 0
            loss = abs(np.sign(win - 1))
            loss[np.diag_indices_from(loss)] = 0
            return win, loss
        
        @st.cache_data
        def winloss_down(data):
            dat_fil = data.loc[list(intersect_set),]
            win = np.transpose(np.sign(np.sign((np.array(dat_fil[name])[:,None] - np.array(dat_fil[name])[None,:])) + 1))
            win[np.diag_indices_from(win)] = 0
            loss = abs(np.sign(win - 1))
            loss[np.diag_indices_from(loss)] = 0
            return win, loss
            
        if reg == 'Up-regulation':
            a = []
            win_combine = loss_combine = np.zeros((N, N))
            for i in range(len(table)):
                a = winloss_up(table[i])
                win_combine = win_combine + a[0]
                loss_combine = loss_combine + a[1]
                
        elif reg == 'Down-regulation':
            a = []
            win_combine = loss_combine = np.zeros((N, N))
            for i in range(len(table)):
                a = winloss_down(table[i])
                win_combine = win_combine + a[0]
                loss_combine = loss_combine + a[1]
        
        win_combine = pd.DataFrame(win_combine)
        win_combine.index = win_combine.columns = intersect_set

        loss_combine = pd.DataFrame(loss_combine)
        loss_combine.index = loss_combine.columns = intersect_set

    elif strategy == 'Union':
        
        @st.cache_data
        def reorder(dat_matrix,dat_remain):
            a = pd.concat([dat_matrix, dat_remain]).fillna(0)
            a = a.reindex(union_set,columns=union_set)
            return a
        
        @st.cache_data
        def winloss_up(data):
            dat_fil = data.loc[(data.index).intersection(union_set),]
            remain = set([item for item in union_set if not(pd.isnull(item)) == True])-set(data.index)
            matrix_remain = pd.DataFrame(np.zeros((len(remain),len(remain))),index=list(remain),columns=list(remain))
            win = np.transpose(np.sign(np.sign((np.array(dat_fil[name])[None,:] - np.array(dat_fil[name])[:,None])) + 1))
            win[np.diag_indices_from(win)] = 0
            loss = abs(np.sign(win - 1))
            loss[np.diag_indices_from(loss)] = 0
            win_dat = pd.DataFrame(win,index = dat_fil.index, columns = dat_fil.index)
            loss_dat = pd.DataFrame(loss,index = dat_fil.index, columns = dat_fil.index)
            win_all = reorder(win_dat,matrix_remain)
            loss_all = reorder(loss_dat,matrix_remain)
            return win_all, loss_all
        
        @st.cache_data
        def winloss_down(data):
            dat_fil = data.loc[(data.index).intersection(union_set),]
            remain = set([item for item in union_set if not(pd.isnull(item)) == True])-set(data.index)
            matrix_remain = pd.DataFrame(np.zeros((len(remain),len(remain))),index=list(remain),columns=list(remain))
            win = np.transpose(np.sign(np.sign((np.array(dat_fil[name])[:,None] - np.array(dat_fil[name])[None,:])) + 1))
            win[np.diag_indices_from(win)] = 0
            loss = abs(np.sign(win - 1))
            loss[np.diag_indices_from(loss)] = 0
            win_dat = pd.DataFrame(win,index = dat_fil.index, columns = dat_fil.index)
            loss_dat = pd.DataFrame(loss,index = dat_fil.index, columns = dat_fil.index)
            win_all = reorder(win_dat,matrix_remain)
            loss_all = reorder(loss_dat,matrix_remain)
            return win_all, loss_all
        
        all_data1 = []
        for i in range(len(table)):
            all_data1.append(i)

        if reg == 'Up-regulation':
            for i in range(len(table)):
                all_data1[i] = (table[i])[(table[i][name] > FC)]

            check_list = list(set(l.index) for l in (all_data1))
            union_set = set(all_data1[0].index)
            for s in check_list:
                union_set = union_set.union(s)
            N = len(union_set) ## number of games
            a = []
            win_combine = loss_combine = np.zeros((N, N))
            for i in range(len(all_data1)):
                a = winloss_up(all_data1[i])
                #st.write(a[0])
                win_combine = win_combine + a[0]
                loss_combine = loss_combine + a[1]
            
        
        elif reg == 'Down-regulation':
            for i in range(len(table)):
                all_data1[i] = (table[i])[(table[i][name] < FC)]

            check_list = list(set(l.index) for l in (all_data1))
            union_set = set(all_data1[0].index)
            for s in check_list:
                union_set = union_set.union(s)
            N = len(union_set) ## number of games 
            a = []
            win_combine = loss_combine = np.zeros((N, N))
            for i in range(len(all_data1)):
                a = winloss_down(all_data1[i])
                win_combine = win_combine + a[0]
                loss_combine = loss_combine + a[1]
        
        # N_ij
        N_ij = win_combine + loss_combine # number of games between i  and j
        

    mask = np.triu(np.ones_like(win_combine), k=1)
    winners, losers = np.nonzero(np.logical_and(mask, ((win_combine.values > 0) | (loss_combine.values > 0))))
    result = pd.DataFrame({'Winner': win_combine.columns[winners],
                        'Loser': win_combine.columns[losers],
                        'WinScore': win_combine.values[winners, losers],
                        'LossScore': loss_combine.values[winners, losers]})

    return result, win_combine

@st.cache_data
def num_candidate(strategy,table,reg,name,FC=None):
    if strategy == 'Intersect':
        check_list = list(set(l.index) for l in (table))
        intersect_set = set(table[0].index)
        for s in check_list:
            intersect_set = intersect_set.intersection(s)
        N = len(intersect_set) ## number of games

    elif strategy == 'Union':
        all_data1 = []
        for i in range(len(table)):
            all_data1.append(i)        
        if reg == 'Up-regulation':  
            for i in range(len(table)):
                all_data1[i] = (table[i])[(table[i][name] > FC)] # logFC > 0

            check_list = list(set(l.index) for l in (all_data1))
            union_set = set(all_data1[0].index)
            for s in check_list:
                union_set = union_set.union(s)
            N = len(union_set) ## number of games

        elif reg == 'Down-regulation':
            for i in range(len(table)):
                all_data1[i] = (table[i])[(table[i][name] < FC)] # logFC < 0

            check_list = list(set(l.index) for l in (all_data1))
            union_set = set(all_data1[0].index)
            for s in check_list:
                union_set = union_set.union(s)
            N = len(union_set) ## number of games
    return N

st.title('🧬 Integration of multiple gene expression with GeneCompete 🏆')

st.subheader(" \n")
st.subheader('What is GeneCompete?')
st.write('GeneCompete is a tool to combine heterogeneous gene expression datasets to order gene importance.')

# from PIL import Image

# image = Image.open('flow1.png')

# st.image(image, caption='Steps of GeneCompete')

st.subheader('Quick start')
st.write('GeneCompete requires the following input files.')
st.write('**1. Gene expression data:** Multiple csv files where the first column is gene name. These data can be prepared by any tools.')

st.sidebar.header('Input')

table1 = st.sidebar.file_uploader('**Input file**', type='csv', accept_multiple_files=True)

import zipfile
import os
def create_zip(csv_files):
    zip_file = os.path.join(os.getcwd(), "sample_data.zip")
    with zipfile.ZipFile(zip_file, "w") as zip_obj:
        for file in csv_files:
            file_name = os.path.basename(file)
            zip_obj.write(file, file_name)
    return zip_file

csv_files = ["sample_data/dat1.csv","sample_data/dat2.csv","sample_data/dat3.csv","sample_data/dat4.csv"]
#csv_files = ["/workspace/test2_st/sample_data/dat1.csv","/workspace/test2_st/sample_data/dat2.csv","/workspace/test2_st/sample_data/dat3.csv","/workspace/test2_st/sample_data/dat4.csv"]
#selected_files = st.multiselect("Select example input files to download", csv_files)
zip_file = create_zip(csv_files)

if st.button('Preview example'):
    df_ex = pd.read_csv(csv_files[0] ,index_col=0)
    df_ex['adj.P.Val'] = df_ex['adj.P.Val'].apply(lambda x: "{:.1e}".format(x))
    df_ex['P.Value'] = df_ex['P.Value'].apply(lambda x: "{:.1e}".format(x))
    #st.dataframe(df_ex.round(2))
    st.write(df_ex)
st.download_button(label="Download as zip", data=zip_file, file_name="my_zip_file.zip", mime="application/zip")

st.write('**2. Column name:** The interested value that will be used as competing score (in the example is logFC).')
st.write('**3. Regulation:** Select Up-regulation or Down-regulation.')
st.write('**4. Strategy:** Select Intersect or Union.')
st.write('**5. logFC threshold:** If the union strategy is selected, the number of genes can be large and consume computational time. Before ranking, datasets are filtered with _logFC > (logFC threshold)_ in case of up-regulation and _logFC < (logFC threshold)_ for down-regulation.')
st.write('**6. Ranking Method:** Select Win-loss, Massey, Colley, Keener, Elo, Markov, PageRank, or Bi-PageRank')

list_table1 = list()
for table_i in table1:
    df = pd.read_csv(table_i ,index_col=0)
    #st.write(df.index)
    list_table1.append(df)

name1 = st.sidebar.text_input("**Column name**","logFC")
reg1 = st.sidebar.radio("**Regulation**", ["Up-regulation","Down-regulation"])
strategy1 = st.sidebar.radio("**Strategy**", ["Intersect","Union"])


if strategy1 == 'Union':
    FC1 = st.sidebar.slider('**logFC threshold**', -5.0, 5.0, step = 0.1)
    #FC1 = st.sidebar.number_input('**logFC threshold**')
else:
    FC1 = None

#if st.sidebar.button('Check your input data'):
st.text(" \n")
st.subheader("📁 Preparing Input:")
st.write('**:red[Total number of file uploaded:]**',len(list_table1))
if not list_table1:
    st.error('Error: Please upload files', icon="🚨")
if not name1:
    st.error('Error: Please specify column name', icon="🚨")
for i in range(len(list_table1)):
    st.write('Total number of genes in dataset',i+1,'is',len(list_table1[i]))

if list_table1 and name1:
    # df_sum = pd.DataFrame({'Strategy':strategy1,'Regulation':reg1,'logFC threshold':FC1})
    # st.write(df_sum)
    # st.write('**:red[Strategy:]**',strategy1)
    # st.write('**:red[Regulation:]**',reg1)
    # if strategy1 == 'Union':
    #     st.write('**:red[logFC threshold:]**',FC1)
    # else:
    #     FC1 = None
    can_num = num_candidate(strategy1,list_table1,reg1,name1,FC1)
    #st.write('➡️ **Number of candidate genes:**',can_num)

    st.write('👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇')
    #st.write('🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹')
## check

    if strategy1 == 'Union':
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("**:red[Number of genes:]**",can_num)
        col2.metric("**:red[Strategy:]**",strategy1)
        col3.metric("**:red[Regulation:]**", 'UP' if reg1=='Up-regulation' else 'DOWN')
        col4.metric("**:red[logFC threshold:]**",FC1)

    else:
        FC1 = None
        col1, col2, col3 = st.columns(3)
        col1.metric("**:red[Number of genes:]**",can_num)
        col2.metric("**:red[Strategy:]**",strategy1)
        col3.metric("**:red[Regulation:]**", 'UP' if reg1=='Up-regulation' else 'DOWN')

    st.write('👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆')
    #st.write('🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹')

    if can_num>10000:
        st.warning('Warning: Modify the input to reduce computational time', icon="⚠️")
    else:
        st.success('Press submit to obtain ranking score')

st.subheader("**⛹️‍♂️ Ranking scores:**")

st.write('**Who are playing?**')
match = st.button('Click')
if match:
    if not list_table1:
        st.error('Error: Please upload files', icon="🚨")
    if not name1:
        st.error('Error: Please specify column name', icon="🚨")
    if list_table1 and name1:
        mm = Match(table = list_table1,name = name1,strategy = strategy1,reg = reg1,FC = FC1)
        st.write(mm[1])
        st.write(len(mm[0]),'pair of genes are playing games.')
        st.write(mm[0])


method2 = st.selectbox("**Ranking Score (Select method)**", ["Win-loss", "Massey", "Colley","Keener","Elo","Markov","PageRank","BiPageRank"])
submit = st.button('Submit')
if submit:
    if not list_table1:
        st.error('Error: Please upload files', icon="🚨")
    if not name1:
        st.error('Error: Please specify column name', icon="🚨")
    if not method2:
        st.error('Error: Please select method(s)', icon="🚨")
    if list_table1 and name1 and method2:
        #st.text(" \n")
        # begin = time.time()
        # out = GeneCompete(table = list_table1,name = name1,strategy = strategy1,method = method1,reg = reg1,FC = FC1)
        # end = time.time()
        # #time.sleep(5)
        # st.success('Success! Here is your ranking score.')
        #st.write('Time:',end-begin)
        #with st.spinner('Please wait ...'):
        #begin = time.time()
        #score = []
        if strategy1 == 'Union':
            out1 = GeneCompete_Union(table = list_table1,name = name1,method = method2,reg = reg1,FC = FC1)
        elif strategy1 == 'Intersect':
            out1 = GeneCompete_Intersect(table = list_table1,name = name1,method = method2,reg = reg1)
        #end = time.time()
        #time.sleep(5)
        #score.append(out)
        #dfs = [df.set_index('Name') for df in score]
        #score2 = pd.concat(dfs, axis=1)
        st.success('Success! Here is your ranking score.', icon="✅")
            #st.write('Time:',end-begin)
        if strategy1 == 'Intersect':
            st.write('This is', reg1,'intersection ranking score')
        elif strategy1 == 'Union':
            st.write('This is union ranking score using')
            if reg1 == 'Up-regulation':
                st.write('Total genes with LFC >',FC1,'are',len(out1))
            elif reg1 == 'Down-regulation':
                st.write('Total genes with LFC <',FC1,'are',len(out1))
        st.write(out1)
    
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        st.download_button(label="Download data as CSV", data=convert_df(out1),file_name='GeneCompete_ranking.csv',mime='text/csv',)



method1 = st.multiselect("**Compare among methods**", ["Win-loss", "Massey", "Colley","Keener","Elo","Markov","PageRank","BiPageRank"])

compare = st.button('Compare')
       
if compare:
    if not list_table1:
        st.error('Error: Please upload files', icon="🚨")
    if not name1:
        st.error('Error: Please specify column name', icon="🚨")
    if not method1:
        st.error('Error: Please select method(s)', icon="🚨")
    if list_table1 and name1 and method1:
        #st.text(" \n")
        # begin = time.time()
        # out = GeneCompete(table = list_table1,name = name1,strategy = strategy1,method = method1,reg = reg1,FC = FC1)
        # end = time.time()
        # #time.sleep(5)
        # st.success('Success! Here is your ranking score.')
        #st.write('Time:',end-begin)
        #with st.spinner('Please wait ...'):
            #begin = time.time()
        score = []
        for met in method1:
            if strategy1 == 'Union':
                out = GeneCompete_Union(table = list_table1,name = name1,method = met,reg = reg1,FC = FC1)
            elif strategy1 == 'Intersect':
                out = GeneCompete_Intersect(table = list_table1,name = name1,method = met,reg = reg1)
                #st.write(out)    
            score.append(out)
        dfs = [df.set_index('Name') for df in score]
        score1 = pd.concat(dfs, axis=1)


        st.success('Success! Here is your ranking score.')
            #st.write('Time:',end-begin)
        if strategy1 == 'Intersect':
            st.write('This is', reg1,'intersection ranking score')
        elif strategy1 == 'Union':
            st.write('This is union ranking score using')
            if reg1 == 'Up-regulation':
                st.write('Total genes with LFC >',FC1,'are',len(score1))
            elif reg1 == 'Down-regulation':
                st.write('Total genes with LFC <',FC1,'are',len(score1))
        st.write(score1)
    
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        st.download_button(label="Download data as CSV", data=convert_df(score1),file_name='GeneCompete_ranking.csv',mime='text/csv',)

        if st.button("Clear output"):
            # Clear values from *all* all in-memory and on-disk data caches:
            # i.e. clear values from both square and cube
            st.cache_data.clear()

#!/usr/bin/env python
# coding: utf-8

# In[6]:


def GeneCompete_Union(table,name,method,reg,FC):

    import pandas as pd
    import numpy as np
    from numpy.linalg import inv
    from scipy import sparse
    from scipy.sparse.linalg import eigs
    from fast_pagerank import pagerank
    from fast_pagerank import pagerank_power
    from scipy.sparse.linalg import eigs


    def reorder(dat_matrix,dat_remain):
        a = pd.concat([dat_matrix, dat_remain]).fillna(0)
        a = a.reindex(union_set,columns=union_set)
        return a

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
        w = l = np.zeros((N, N))
        for i in range(len(all_data1)):
            a = winloss_up(all_data1[i])
            w = w + a[0]
            l = l + a[1]

    elif reg == 'Down-regulation':
        for i in range(len(table)):
            all_data1[i] = (table[i])[(table[i][name] < -FC)]

        check_list = list(set(l.index) for l in (all_data1))
        union_set = set(all_data1[0].index)
        for s in check_list:
            union_set = union_set.union(s)
        N = len(union_set) ## number of games 
        a = []
        w = l = np.zeros((N, N))
        for i in range(len(all_data1)):
            a = winloss_down(all_data1[i])
            w = w + a[0]
            l = l + a[1]

    N_ij = w + l
    w = np.array(w)
    test1 = pd.DataFrame(w)
    test1.index = test1.columns = union_set
    N = len(w) # N is the number of genes
    results = []
    if 'Win-loss' in method:
        win_perc = test1/N_ij
        win_s = win_perc.sum(axis=1)
        win_df = pd.DataFrame({'Name(Win-loss)':list(union_set),'Score(Win-loss)':list(win_s)})
        win_df = win_df.sort_values(by="Score(Win-loss)", ascending=False)
        win_df['Rank(Win-loss)'] = range(1,len(win_df)+1)
        results.append(win_df)
    
    if 'Massey' in method:
        A = (np.array(w)).T # adjacency

        M = np.ones((N, N))*-1
        M[np.diag_indices_from(M)] = N-1
        M[N-1,:] = 1

        p_massey = sum((np.array(w.T))) - sum(np.array(w))
        p_massey[N-1] = 0

        r_massey = np.dot(inv(M) ,p_massey)
        massey_df = pd.DataFrame({'Name(Massey)':list(union_set),'Score(Massey)':r_massey})
        massey_df = massey_df.sort_values(by="Score(Massey)", ascending=False)
        massey_df['Rank(Massey)'] = range(1,len(massey_df)+1)
        results.append(massey_df)

    if 'Colley' in method:
        win = w.sum(axis = 1)
        loss = l.sum(axis = 1)
        C = -np.array(N_ij)
        C[np.diag_indices_from(C)] = N_ij.sum(axis=1)+2
        b_colley = 1+((win-loss)/2)
        r_colley = np.dot(inv(C) ,b_colley)
        colley_df = pd.DataFrame({'Name(Colley)':list(union_set),'Score(Colley)':r_colley})
        colley_df = colley_df.sort_values(by="Score(Colley)", ascending=False)
        colley_df['Rank(Colley)'] = range(1,len(colley_df)+1)
        results.append(colley_df)
    
    if 'Keener' in method:
        a_k = (w + 1) / (w + np.transpose(w) + 2)
        K_matrix = 0.5+0.5*(np.sign(a_k - 0.5) * np.sqrt(abs(2*a_k-1)))
        K_matrix[np.diag_indices_from(K_matrix)] = 0
        from scipy.sparse.linalg import eigs
        val, vec = eigs(K_matrix, k=1, which='LM')
        kk = vec[:, 0].real.flatten()
        df_keener = pd.DataFrame({'Name(Keener)':list(union_set),'Score(Keener)':kk})
        df_keener = df_keener.sort_values(by="Score(Keener)", ascending=False)
        df_keener['Rank(Keener)'] = range(1,len(df_keener)+1)
        results.append(df_keener)
    
    if 'Elo' in method:
        team_names = list(union_set)  # Replace with actual team names

        result_rows = []
        matchup_set = set()

        for i in range(len(w)):
            for j in range(len(w[i])):
                if i != j:  # Exclude diagonal elements
                    matchup_key = tuple(sorted([i, j]))  # Sort team indices to ensure consistent key
                    if matchup_key not in matchup_set:
                        winner = team_names[i] if w[i, j] > w[j, i] else team_names[j]
                        loser = team_names[j] if w[i, j] > w[j, i] else team_names[i]
                        wscore = max(w[i, j], w[j, i])
                        lscore = min(w[i, j], w[j, i])
                        result_rows.append({'TeamW': winner, 'TeamL': loser, 'Wscore': wscore, 'Lscore': lscore})
                        matchup_set.add(matchup_key)

        result = pd.DataFrame(result_rows)
        result = result[(result['Wscore'] != 0) | (result['Lscore'] != 0)]
        from rankit.Table import Table
        data = Table(result, col=['TeamW', 'TeamL', 'Wscore', 'Lscore'])

        from rankit.Ranker import EloRanker
        eloRanker = EloRanker()
        eloRanker.update(data)
        eloRank = eloRanker.leaderboard()
        eloRank.columns = ['Name','Score(Elo)','Rank(Elo)']
        results.append(eloRank)
    
    if 'Markov' in method:
        loss = l.sum(axis = 1)
        N_G = -np.array(w)
        N_G[np.diag_indices_from(N_G)] = loss
        P2 = np.vstack([N_G,np.ones((1, N))])
        z2 = np.zeros((1, N+1))
        z2[0,N] = 1
        x_qr2 = np.linalg.lstsq(P2, z2.T,rcond=None)[3]
        k_m = x_qr2.flatten()
        df_markov = pd.DataFrame({'Name(Markov)':list(union_set),'Score(Markov)':k_m})
        df_markov = df_markov.sort_values(by="Score(Markov)", ascending=False)
        df_markov['Rank(Markov)'] = range(1,len(df_markov)+1)
        #df_markov = df_markov[['Name(Markov)','Score(Markov)','Rank(Markov)']]
        results.append(df_markov)
    
    if 'PageRank' in method:
        from scipy import sparse
        from fast_pagerank import pagerank
        from fast_pagerank import pagerank_power

        A = (np.array(w)).T
        sA = sparse.csr_matrix(A)
        pr_A = pagerank_power(sA, p=0.85)
        pgRank = pd.DataFrame({'Name(PageRank)':list(union_set),'Score(PageRank)':list(pr_A)})
        pgRank = pgRank.sort_values(by="Score(PageRank)", ascending=False)
        pgRank['Rank(PageRank)'] = range(1,len(pgRank)+1)
        results.append(pgRank)
    
    if 'BiPagerank' in method:
        A = (np.array(w)).T
        sA = sparse.csr_matrix(A)
        pr_A = pagerank_power(sA, p=0.85)
        B = (np.array(l)).T
        sB = sparse.csr_matrix(B)
        pr_B = pagerank_power(sB, p=0.85)
        bipagerank = pd.DataFrame({'Name(BiPageRank)':list(union_set),'Score(BiPageRank)':list(pr_A-pr_B)})
        bipagerank = bipagerank.sort_values(by="Score(BiPageRank)", ascending=False)
        bipagerank['Rank(BiPageRank)'] = range(1,len(bipagerank)+1)
        results.append(bipagerank)
        
    final_result = pd.concat(results, axis=1)
    return final_result
    # elif method == 'all':

    #     ## win
    #     win_perc = test1/N_ij
    #     win_s = win_perc.sum(axis=1)
    #     win_df = pd.DataFrame({'Name':list(union_set),'Score(Win)':win_s})
    #     win_df = win_df.sort_values(by="Score(Win)", ascending=False)
    #     win_df['Rank(Win)'] = range(1,len(win_df)+1)

    #     # Massey

    #     A = (np.array(w)).T # adjacency

    #     M = np.ones((N, N))*-1
    #     M[np.diag_indices_from(M)] = N-1
    #     M[N-1,:] = 1

    #     p_massey = sum((np.array(w.T))) - sum(np.array(w))
    #     p_massey[N-1] = 0

    #     r_massey = np.dot(inv(M) ,p_massey)
    #     massey_df = pd.DataFrame(r_massey,index= list(union_set))
    #     massey_df.columns = ['Score(Massey)']
    #     massey_df['Name'] = list(union_set)
    #     massey_df = massey_df.sort_values(by="Score(Massey)", ascending=False)
    #     massey_df['Rank(Massey)'] = range(1,len(massey_df)+1)

    #     ## Colley

    #     win = w.sum(axis = 1)
    #     loss = l.sum(axis = 1)
    #     C = -np.array(N_ij)
    #     C[np.diag_indices_from(C)] = N_ij.sum(axis=1)+2
    #     b_colley = 1+((win-loss)/2)
    #     r_colley = np.dot(inv(C) ,b_colley)
    #     colley_df = pd.DataFrame(r_colley,index= list(union_set))
    #     colley_df.columns = ['Score(Colley)']
    #     colley_df['Name'] = list(union_set)
    #     colley_df = colley_df.sort_values(by="Score(Colley)", ascending=False)
    #     colley_df['Rank(Colley)'] = range(1,len(colley_df)+1)

    #     ## Keener

    #     a_k = (w + 1) / (w + np.transpose(w) + 2)
    #     K_matrix = 0.5+0.5*(np.sign(a_k - 0.5) * np.sqrt(abs(2*a_k-1)))
    #     K_matrix[np.diag_indices_from(K_matrix)] = 0
    #     from scipy.sparse.linalg import eigs
    #     val, vec = eigs(K_matrix, k=1, which='LM')
    #     df_keener = pd.DataFrame(abs(vec.real), index= list(union_set))
    #     df_keener.columns = ['Score(Keener)']
    #     df_keener['Name'] = list(union_set)
    #     df_keener = df_keener.sort_values(by="Score(Keener)", ascending=False)
    #     df_keener['Rank(Keener)'] = range(1,len(df_keener)+1)

    #     ## Elo

    #     team_names = list(union_set)  # Replace with actual team names

    #     result_rows = []
    #     matchup_set = set()

    #     for i in range(len(w)):
    #         for j in range(len(w[i])):
    #             if i != j:  # Exclude diagonal elements
    #                 matchup_key = tuple(sorted([i, j]))  # Sort team indices to ensure consistent key
    #                 if matchup_key not in matchup_set:
    #                     winner = team_names[i] if w[i, j] > w[j, i] else team_names[j]
    #                     loser = team_names[j] if w[i, j] > w[j, i] else team_names[i]
    #                     wscore = max(w[i, j], w[j, i])
    #                     lscore = min(w[i, j], w[j, i])
    #                     result_rows.append({'TeamW': winner, 'TeamL': loser, 'Wscore': wscore, 'Lscore': lscore})
    #                     matchup_set.add(matchup_key)

    #     result = pd.DataFrame(result_rows)
    #     result = result[(result['Wscore'] != 0) | (result['Lscore'] != 0)]
    #     from rankit.Table import Table
    #     data = Table(result, col=['TeamW', 'TeamL', 'Wscore', 'Lscore'])

    #     from rankit.Ranker import EloRanker
    #     eloRanker = EloRanker()
    #     eloRanker.update(data)
    #     eloRank = eloRanker.leaderboard()
    #     eloRank.columns = ['Name','Score(Elo)','Rank(Elo)']

    #     ## Markov

    #     loss = l.sum(axis = 1)
    #     N_G = -np.array(w)
    #     N_G[np.diag_indices_from(N_G)] = loss
    #     P2 = np.vstack([N_G,np.ones((1, N))])
    #     z2 = np.zeros((1, N+1))
    #     z2[0,N] = 1
    #     x_qr2 = np.linalg.lstsq(P2, z2.T,rcond=None)[0]
    #     #df_markov = pd.DataFrame(((x_qr2.T*(np.array(N_ij.sum(axis=1))))).T, index= list(union_set))
    #     df_markov = pd.DataFrame(np.dot(N_ij,x_qr2), index= list(union_set))
    #     df_markov.columns = ['Score(Markov)']
    #     df_markov['Name'] = list(union_set)
    #     df_markov = df_markov.sort_values(by="Score(Markov)", ascending=False)
    #     df_markov['Rank(Markov)'] = range(1,len(df_markov)+1)

    #     ## PageRank

    #     from scipy import sparse
    #     from fast_pagerank import pagerank
    #     from fast_pagerank import pagerank_power

    #     A = (np.array(w)).T
    #     sA = sparse.csr_matrix(A)
    #     pr_A = pagerank_power(sA, p=0.85)
    #     pgRank = pd.DataFrame(list(pr_A), index = list(union_set))
    #     pgRank.columns = ['Score(PageRank)']
    #     pgRank['Name'] = list(union_set)
    #     pgRank = pgRank.sort_values(by="Score(PageRank)", ascending=False)
    #     pgRank['Rank(PageRank)'] = range(1,len(pgRank)+1)

    #     ## BiPageRank

    #     B = (np.array(l)).T
    #     sB = sparse.csr_matrix(B)
    #     pr_B = pagerank_power(sB, p=0.85)
    #     bipagerank = pd.DataFrame(list(pr_A-pr_B), index = list(union_set))
    #     bipagerank.columns = ['Score(BiPageRank)']
    #     bipagerank['Name'] = list(union_set)
    #     bipagerank = bipagerank.sort_values(by="Score(BiPageRank)", ascending=False)
    #     bipagerank['Rank(BiPageRank)'] = range(1,len(bipagerank)+1)


    #     dfs = [win_df,massey_df,colley_df,df_keener,eloRank,df_markov,pgRank,bipagerank]

    #     # initialize the merged dataframe with the first dataframe
    #     merged_df = dfs[0]

    #     # merge the remaining dataframes one by one based on the 'name' column
    #     for df in dfs[1:]:
    #         merged_df = pd.merge(merged_df, df, on='Name')

    #     return merged_df


#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
    N = len(intersect_set) # number of games
    
    w = np.zeros((N, N)) # win matrix
    l = np.zeros((N, N)) # loss matrix
    a = np.zeros((N, N))

    if reg == 'Up-regulation':
        for i in range(len(table)):
            dat_fil = table[i].loc[list(intersect_set),]
            team_values = np.array(dat_fil[name])

            np.subtract.outer(team_values, team_values, out=a)
            np.sign(np.sign(a) + 1, out=a)
            np.fill_diagonal(a, 0)

            np.add(w, a, out=w)
            np.subtract(1, a, out=a)
            np.abs(a, out=a)
            np.fill_diagonal(a, 0)
            np.add(l, a, out=l)

    elif reg == 'Down-regulation':
        for i in range(len(table)):
            dat_fil = table[i].loc[list(intersect_set),]
            team_values = np.array(dat_fil[name])

            np.subtract.outer(team_values, team_values, out=a)
            np.sign(np.sign(a) + 1, out=a)
            np.fill_diagonal(a, 0)

            np.add(l, a, out=l)
            np.subtract(1, a, out=a)
            np.abs(a, out=a)
            np.fill_diagonal(a, 0)
            np.add(w, a, out=w)

    if method == 'Win-loss':
        win = w.sum(axis = 1)
        win_df = pd.DataFrame({'Name':list(intersect_set),'Score(Win)':win})
        win_df = win_df.sort_values(by="Score(Win)", ascending=False)
        win_df['Rank(Win)'] = range(1,len(win_df)+1)
        return win_df

    elif method == 'Massey':
        from numpy.linalg import inv
        from scipy.sparse import coo_matrix

        result = pd.DataFrame(columns=['TeamW', 'TeamL', 'Wscore', 'Lscore'])
        team_names = list(intersect_set)  # Replace with actual team names

        result_rows = []
        matchup_set = set()

        for i in range(len(w)):
            for j in range(i+1, len(w[i])):  # Iterate only over upper triangular part
                matchup_key = tuple(sorted([i, j]))  # Sort team indices to ensure consistent key
                if matchup_key not in matchup_set:
                    if w[i, j] > w[j, i]:
                        winner = team_names[i]
                        loser = team_names[j]
                        wscore = w[i, j]
                        lscore = w[j, i]
                    else:
                        winner = team_names[j]
                        loser = team_names[i]
                        wscore = w[j, i]
                        lscore = w[i, j]
                    result_rows.append({'TeamW': winner, 'TeamL': loser, 'Wscore': wscore, 'Lscore': lscore})
                    matchup_set.add(matchup_key)

        result = pd.DataFrame(result_rows)

        from rankit.Table import Table
        data = Table(result, col=['TeamW', 'TeamL', 'Wscore', 'Lscore'])
        data1 = data.table[['hidx', 'vidx', 'hscore', 'vscore', 'weight']]

        m = data1.shape[0]
        n = data.itemnum

        dat = np.zeros(m * 2, dtype=np.float64)
        col = np.zeros(m * 2, dtype=int)
        row = np.zeros(m * 2, dtype=int)
        y = np.zeros(m)

        for i, itm in enumerate(data1.itertuples(index=False, name=None)):
            row[i * 2] = i
            col[i * 2] = itm[0]
            dat[i * 2] = itm[4]
            row[i * 2 + 1] = i
            col[i * 2 + 1] = itm[1]
            dat[i * 2 + 1] = -itm[4]
            if np.abs(itm[2] - itm[3]) <= 0:
                y[i] = 0.0
            else:
                y[i] = itm[4] * (itm[2] - itm[3])

        # Construct the sparse matrix X
        X = coo_matrix((dat, (row, col)), shape=(m, n)).tocsr()
        X = X[:, np.unique(col)]
        
        M = np.ones((N, N))*-1
        M[np.diag_indices_from(M)] = N-1
        M[N-1,:] = 1

        p_sp = (X.T).dot(y)
        p_massey = p_sp
        p_massey[N-1] = 0

        r_massey = np.dot(inv(M) ,p_massey)
        massey_df = pd.DataFrame(r_massey,index= list(intersect_set))
        massey_df.columns = ['Score(Massey)']
        massey_df['Name'] = list(intersect_set)
        massey_df = massey_df.sort_values(by="Score(Massey)", ascending=False)
        massey_df['Rank(Massey)'] = range(1,len(massey_df)+1)
        return massey_df

    elif method == 'Colley':
        win = w.sum(axis = 1)
        loss = l.sum(axis = 1)
        C =np.ones((N, N))*-1
        C[np.diag_indices_from(C)] = (N-1)+2
        b_colley = 1+((win-loss)/2)
        r_colley = np.dot(inv(C) ,b_colley)
        colley_df = pd.DataFrame(r_colley,index= list(intersect_set))
        colley_df.columns = ['Score(Colley)']
        colley_df['Name'] = list(intersect_set)
        colley_df = colley_df.sort_values(by="Score(Colley)", ascending=False)
        colley_df['Rank(Colley)'] = range(1,len(colley_df)+1)
        return colley_df

    elif method == 'Keener':
        a_k = (w + 1) / (w + np.transpose(w) + 2)
        K_matrix = 0.5+0.5*(np.sign(a_k - 0.5) * np.sqrt(abs(2*a_k-1)))
        K_matrix[np.diag_indices_from(K_matrix)] = 0
        
        from scipy.sparse.linalg import eigs

        val, vec = eigs(K_matrix, k=1, which='LM')
        df_keener = pd.DataFrame(abs(vec.real), index= list(intersect_set))
        df_keener.columns = ['Score(Keener)']
        df_keener['Name'] = list(intersect_set)
        df_keener = df_keener.sort_values(by="Score(Keener)", ascending=False)
        df_keener['Rank(Keener)'] = range(1,len(df_keener)+1)
        return df_keener
    
    elif method == 'Elo':
        from numpy.linalg import inv
        from scipy.sparse import coo_matrix

        result = pd.DataFrame(columns=['TeamW', 'TeamL', 'Wscore', 'Lscore'])
        team_names = list(intersect_set)  # Replace with actual team names
        result_rows = []
        matchup_set = set()

        for i in range(len(w)):
            for j in range(i+1, len(w[i])):  # Iterate only over upper triangular part
                matchup_key = tuple(sorted([i, j]))  # Sort team indices to ensure consistent key
                if matchup_key not in matchup_set:
                    if w[i, j] > w[j, i]:
                        winner = team_names[i]
                        loser = team_names[j]
                        wscore = w[i, j]
                        lscore = w[j, i]
                    else:
                        winner = team_names[j]
                        loser = team_names[i]
                        wscore = w[j, i]
                        lscore = w[i, j]
                    result_rows.append({'TeamW': winner, 'TeamL': loser, 'Wscore': wscore, 'Lscore': lscore})
                    matchup_set.add(matchup_key)

        result = pd.DataFrame(result_rows)
        
        from rankit.Table import Table
        data = Table(result, col = ['TeamW', 'TeamL', 'Wscore', 'Lscore'])
        
        from rankit.Ranker import EloRanker
        eloRanker = EloRanker()
        eloRanker.update(data)
        eloRank = eloRanker.leaderboard()
        eloRank.columns = ['Name','Score(Elo)','Rank(Elo)']
        return eloRank    

    elif method == 'Markov':
        loss = l.sum(axis = 1)
        N_G = -w
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
        from scipy import sparse
        from fast_pagerank import pagerank
        from fast_pagerank import pagerank_power

        A = (np.array(w)).T
        sA = sparse.csr_matrix(A)
        pr_A = pagerank_power(sA, p=0.85)
        pgRank = pd.DataFrame(list(pr_A), index = intersect_set)
        pgRank.columns = ['Score(PageRank)']
        pgRank['Name'] = list(intersect_set)
        pgRank = pgRank.sort_values(by="Score(PageRank)", ascending=False)
        pgRank['Rank(PageRank)'] = range(1,len(pgRank)+1)
        return pgRank
    
    elif method == 'BiPageRank':
        A = (np.array(w)).T
        sA = sparse.csr_matrix(A)
        pr_A = pagerank_power(sA, p=0.85)
        B = (np.array(l)).T
        sB = sparse.csr_matrix(B)
        pr_B = pagerank_power(sB, p=0.85)
        bipagerank = pd.DataFrame(list(pr_A-pr_B), index = intersect_set)
        bipagerank.columns = ['Score(BiPageRank)']
        bipagerank['Name'] = list(intersect_set)
        bipagerank = bipagerank.sort_values(by="Score(BiPageRank)", ascending=False)
        bipagerank['Rank(BiPageRank)'] = range(1,len(bipagerank)+1)
        return bipagerank
    
    elif method == 'all':
        win = w.sum(axis = 1)
        win_df = pd.DataFrame({'Name':list(intersect_set),'Score(Win)':win})
        win_df = win_df.sort_values(by="Score(Win)", ascending=False)
        win_df['Rank(Win)'] = range(1,len(win_df)+1)
        
        from numpy.linalg import inv
        from scipy.sparse import coo_matrix

        result = pd.DataFrame(columns=['TeamW', 'TeamL', 'Wscore', 'Lscore'])
        team_names = list(intersect_set)  # Replace with actual team names

        result_rows = []
        matchup_set = set()

        for i in range(len(w)):
            for j in range(i+1, len(w[i])):  # Iterate only over upper triangular part
                matchup_key = tuple(sorted([i, j]))  # Sort team indices to ensure consistent key
                if matchup_key not in matchup_set:
                    if w[i, j] > w[j, i]:
                        winner = team_names[i]
                        loser = team_names[j]
                        wscore = w[i, j]
                        lscore = w[j, i]
                    else:
                        winner = team_names[j]
                        loser = team_names[i]
                        wscore = w[j, i]
                        lscore = w[i, j]
                    result_rows.append({'TeamW': winner, 'TeamL': loser, 'Wscore': wscore, 'Lscore': lscore})
                    matchup_set.add(matchup_key)

        result = pd.DataFrame(result_rows)

        from rankit.Table import Table
        data = Table(result, col=['TeamW', 'TeamL', 'Wscore', 'Lscore'])
        data1 = data.table[['hidx', 'vidx', 'hscore', 'vscore', 'weight']]

        m = data1.shape[0]
        n = data.itemnum

        dat = np.zeros(m * 2, dtype=np.float64)
        col = np.zeros(m * 2, dtype=int)
        row = np.zeros(m * 2, dtype=int)
        y = np.zeros(m)

        for i, itm in enumerate(data1.itertuples(index=False, name=None)):
            row[i * 2] = i
            col[i * 2] = itm[0]
            dat[i * 2] = itm[4]
            row[i * 2 + 1] = i
            col[i * 2 + 1] = itm[1]
            dat[i * 2 + 1] = -itm[4]
            if np.abs(itm[2] - itm[3]) <= 0:
                y[i] = 0.0
            else:
                y[i] = itm[4] * (itm[2] - itm[3])

        # Construct the sparse matrix X
        X = coo_matrix((dat, (row, col)), shape=(m, n)).tocsr()
        X = X[:, np.unique(col)]
        
        M = np.ones((N, N))*-1
        M[np.diag_indices_from(M)] = N-1
        M[N-1,:] = 1

        p_sp = (X.T).dot(y)
        p_massey = p_sp
        p_massey[N-1] = 0

        r_massey = np.dot(inv(M) ,p_massey)
        massey_df = pd.DataFrame(r_massey,index= list(intersect_set))
        massey_df.columns = ['Score(Massey)']
        massey_df['Name'] = list(intersect_set)
        massey_df = massey_df.sort_values(by="Score(Massey)", ascending=False)
        massey_df['Rank(Massey)'] = range(1,len(massey_df)+1)
    
        loss = l.sum(axis = 1)
        C =np.ones((N, N))*-1
        C[np.diag_indices_from(C)] = (N-1)+2
        b_colley = 1+((win-loss)/2)
        r_colley = np.dot(inv(C) ,b_colley)
        colley_df = pd.DataFrame(r_colley,index= list(intersect_set))
        colley_df.columns = ['Score(Colley)']
        colley_df['Name'] = list(intersect_set)
        colley_df = colley_df.sort_values(by="Score(Colley)", ascending=False)
        colley_df['Rank(Colley)'] = range(1,len(colley_df)+1)
        
        ## Keener
    
        a_k = (w + 1) / (w + np.transpose(w) + 2)
        K_matrix = 0.5+0.5*(np.sign(a_k - 0.5) * np.sqrt(abs(2*a_k-1)))
        K_matrix[np.diag_indices_from(K_matrix)] = 0
        
        from scipy.sparse.linalg import eigs

        val, vec = eigs(K_matrix, k=1, which='LM')
        df_keener = pd.DataFrame(abs(vec.real), index= list(intersect_set))
        df_keener.columns = ['Score(Keener)']
        df_keener['Name'] = list(intersect_set)
        df_keener = df_keener.sort_values(by="Score(Keener)", ascending=False)
        df_keener['Rank(Keener)'] = range(1,len(df_keener)+1)
        
        ## Elo
    
        from rankit.Ranker import EloRanker
        eloRanker = EloRanker()
        eloRanker.update(data)
        eloRank = eloRanker.leaderboard()
        eloRank.columns = ['Name','Score(Elo)','Rank(Elo)']
        
        ## Markov 
    
        N_G = -w
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
    
        ## PageRank
        
        A = (np.array(w)).T
        sA = sparse.csr_matrix(A)
        pr_A = pagerank_power(sA, p=0.85)
        pgRank = pd.DataFrame(list(pr_A), index = intersect_set)
        pgRank.columns = ['Score(PageRank)']
        pgRank['Name'] = list(intersect_set)
        pgRank = pgRank.sort_values(by="Score(PageRank)", ascending=False)
        pgRank['Rank(PageRank)'] = range(1,len(pgRank)+1)
        
        ## BiPageRank
    
        B = (np.array(l)).T
        sB = sparse.csr_matrix(B)
        pr_B = pagerank_power(sB, p=0.85)
        bipagerank = pd.DataFrame(list(pr_A-pr_B), index = intersect_set)
        bipagerank.columns = ['Score(BiPageRank)']
        bipagerank['Name'] = list(intersect_set)
        bipagerank = bipagerank.sort_values(by="Score(BiPageRank)", ascending=False)
        bipagerank['Rank(BiPageRank)'] = range(1,len(bipagerank)+1)

        
        dfs = [win_df,massey_df,colley_df,df_keener,eloRank,df_markov,pgRank,bipagerank]

        # initialize the merged dataframe with the first dataframe
        merged_df = dfs[0]

        # merge the remaining dataframes one by one based on the 'name' column
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='Name')
        return merged_df


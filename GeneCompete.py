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
# import sknetwork
# from sknetwork.ranking import PageRank
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power
from scipy.sparse.linalg import eigs
from GeneCompete_Intersect import*
from GeneCompete_Union import*

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
                all_data1[i] = (table[i])[(table[i][name] < -FC)]

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
        # all_data1 = []
        # for i in range(len(table)):
        #     all_data1.append(i)        
        # if reg == 'Up-regulation':  
        #     for i in range(len(table)):
        #         all_data1[i] = (table[i])[(table[i][name] > FC)] # logFC > 0
        # elif reg == 'Down-regulation':  
        #     for i in range(len(table)):
        #         all_data1[i] = (table[i])[(table[i][name] > -FC)] # logFC > 0
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
                all_data1[i] = (table[i])[(table[i][name] < -FC)] # logFC < 0

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
st.write('GeneCompete requires the following input files. The tutorial for GeneCompete,')

def embed_pdf(pdf_path):
    pdf_display = f'<iframe src="{pdf_path}" width="700" height="1000"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    #st.title("Streamlit PDF Viewer")
    
    pdf_path = "panisajan/GeneCompete/GeneCompete_tutorial.pdf"
    
    if st.button("GeneCompete tutorial"):
        embed_pdf(pdf_path)

if __name__ == "__main__":
    main()

st.write('**1. Gene expression data:** Multiple csv files where the first column is gene name. These data can be prepared by any tools.')

st.sidebar.header('1️⃣ Input')


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



if st.button('Data example'):
    df_ex = pd.read_csv(csv_files[0] ,index_col=0)
    df_ex['adj.P.Val'] = df_ex['adj.P.Val'].apply(lambda x: "{:.1e}".format(x))
    df_ex['P.Value'] = df_ex['P.Value'].apply(lambda x: "{:.1e}".format(x))
    #st.dataframe(df_ex.round(2))
    st.write(df_ex)
#st.download_button(label="Download as zip", data=zip_file, file_name="my_zip_file.zip", mime="application/zip")
with open(zip_file, "rb") as f:
    bytes = f.read()
    st.download_button(label="Download example as zip", data=bytes, file_name="sample.zip", mime="application/octet-stream")

st.write('**2. Competition score (must be a column name):** The interested value that will be used as competing score (in the example is logFC).')
st.write('**3. Regulation:** Select Up-regulation or Down-regulation.')
st.write('**4. Strategy:** Select Intersect or Union.')
st.write('**5. threshold:** If the union strategy is selected, the number of genes can be large and consume computational time. Before ranking, datasets are filtered with _Competition score > (threshold)_ in case of up-regulation and _Competition score < -(threshold)_ for down-regulation.')
st.write('**6. Ranking Method:** Select Win-loss, Massey, Colley, Keener, Elo, Markov, PageRank., or Bi-PageRank')

st.sidebar.write('**Gene expression data**')
# if 'list_table1' not in st.session_state:
#     st.session_state.list_table1 = []
# list_table1 = []
if 'list_table1' not in st.session_state:
    st.session_state.list_table1 = []

#if st.sidebar.button("Upload files"):
    # table1 = st.sidebar.file_uploader('**Upload here**', type='csv', accept_multiple_files=True)
    # if table1 is not None:
    #     for uploaded_file in table1:
    #         with uploaded_file:
    #             df = pd.read_csv(uploaded_file, index_col=0)
    #             st.session_state.list_table1.append(df)
uploaded_files = st.sidebar.file_uploader('**⬇️ Upload your file here ⬇️**', type='csv', accept_multiple_files=True)
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file, index_col=0)
            st.session_state.list_table1.append(df)
        except Exception as e:
            st.error(f"Error processing file: {e}")


if st.sidebar.button("Apply sample data"):
    for i in range(len(csv_files)):
        df_i = pd.read_csv(csv_files[i],index_col=0)
        df_i['adj.P.Val'] = df_i['adj.P.Val'].apply(lambda x: "{:.1e}".format(x))
        df_i['P.Value'] = df_i['P.Value'].apply(lambda x: "{:.1e}".format(x))
        #st.write(df_i)
        st.session_state.list_table1.append(df_i)

    # if table1 is not None:  # Check if files are uploaded
    #     for table_i in table1:
    #         df = pd.read_csv(table_i, index_col=0)
    #         st.session_state.list_table1.append(df)

# if st.sidebar.button("Upload files"):
#     table1 = st.sidebar.file_uploader('**Upload here**', type='csv', accept_multiple_files=True)
#     for table_i in table1:
#         df = pd.read_csv(table_i ,index_col=0)
#         #st.write(df)
#         #st.write(df.index)
#         st.session_state.list_table1.append(df)

name1 = st.sidebar.text_input("**Competition score (must be a column name)**","logFC")
reg1 = st.sidebar.radio("**Regulation**", ["Up-regulation","Down-regulation"])
strategy1 = st.sidebar.radio("**Strategy**", ["Union","Intersect"])
# FC1 = st.sidebar.slider('**logFC threshold**', 0.0, 5.0, step = 0.1)


if strategy1 == 'Union':
    FC1 = st.sidebar.slider('**threshold**', 0.0, 5.0, value=1.0,step = 0.1)
    #FC1 = st.sidebar.number_input('**logFC threshold**')
else:
    FC1 = None

st.subheader("2️⃣ Preparing Input:")
#if st.sidebar.button('Check your input data'):
st.text(" \n")
if st.button("Preview uploaded data"):
    st.write('Uploaded files:', st.session_state.list_table1)
#for i in range(len(list_table1)):
#st.write(list_table1)
st.text(" \n")
st.write('**:red[Total number of file uploaded:]**',len(st.session_state.list_table1))


if not st.session_state.list_table1:
    st.error('Error: Please upload files', icon="🚨")
if not name1:
    st.error('Error: Please specify column name', icon="🚨")
for i in range(len(st.session_state.list_table1)):
    st.write('Total number of genes in dataset',i+1,'is',len(st.session_state.list_table1[i]))

if st.session_state.list_table1 and name1:
    # df_sum = pd.DataFrame({'Strategy':strategy1,'Regulation':reg1,'logFC threshold':FC1})
    # st.write(df_sum)
    # st.write('**:red[Strategy:]**',strategy1)
    # st.write('**:red[Regulation:]**',reg1)
    # if strategy1 == 'Union':
    #     st.write('**:red[logFC threshold:]**',FC1)
    # else:
    #     FC1 = None
    can_num = num_candidate(strategy1,st.session_state.list_table1,reg1,name1,FC1)
    #st.write('➡️ **Number of candidate genes:**',can_num)

    st.write('👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇')
    #st.write('🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹')
## check
    # col1, col2, col3, col4 = st.columns(4)
    # col1.metric("**:red[Number of genes:]**",can_num)
    # col2.metric("**:red[Strategy:]**",strategy1)
    # col3.metric("**:red[Regulation:]**", 'UP' if reg1=='Up-regulation' else 'DOWN')
    # col4.metric("**:red[logFC threshold:]**",FC1)
    if strategy1 == 'Union':
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("**:red[Number of genes:]**",can_num)
        col2.metric("**:red[Strategy:]**",strategy1)
        col3.metric("**:red[Regulation:]**", 'UP' if reg1=='Up-regulation' else 'DOWN')
        col4.metric("**:red[logFC threshold:]**",FC1 if reg1=='Up-regulation' else -FC1)

    else:
        #FC1 = None
        col1, col2, col3 = st.columns(3)
        col1.metric("**:red[Number of genes:]**",can_num)
        col2.metric("**:red[Strategy:]**",strategy1)
        col3.metric("**:red[Regulation:]**", 'UP' if reg1=='Up-regulation' else 'DOWN')
        #col4.metric("**:red[logFC threshold:]**",FC1)

    st.write('👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆')
    #st.write('🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹🔷🔹')

    if can_num>10000:
        st.warning('Warning: Modify the input to reduce computational time', icon="⚠️")
    else:
        st.success('Press submit to obtain ranking score')

st.subheader("**3️⃣ Ranking scores:** ⛹️‍♂️")

# st.write('**Who are playing?**')
# match = st.button('Click')
# if match:
#     if not list_table1:
#         st.error('Error: Please upload files', icon="🚨")
#     if not name1:
#         st.error('Error: Please specify column name', icon="🚨")
#     if list_table1 and name1:
#         mm = Match(table = list_table1,name = name1,strategy = strategy1,reg = reg1,FC = FC1)
#         st.write(mm[1])
#         st.write(len(mm[0]),'pair of genes are playing games.')
#         st.write(mm[0])


# method2 = st.selectbox("**Ranking Score**", ["Win-loss", "Massey", "Colley","Keener","Elo","Markov","PageRank","BiPagerank"])
# submit = st.button('Submit')
# if submit:
#     if not list_table1:
#         st.error('Error: Please upload files', icon="🚨")
#     if not name1:
#         st.error('Error: Please specify column name', icon="🚨")
#     if not method2:
#         st.error('Error: Please select method(s)', icon="🚨")
#     if list_table1 and name1 and method2:
#         #st.text(" \n")
#         # begin = time.time()
#         # out = GeneCompete(table = list_table1,name = name1,strategy = strategy1,method = method1,reg = reg1,FC = FC1)
#         # end = time.time()
#         # #time.sleep(5)
#         # st.success('Success! Here is your ranking score.')
#         #st.write('Time:',end-begin)
#         #with st.spinner('Please wait ...'):
#         #begin = time.time()
#         #score = []
#         with st.spinner('Please wait ...'):
#             if strategy1 == 'Union':
#                 out1 = GeneCompete_Union(table = list_table1,name = name1,method = method2,reg = reg1,FC = FC1)
#             elif strategy1 == 'Intersect':
#                 out1 = GeneCompete_Intersect(table = list_table1,name = name1,method = method2,reg = reg1, FC=None)
#         #end = time.time()
#         #time.sleep(5)
#         #score.append(out)
#         #dfs = [df.set_index('Name') for df in score]
#         #score2 = pd.concat(dfs, axis=1)
#         st.success('Success! Here is your ranking score.', icon="✅")
#             #st.write('Time:',end-begin)
#         if strategy1 == 'Intersect':
#             st.write('This is', reg1,'intersection ranking score')
#         elif strategy1 == 'Union':
#             st.write('This is union ranking score using')
#             if reg1 == 'Up-regulation':
#                 st.write('Total genes with LFC >',FC1,'are',len(out1))
#             elif reg1 == 'Down-regulation':
#                 st.write('Total genes with LFC <',FC1,'are',len(out1))
#         st.write(out1)
    
#         @st.cache_data
#         def convert_df(df):
#             return df.to_csv().encode('utf-8')

#         st.download_button(label="Download data as CSV", data=convert_df(out1),file_name='GeneCompete_ranking.csv',mime='text/csv',)



method1 = st.multiselect("**Select ranking method(s)**", ["Win-loss", "Massey", "Colley","Keener","Elo","Markov","PageRank","BiPagerank"])


# Your existing code for applying sample data and uploading files

compare = st.button('Submit')
       
if compare:
    if not st.session_state.list_table1:
        st.error('Error: Please upload files', icon="🚨")
    # if not list_table1:
    #     st.error('Error: Please upload files', icon="🚨")
    if not name1:
        st.error('Error: Please specify column name', icon="🚨")
    if not method1:
        st.error('Error: Please select method(s)', icon="🚨")
    if st.session_state.list_table1 and name1 and method1:
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
        with st.spinner('Please wait ...'):
            if strategy1 == 'Intersect':
                out = GeneCompete_Intersect(table = st.session_state.list_table1,name = name1,method = method1,reg = reg1,FC=None)
            elif strategy1 == 'Union':
                out = GeneCompete_Union(table = st.session_state.list_table1,name = name1,method = method1,reg = reg1,FC=FC1)
        # for met in method1:
        #     if strategy1 == 'Union':
        #         out = GeneCompete_Union(table = list_table1,name = name1,method = met,reg = reg1,FC = FC1)
        #     elif strategy1 == 'Intersect':
        #         out = GeneCompete_Intersect(table = list_table1,name = name1,method = met,reg = reg1)
        #         #st.write(out)    
        #     score.append(out)
        # dfs = [df.set_index('Name') for df in score]
        # score1 = pd.concat(dfs, axis=1)
    
    
        st.success('Success! Here is your ranking score.')
            #st.write('Time:',end-begin)
        if strategy1 == 'Intersect':
            st.write('This is', reg1,'intersection ranking score')
        elif strategy1 == 'Union':
            st.write('This is union ranking score using')
            if reg1 == 'Up-regulation':
                st.write('Total genes with LFC >',FC1,'are',len(out))
            elif reg1 == 'Down-regulation':
                st.write('Total genes with LFC <',FC1,'are',len(out))
        st.write(out)
    
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')
    
        st.download_button(label="Download data as CSV", data=convert_df(out),file_name='GeneCompete_ranking.csv',mime='text/csv',)
    
        # if st.button("Clear output"):
        #     # Clear values from *all* all in-memory and on-disk data caches:
        #     # i.e. clear values from both square and cube
        # st.cache_data.clear()

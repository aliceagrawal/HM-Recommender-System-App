import streamlit as st
import pandas as pd
import pickle
import surprise
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

# Header
header_image = Image.open('Images/h&mBanner.jpeg')
st.image(header_image)

# Creating sidebar comments
st.sidebar.title('H&M article Recommendations')

st.sidebar.caption('By [Alice Agrawal](https://www.linkedin.com/in/alice-agrawal/)')

# Load in appropriate DataFrames, user ratings
articles_df = pd.read_csv('Data/articles.csv.zip', index_col='article_id')
articles_df2 = pd.read_csv('Data/articles.csv.zip')

# Customer data for collabortive filtering
df_customer = pd.read_csv('Data/df_customer.csv', index_col='customer_id')

# Meta data for collabortive filtering
transactions = pd.read_csv('Data/out.zip')

# Meta data for content based
meta_data = pd.read_csv('Data/out_content.zip')

# Import final collab model
collab_model = pickle.load(open('Model/collaborative_model.sav', 'rb'))
# st.write(articles_df['article_id'] == 893059004)


# Def function using model to return recommendations - collaborative filtering
def customer_article_recommend(customer,n_recs):
    
    have_bought = list(df_customer.loc[customer, 'article_id'])
    not_bought = articles_df.copy()
    # [not_bought.drop(x, inplace=True) for x in have_bought]
    not_bought.drop(have_bought, inplace=True)
    not_bought.reset_index(inplace=True)
    not_bought['est_purchase'] = not_bought['article_id'].apply(lambda x: collab_model.predict(customer, x).est)
    not_bought.sort_values(by='est_purchase', ascending=False, inplace=True)
    
    not_bought.rename(columns={'prod_name':'Product Name', 'author':'Author',
                               'product_type_name':'Product Type Name', 'product_group_name':'Product Group Name',
                               'index_group_name':'Index Group Name', 'garment_group_name ':'Garment Group Name'}, inplace=True)
    not_bought = not_bought.iloc[:100, :]
    not_bought.drop(['product_code', 'product_type_no', 'graphical_appearance_no','graphical_appearance_name', 'colour_group_code', 'colour_group_name',
    'perceived_colour_value_id', 'perceived_colour_value_name','perceived_colour_master_id', 'perceived_colour_master_name',
    'department_no', 'department_name', 'index_code', 'index_name','index_group_no', 'section_no', 'section_name',
    'garment_group_no', 'detail_desc','est_purchase'], axis=1, inplace=True)
    # not_bought = not_bought[['article_id','Product Name', 'Product Type Name', 'Product Group Name', 'Index Group Name', 'Garment Group Name']]
    not_bought = not_bought.sample(frac=1).reset_index(drop=True)
    
    return not_bought.head(n_recs)

# Second function for content based recommendations
def article_recommend(article_input, n_recs2):
    # st.write(article_input[0][0])
    article = articles_df2[articles_df2['article_id'] == article_input].index
    y = np.array(meta_data.loc[article]).reshape(1, -1)
    # st.write(article)
    cos_sim = cosine_similarity(meta_data, y)
    cos_sim = pd.DataFrame(data=cos_sim, index=meta_data.index)
    cos_sim.sort_values(by = 0, ascending = False, inplace=True)
    results = cos_sim.index.values
    # results = cos_sim.index.values[1:n_recs2+1]
    results_df = articles_df2.loc[results]
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'prod_name':'Product Name', 'author':'Author',
                               'product_type_name':'Product Type Name', 'product_group_name':'Product Group Name',
                               'index_group_name':'Index Group Name', 'garment_group_name ':'Garment Group Name'}, inplace=True)
    results_df = results_df.iloc[:100, :]
    results_df.drop(['product_code', 'product_type_no', 'graphical_appearance_no','graphical_appearance_name', 'colour_group_code', 'colour_group_name',
    'perceived_colour_value_id', 'perceived_colour_value_name','perceived_colour_master_id', 'perceived_colour_master_name',
    'department_no', 'department_name', 'index_code', 'index_name','index_group_no', 'section_no', 'section_name',
    'garment_group_no', 'detail_desc', 'index'], axis=1, inplace=True)
    
    results_df = results_df.sample(frac=1).reset_index(drop=True)
    
    
    return results_df.head(n_recs2)

# print the image of the articles recommended
def print_image_cf(results_cf, n_recs):
    f, ax = plt.subplots(1, n_recs, figsize=(100,50))
    i = 0
    article_id_cf = results_cf['article_id']
    for index, data in enumerate(article_id_cf):
        desc = articles_df2[articles_df2['article_id'] == data]['detail_desc'].iloc[0]
        desc_list = desc.split(' ')
        for j, elem in enumerate(desc_list):
            if j > 0 and j % 5 == 0:
                desc_list[j] = desc_list[j] + '\n'
        desc = ' '.join(desc_list)
        img = mpimg.imread(f'Data/h-and-m-personalized-fashion-recommendations/images/0{str(data)[:2]}/0{int(data)}.jpg')
        ax[i].imshow(img)
        ax[i].set_xticks([], [])
        ax[i].set_yticks([], [])
        ax[i].grid(False)
        ax[i].set_xlabel(desc, fontsize=90)
        i += 1
    return plt.show()

# print the image of the articles recommended by CB
def print_image_cf(results_cf, n_recs):
    f, ax = plt.subplots(1, n_recs, figsize=(100,50))
    i = 0
    article_id_cf = results_cf['article_id']
    for index, data in enumerate(article_id_cf):
        desc = articles_df2[articles_df2['article_id'] == data]['detail_desc'].iloc[0]
        desc_list = desc.split(' ')
        for j, elem in enumerate(desc_list):
            if j > 0 and j % 5 == 0:
                desc_list[j] = desc_list[j] + '\n'
        desc = ' '.join(desc_list)
        img = mpimg.imread(f'Data/h-and-m-personalized-fashion-recommendations/images/0{str(data)[:2]}/0{int(data)}.jpg')
        ax[i].imshow(img)
        ax[i].set_xticks([], [])
        ax[i].set_yticks([], [])
        ax[i].grid(False)
        ax[i].set_xlabel(desc, fontsize=90)
        i += 1
    return plt.show()

st.sidebar.subheader('This recommendation system can make two forms of recommendations.')
st.sidebar.write('Existing customers looking for articles they might like.')
st.sidebar.write('Similiar articles based on article input by customer.')

st.title('H&M Recommender System')
st.subheader('This app is a article recommender system for H&M')
st.subheader("See the sidebar navigation for options")

page_names = ['Existing Customer', 'Similar Article']
page = st.sidebar.radio('Navigation', page_names)

st.sidebar.caption('Please refer to my [Github](https://github.com/aliceagrawal/HM-Recommender-System-App) for reference to the code.')



if page == 'Existing Customer':
    st.header("You chose the existing customer option.")
    customer_input = st.text_input("Please input your unique Customer ID.")
    n_recs = st.number_input("Please enter the number of article recommendations you would like.", max_value=20)
    rec_button = st.button("Get some recommendations...")
    if rec_button:
        results = customer_article_recommend(customer_input, n_recs)
        st.table(results)
        result_image = print_image_cf(results, n_recs)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(result_image)


else:
    st.header("You chose the similar articles option.")
    article_input = st.number_input("Please enter a article ID.", max_value=959461001)
    # article = articles_df2.index[articles_df2['article_id'] == article_input]
    n_recs2 = st.number_input("Please enter the number of recommendations you would like.", max_value=20, key=2)
    book_button = st.button("Get some recommendations...", key=2)
    if book_button:
        results2 = article_recommend(article_input, n_recs2)
        st.table(results2)
        result_image2 = print_image_cf(results2, n_recs2)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(result_image2)


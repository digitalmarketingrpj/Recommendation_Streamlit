import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
#import matplotlib
#matplotlib.use('Agg')



st.title("Recommendation System")
st.write("----------------------------------------------------")

def get_data():
	return pd.read_csv('top_k_scores.csv')

def get_score_data():
    return pd.read_csv('scores.csv')

def get_test_df_data():
    return pd.read_csv('test_df.csv')

def get_train_data():
    return pd.read_csv('train_valid_df.csv')


#st.subheader("Select an image from drop down menu :")
snacks=pd.read_csv('snacks.csv')
snackid=snacks['SnackId'].unique().tolist()
#print(len(snackid))
snack_cat_names=['Crunchy&Salty','Sweet','Crunchy&Salty','Crunchy&Sweet','Salty','Sweet','Salty','Other','Other','Sweet','Salty','Crunchy&Salty',
                'Crunchy&Sweet','Crunchy&Sweet','Sweet','Salty','Other','Other','Sweet','Crunchy&Sweet','Sweet','Crunchy&Salty','Sweet','Other','Crunchy&Sweet',
                'Crunchy&Sweet','Sweet','Other','Other','Sweet']
snacks_names = ['Chips', 'Cookies', 'Mixed Nuts', 'Fruit Snacks','Crackers', 'Muffins', 'Popcorns','David Sunflower Seeds','Raisinets','Welch Fruit Snacks','Premium Saltines','Corn Nuts',
                'Popchips','Cinnamon Almonds','Donettes','Chicken in a Biskit','Swedish Fish','Twizzlers','Hostess Pies','Honey Maid Grahams','Zebra Cakes','Kettle Chips','Milanos','Starburst',
                'Corn Rice Cakes','Quaker cakes','M&M','Hummus','Carrots','Skim Latte']
snack_calories=['250','300','150','200','195','300','100','55','58','150','80','50','60','40','250','197','65','150','250','95','190','200','100','79','105',
                '68','205','35','40','45']
snack_rate=['$ 2.00','$ 2.00','$ 2.00','$ 2.00','$ 2.00','$ 2.00','$ 2.00','$ 1.99','$ 2.99','$ 2.49','$ 2.99','$ 2.49','$ 2.99','$ 2.99','$ 4.99','$ 3.49',
            '$ 2.89','$ 2.49','$ 3.99','$ 2.79','$ 2.29','$ 2.49','$ 2.99','$ 3.00','$ 2.49','$ 2.99','$ 2.99','$ 4.99','$ 3.49','$ 2.89']
snack_image=['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg','11.jpg','12.jpg','13.jpg','14.jpg','15.jpg','16.jpg','17.jpg',
            '18.jpg','19.jpg','20.jpg','21.jpg','22.jpg','23.jpg','24.jpg','25.jpg','26.jpg','27.jpg','28.jpg','29.jpg','30.jpg']

snack_names={'SnackId':snackid,'Category': snack_cat_names,'Names':snacks_names,'Calories':snack_calories,'Price':snack_rate,'Image':snack_image}

df_snacks = pd.DataFrame (snack_names, columns = ['SnackId','Category','Names','Calories','Price','Image'])
#print(df_snacks)


n=1
df=get_data()
train_df = get_train_data()
test_df=get_test_df_data()
scores= get_score_data()
com_df = df.merge(df_snacks, on = 'SnackId')
print(com_df)
avg_df = com_df.groupby('Names').mean().reset_index()[['Names','Prediction']]
train_snack = train_df.merge(df_snacks, on ='SnackId')
num_of_users = train_snack.groupby('Names').count().reset_index()[['Names','UserId']]

num_of_users=num_of_users.rename(columns={'Names':'Snacks','UserId':'Number of Reviews'})

df=df.sort_values(by=['UserId', 'Prediction'], ascending=False)


snacks_cat=df_snacks['Category'].unique()

st.sidebar.title("Recommendation System") 
select = st.sidebar.selectbox( 'Analyzing FastAI algorithm with simple snack dataset?' ,
                ('Recommendations','Recommendations based on Categories','Graphs'))


if select  == 'Recommendations based on Categories':
    st.subheader("Please enter User Name:")
    user=st.text_input('User Name:','A1033RWNZWEMR5')
    st.subheader("Select a Category from drop down menu :")
    cat=st.selectbox('Choices:',snacks_cat)

    #Displaying output
    z=st.slider('How many recommendations do you want to see?',1,10,5)
    st.write("---------------------------------------------------------")
    st.subheader("Recommendations for you:")
    for index,row in df.iterrows():
        if(row[1]==user):
            x=row[2]
            for i,j in df_snacks.iterrows():
                if(x==j[0]):
                    m_cat=j[1]
                    m_name=j[2]
                    m_cal=j[3]
                    m_price=j[4]
                    m_img=j[5]
            if(cat==m_cat):
                st.write('**',m_name,'**')
                st.write('SnackId:',row[2])
                st.write('Calories:',m_cal)
                st.write('Price:',m_price)
                st.image(m_img,use_column_width=None,caption=m_name)
                n=n+1
                if(n>z):
                    break


elif select =='Graphs':
    import altair as alt
    basic_chart = alt.Chart(avg_df).mark_line().encode(
        x='Names',
        y='Prediction'
        ).properties(
        title='Average Rating Prediction per Snack',
        width=800,
        height=500)
    st.altair_chart(basic_chart)

    c = alt.Chart(num_of_users).mark_circle().encode(
        x='Snacks', y='Number of Reviews').properties(
        title='Number of Reviews per Snack',
        width=800,
        height=500)
    st.altair_chart(c)

elif select =='Recommendations':
    st.subheader("Please enter User Name:")
    user=st.text_input('User Name:','A1033RWNZWEMR5')
    z=st.slider('How many recommendations do you want to see?',1,10,5)
    st.write("---------------------------------------------------------")
    st.subheader("Recommendations for you:")
    for index,row in com_df.iterrows():
        if(row[1]==user):
            st.write('**',row[5],'**')
            st.write('SnackId:',row[1])
            st.write('Calories:',row[6])
            st.write('Price:',row[7])
            st.image(row[8],use_column_width=None,caption=row[5])
            n=n+1
            if(n>z):
                break


 
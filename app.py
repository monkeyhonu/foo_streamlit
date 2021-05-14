import streamlit as st 
import plotly.express as px
import numpy as np
import pickle
#import load_data import load_iris as li
import time


#next 2 lines are to use streamlit's convenient functions

# <start of progress bar>
# import time
# with st.spinner(text='In progress'):
#     time.sleep(5)
#     st.success('Done')

# <end of progress bar>
st.color_picker('pick a color')

st.file_uploader('File uploader')

st.title ("My awesome Flower predictor")
st.header("we predict Iris flowers")
st.subheader("No joke")

#load data

#streamlit will run anytime change is made
#might be good to save into cache esp if the 
#function you are running may take awhile
#creating caching function here with decorator @st.cache

df_iris = load_data.load_iris()

st.plotly_chart(px.scatter(df_iris, 'sepal_width', 'sepal_length'))

show_df = st.checkbox("Do you want to see the data?")

if show_df:
    df_iris

#Get user input from flower

s_l = st.number_input('Input the Sepal Length')
s_w = st.number_input('Input the Sepal Width')
p_l = st.number_input('Input the Petal Length')
p_w = st.number_input('Input the Petal Width')

user_values = np.array([s_l, s_w, p_l, p_w])

#load model

with open('saved-iris-model-2.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(user_values.reshape(1, -1))
prediction


# type(prediction)

# t = st.write(prediction)
# t
#st.write(type(prediction))

st.header(f'The model predicts: {prediction[0]}')

#st.balloons()



#using columns to make your page look nicer
col1, col2, col3 = st.beta_columns(3)
with col1:
    'I am printing things'
with col2:
    df_iris
with col3:
    st.subheader("cool stuff")
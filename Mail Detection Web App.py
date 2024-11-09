import numpy
import pandas
import streamlit as st
import pickle
import sklearn

# Now load the saved model
loaded_model=pickle.load(open('C:/Users/HP/OneDrive/Documents/ML Deployment/trained_model_for_email_use_case.sav','rb'))

# Now load the saved file of vectorizer
loaded_vectorizer=pickle.load(open('C:/Users/HP/OneDrive/Documents/ML Deployment/mail_detection_vectorizer.sav','rb'))

# create a function for prediction

def Email_Prediction(text):
    # convert this textual data into the respective feature vectors(i.e the Numerical Representations)
    vectorized_data=loaded_vectorizer.transform(text)
    prediction=loaded_model.predict(vectorized_data)

    if (prediction==1):
        return 'Spam Mail'
    else:
        return 'Ham Mail'
        

def main():
    # Now here, the streamlit is used to make app and user interfaces 
    # use streamlit to give title to our web page
    st.title('Email Prediction Web App')
    # Now get input from user
    Mail_Data=st.text_input("Enter the Mail Here")
    
    # code for prediction
    Type=''
    # Now create a button for prediction
    if (st.button('Check Type')):
        if(Mail_Data!=''):
            Type=Email_Prediction([Mail_Data])
        else:
            st.markdown(''':red[Please Enter The Mail Message]''')
    
    
            
            
            

        
    st.success(Type)
    

if __name__=='__main__':
    main()
    
    
from django.shortcuts import render
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
from keras import layers 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import Sequential
import cv2
import pickle
from LandingPage.comments import comment_master
#from LandingPage.infer import main

def url_extractor(url):
    
    features =[]
    features.append([len(url)])
    if "https" in url:
        i= 8
    else:
        i=7
    hostname=""
    while url[i] != '/':
        hostname+=url[i]
        i+=1
        

    features.append([len(hostname)])
    features.append([url.count('.')])
    features.append([url.count('-')])
    features.append([url.count('@')])
    features.append([url.count('?')])
    features.append([url.count('&')])
    features.append([url.count('=')])
    features.append([url.count('_')])   
    features.append([url.count('~')])
    features.append([url.count('%')])
    features.append([url.count('/')])
    features.append([url.count('*')])
    features.append([url.count(':')])
    features.append([url.count(',')])
    features.append([url.count(';')])
    features.append([url.count('$')])
    features.append([url.count(' ')])
    features.append([url.count('www')])
    features.append([url.count('com')])
    features.append([url.count('http') - url.count('https')])
    features.append([url.count('https')])
    digits = 0
    for s in url:
    
        # if character is digit (return True)
        if s.isnumeric():
            digits += 1
    dighost =0
    for s in hostname:
    
        # if character is digit (return True)
        if s.isnumeric():
            dighost += 1
    features.append([digits/len(url)])
    features.append([dighost/len(hostname)])
    host = hostname.split('.')


    host.sort(key =len)
    features.append([len((host[0]))])


    features.append([len(host[-1])])
    return features

def phishing_checker(url):
    features = url_extractor(url)
    df = pd.read_csv('LandingPage/dataset_phishing.csv')


    le = preprocessing.LabelEncoder()
    df['labels'] = le.fit_transform(df['status'])

    Y1 = df['labels']
    # X1 = df.drop(columns = ['url','labels','status'])
    X1 =df.drop(columns = ['url','nb_dslash','nb_or','labels','status','phish_hints','domain_in_brand','brand_in_subdomain','brand_in_path','suspecious_tld','statistical_report','nb_hyperlinks','ratio_intHyperlinks','ratio_extHyperlinks','ratio_nullHyperlinks','nb_extCSS','ratio_intRedirection','ratio_extRedirection','ratio_intErrors','ratio_extErrors','login_form','external_favicon','links_in_tags','submit_email','ratio_intMedia','ratio_extMedia','sfh','iframe','popup_window','safe_anchor','onmouseover','right_clic','empty_title','domain_in_title','domain_with_copyright','whois_registered_domain','domain_registration_length','domain_age','web_traffic','dns_record','google_index','page_rank', 'punycode', 'ip','port', 'tld_in_path', 'tld_in_subdomain' , 'abnormal_subdomain', 'nb_subdomains',
    'prefix_suffix','random_domain','shortening_service','path_extension','nb_redirection','nb_external_redirection','length_words_raw','char_repeat','shortest_word_path','longest_word_path','avg_words_raw','avg_word_host','avg_word_path','shortest_words_raw','longest_words_raw'])
    X2 = df['url']



    X_train1,X_test1,Y_train1,Y_test1 = train_test_split(X1,Y1,stratify = Y1,test_size = 0.2,random_state = 42)
    X_train1 = np.expand_dims(X_train1, axis=-1)
    X_test1 = np.expand_dims(X_test1, axis=-1)

    def eval_graph(results):

        acc = results.history['accuracy']
        val_acc = results.history['val_accuracy']
        epochs = range(len(acc))
        fig = plt.figure(figsize=(14,7))
        plt.plot(epochs,acc,'r',label="Training Accuracy")
        plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
        plt.legend(loc='upper left')
        plt.title("ACCURACY GRAPH")
        plt.show()
        
        loss = results.history['loss']
        val_loss = results.history['val_loss']
        epochs = range(len(loss))
        fig = plt.figure(figsize=(14,7))
        plt.plot(epochs,loss,'r',label="Training loss")
        plt.plot(epochs,val_loss,'b',label="Validation loss")
        plt.legend(loc='upper left')
        plt.title("LOSS GRAPH")
        plt.show()

        def conf_matrix(X_test,Y_test,model):
        
            Y_pred = model.predict(X_test)
            Y_pred = Y_pred>0.5
            cm = confusion_matrix(Y_test,Y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap = plt.cm.YlGn)
            plt.title('CONFUSION MATRIX')
            plt.show()




    def CNN(input_size):

        model = keras.Sequential()
        model.add(layers.Input(input_size))
        model.add(layers.Conv1D(filters = 16,kernel_size = 3,activation = 'relu',padding = 'same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
        model.add(layers.Conv1D(filters = 32,kernel_size = 3,activation = 'relu',padding = 'same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
        model.add(layers.Conv1D(filters = 64,kernel_size = 3,activation = 'relu',padding = 'same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
        model.add(layers.Conv1D(filters = 128,kernel_size = 3,activation = 'relu',padding = 'same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
        model.add(layers.Conv1D(filters = 256,kernel_size = 3,activation = 'relu',padding = 'same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(512,activation = 'relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1,activation = 'sigmoid'))
        
        return model


    input_size1 = X_train1[1].shape


    CNN_model1 = CNN(input_size1)

    # CNN_model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    # callbacks = [tf.keras.callbacks.ModelCheckpoint('CNN_MODEL_ON_FEATURE_EXTRACTED.h5',verbose=1,save_best_only=True),
    #              tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001,patience=30,verbose=1)]
    # CNN_results_1 = CNN_model1.fit(X_train1,Y_train1,validation_split=0.2,batch_size=128,epochs=200,callbacks=callbacks)

    def CNN_LSTM(input_size):
        model = keras.Sequential()
        # model.add(layers.Input(input_size))
        # model.add(layers.Conv1D(filters = 16,kernel_size = 3,activation = 'relu',padding = 'same'))
        # model.add(layers.Dropout(0.2))
        # model.add(layers.BatchNormalization())
        # model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
        model.add(layers.Conv1D(filters = 32,kernel_size = 3,activation = 'relu',padding = 'same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
        model.add(layers.Conv1D(filters = 64,kernel_size = 3,activation = 'relu',padding = 'same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
        model.add(layers.Conv1D(filters = 128,kernel_size = 3,activation = 'relu',padding = 'same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
        model.add(layers.Conv1D(filters = 256,kernel_size = 3,activation = 'relu',padding = 'same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.LSTM(128,return_sequences=True))
        model.add(layers.Dropout(0.3))
        model.add(layers.Flatten())
        # model.add(layers.Dense(128,activation = 'relu'))
        # model.add(layers.Dropout(0.3))
        # model.add(layers.Dense(128,activation = 'relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1,activation = 'sigmoid'))
        
        return model

    CNN_LSTM_model1 = CNN_LSTM(input_size1)

    CNN_LSTM_model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    callbacks = [tf.keras.callbacks.ModelCheckpoint('CNN_LSTM_MODEL_ON_FEATURE_EXTRACTED.h5',verbose=1,save_best_only=True),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001,patience=30,verbose=1)]
    CNN_LSTM_results_1 = CNN_LSTM_model1.fit(X_train1,Y_train1,validation_split=0.2,batch_size=128,epochs=200,callbacks=callbacks)
    CNN_LSTM_TESTS= CNN_LSTM_model1.predict([features])
    return CNN_LSTM_TESTS
    

# function to predict whether image at the given path is real or fake
def predict_image(image_path):
    # load the image and resize it to 32x32
    img = cv2.imread(image_path)
    img = tf.image.resize(img, (32, 32))
    
    # load the model
    file_path = 'LandingPage/model.keras'

    model = keras.models.load_model(file_path)
        #model = pickle.load('./model.pkl')
    # predict the class
    y_prob = model.predict(np.expand_dims(img, 0))
    return 'REAL' if y_prob[0]>0.5 else 'FAKE'

# Create your views here.
def index(request):
    url=""
    malware=""
    troll=""
    image=""

    if request.method == 'POST':
        if request.POST.get('url'):
            url = request.POST.get('url')
            
            url = phishing_checker(url)
            url = url[0][0] *100
            url = f'The predicted probability of the url being legitimate is: {url} %'
        
    if request.method == 'POST':
        if request.POST.get('malware'):  
             malware = request.POST.get('malware')
        #     malware = main(malware)
        #     malware = f'The predicted probability of the file being malware is: {malware} %'

    if request.method == 'POST':
        if request.POST.get('troll'):
            troll = request.POST.get('troll')
            
            lst = comment_master(troll)
            if len(lst) == 2:
                troll = lst[1]
                troll = [ f'https://www.youtube.com/channel/{i}' for i in troll]
            else:
                troll = "No trolls found"
            



    if request.method == 'POST':
        if request.POST.get('image'):
            image = request.POST.get('image')

            y_pred = predict_image(image)
            

            

        

    context ={
        'malware':malware,
        'phishing':url,
        'troll':troll,
        'deepfake':image,
        'botattack':'botattack',
    }
    return render(request,'galileo-design.html',context)


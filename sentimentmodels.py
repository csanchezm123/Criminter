from xml.etree.ElementTree import parse
import re
import emoji
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pickle

#Importamos el archivo de la base de datos del análisis de sentimientos normal
dataset_normal = parse("dataset_normal.xml")

def data_adquisition_normal(dataset_normal):
    text_normal = []
    sentiment_normal = []
    i = 0
    for item in dataset_normal.iterfind('tweet'):   #Recorremos las secciones 'tweet' del XML y vamos guardando el texto del tweet (en text_normal) y su valor de sentimiento (en sentiment_normal)
        text_normal.append(item.findtext('content'))
        for var in item.iterfind('sentiment'):
            sentiment_normal.append(var.findtext('value'))
        i = i + 1
    return text_normal, sentiment_normal

#Descomentar en el caso que se quiera hacer modelo de análisis de sentimiento normal
text_normal = data_adquisition_normal(dataset_normal)[0]
sentiment_normal = data_adquisition_normal(dataset_normal)[1]

def data_adquisition_hate():
    rows = []
    i = 0
    text_hate = []
    sentiment_hate = []
    with open("dataset_hate.txt","r", encoding='utf-8') as database:
        row = database.readlines()
        for i in range(len(row)):
          rows.append(row[i].split(";||;"))
        for tweet in range(len(rows)):
          text_hate.append(rows[tweet][1])
          sentiment_hate.append(re.sub("\n", "", rows[tweet][2]))
    database.close()
    return text_hate, sentiment_hate

#Descomentar en el caso que se quiera hacer modelo de análisis de sentimiento de odio
text_hate = data_adquisition_hate()[0]
sentiment_hate = data_adquisition_hate()[1]


def cleaning_tweet(tweet):
    tweet_words = []
    tweet = tweet.lower()   #Ponemos el texto en minúsculas
    tweet = tweet.strip()   #Eliminamos los espacios aparentes al inicio y al fin del texto
    tweet = re.sub(r"#", "", tweet)    #Eliminamos el asterisco de los hashtags
    tweet = re.sub(r"@[a-z0-9_/]{1,15}", "", tweet)    #Eliminamos todos los nombres de usuario mencionados
    tweet = re.sub(r"[\n]+", " ", tweet)    #Eliminamos los saltos de línea y los sustituimos por un espacio
    tweet = emoji.replace_emoji(tweet, replace='')     #Eliminamos todos los emoticonos
    tweet = re.sub(r"https?:\/\/(?:www\.)?[-a-z0-9@:%._\+~#=]{1,256}\.[a-z0-9()]{1,6}\b(?:[-a-z0-9()@:%_\+.~#?&\/=]*)", "", tweet)     #Eliminamos las direcciones URLs
    x, y = "áéíóúü", "aeiouu"   #Sustituimos las vocales acentuadas por las mismas sin acentuar
    trans = str.maketrans(x,y)
    tweet = tweet.translate(trans)
    tweet = re.sub(r"[^a-z\s]", " ", tweet)    #Dejamos el texto sin ningún caracter especial (ya sean exclamaciones, puntuaciones, etc.)
    tweet = re.sub(r"[\s]{2,}", " ", tweet)    #Eliminamos los espacios consecutivos y solo incluimos 1
    tweet_words = tweet.split()    #Dividimos el string del tweet en una lista de palabras
    stopwords_list = stopwords.words("spanish")    #Importamos la lista de palabras vacías
    return ' '.join([word for word in tweet_words if word not in stopwords_list])   #Obtenemos un string con todas las palabras menos el


'''   ANÁLISIS DE SENTIMIENTOS NORMAL    ''' #Descomentar para obtención del modelo para el análisis de sentimientos normal
tweet_cleaned_normal = []
long_normal = len(text_normal)
for i in range(long_normal):    #Vamos aplicando el preprocesado a cada uno de los tweets de text_normal
    tweet_cleaned_normal.append(cleaning_tweet(text_normal[i]))
x_normal = tweet_cleaned_normal     #Lista con todos los tweets preprocesados
y_normal = sentiment_normal     #Lista con todos los valores clasificados
x_train_normal, x_test_normal, y_train_normal, y_test_normal = train_test_split(x_normal, y_normal, stratify=y_normal, test_size=0.2, random_state=42)   #Dividimos las listas para tener un 80% para el entrenamiento y un 20% para el proceso de prueba
# vec_x_normal = TfidfVectorizer()   #Para utilizar los métodos de ML, necesitamos vectorizar los elementos
# vec_x_normal.fit_transform(x_train_normal)
# x_train_normal = vec_x_normal.fit_transform(x_train_normal).toarray()
# x_test_normal = vec_x_normal.transform(x_test_normal).toarray()
# print("Entrenamiento:\n", pd.value_counts(y_train_normal), "\nPrueba:\n", pd.value_counts(y_test_normal))     #Imprimimos la cantidad de elementos de cada clase que se han utilizado en cada fase



#Clasificadores probabilísticos Naïve Bayes

#Multinomial
# model_multi = MultinomialNB()
# model_multi.fit(x_train_normal, y_train_normal)     #Obtenemos el modelo para esta técnica
# y_predicted_multi = model_multi.predict(x_test_normal)  #Obtenemos sus predicciones
# print("\n\n\nNAÏVE BAYES: MULTINOMIAL")
# print("Accuracy: ", metrics.accuracy_score(y_test_normal, y_predicted_multi))   #Obtenemos sus métricas
# print(confusion_matrix(y_test_normal, y_predicted_multi))
# print(classification_report(y_test_normal, y_predicted_multi))

#
# #Gaussiano
# model_gauss = GaussianNB()
# model_gauss.fit(x_train_normal, y_train_normal)
# y_predicted_gauss = model_gauss.predict(x_test_normal)
# print("\n\n\nNAÏVE BAYES: GAUSSIANO")
# print("Accuracy: ", metrics.accuracy_score(y_test_normal, y_predicted_gauss))
# print(confusion_matrix(y_test_normal, y_predicted_gauss))
# print(classification_report(y_test_normal, y_predicted_gauss))
#
#
# #Clasificadores basados en Árboles
#
# #Bosques Aleatorios
# model_random = RandomForestClassifier()
# model_random.fit(x_train_normal, y_train_normal)
# y_predicted_random = model_random.predict(x_test_normal)
# print("\n\n\nBASADOS EN ÁRBOLES: BOSQUES ALEATORIOS")
# print("Accuracy: ", metrics.accuracy_score(y_test_normal, y_predicted_random))
# print(confusion_matrix(y_test_normal, y_predicted_random))
# print(classification_report(y_test_normal, y_predicted_random))
#
# #Árboles de Decisión
# model_tree = DecisionTreeClassifier()
# model_tree.fit(x_train_normal, y_train_normal)
# y_predicted_tree = model_tree.predict(x_test_normal)
# print("\n\n\nBASADOS EN ÁRBOLES: ÁRBOLES DE DECISIÓN")
# print("Accuracy: ", metrics.accuracy_score(y_test_normal, y_predicted_tree))
# print(confusion_matrix(y_test_normal, y_predicted_tree))
# print(classification_report(y_test_normal, y_predicted_tree))
#
#
# #Clasificadores lineales de Máquinas Vector Soporte (SVM)
#
# #Kernel: Lineal
# model_linear = SVC(kernel='linear')
# model_linear.fit(x_train_normal, y_train_normal)
# y_predicted_linear = model_linear.predict(x_test_normal)
# print("\n\n\nSVM: KERNEL LINEAL")
# print("Accuracy: ", metrics.accuracy_score(y_test_normal, y_predicted_linear))
# print(confusion_matrix(y_test_normal, y_predicted_linear))
# print(classification_report(y_test_normal, y_predicted_linear))
#
# #Kernel: RBF
# model_rbf = SVC(kernel='rbf')
# model_rbf.fit(x_train_normal, y_train_normal)
# y_predicted_rbf = model_rbf.predict(x_test_normal)
# print("\n\n\nSVM: KERNEL RBF")
# print("Accuracy: ", metrics.accuracy_score(y_test_normal, y_predicted_rbf))
# print(confusion_matrix(y_test_normal, y_predicted_rbf))
# print(classification_report(y_test_normal, y_predicted_rbf))
#
#
#
# #Guardamos modelo escogido (RBF en este caso)
# with open('normal_model.pickle', "wb") as file:
#     pickle.dump(model_rbf, file)
# print("Modelo guardado")


'''   ANÁLISIS DE SENTIMIENTOS DE ODIO    ''' #Descomentar para obtención del modelo para el análisis de sentimientos de odio
tweet_cleaned_hate = []
long_hate = len(text_hate)
for i in range(long_hate):    #Vamos aplicando el preprocesado a cada uno de los tweets de text_hate
    tweet_cleaned_hate.append(cleaning_tweet(text_hate[i]))
x_hate = tweet_cleaned_hate    #Lista con todos los tweets preprocesados
y_hate = sentiment_hate     #Lista con todos los valores clasificados
x_train_hate, x_test_hate, y_train_hate, y_test_hate = train_test_split(x_hate, y_hate, stratify=y_hate, test_size=0.2, random_state=42)   #Dividimos las listas para tener un 80% para el entrenamiento y un 20% para el proceso de prueba
# vec_x_hate = TfidfVectorizer()   #Para utilizar los métodos de ML, necesitamos vectorizar los elementos
# vec_x_hate.fit_transform(x_train_hate)
# x_train_hate = vec_x_hate.fit_transform(x_train_hate).toarray()
# x_test_hate = vec_x_hate.transform(x_test_hate).toarray()
# print("Entrenamiento:\n", pd.value_counts(y_train_hate), "\nPrueba:\n", pd.value_counts(y_test_hate))     #Imprimimos la cantidad de elementos de cada clase que se han utilizado en cada fase
#
#
#
# #Clasificadores probabilísticos Naïve Bayes
#
# #Multinomial
# model_multi = MultinomialNB()
# model_multi.fit(x_train_hate, y_train_hate)     #Obtenemos el modelo para esta técnica
# y_predicted_multi = model_multi.predict(x_test_hate)  #Obtenemos sus predicciones
# print("\n\n\nNAÏVE BAYES: MULTINOMIAL")
# print("Accuracy: ", metrics.accuracy_score(y_test_hate, y_predicted_multi))   #Obtenemos sus métricas
# print(confusion_matrix(y_test_hate, y_predicted_multi))
# print(classification_report(y_test_hate, y_predicted_multi))
#
# #Gaussiano
# model_gauss = GaussianNB()
# model_gauss.fit(x_train_hate, y_train_hate)
# y_predicted_gauss = model_gauss.predict(x_test_hate)
# print("\n\n\nNAÏVE BAYES: GAUSSIANO")
# print("Accuracy: ", metrics.accuracy_score(y_test_hate, y_predicted_gauss))
# print(confusion_matrix(y_test_hate, y_predicted_gauss))
# print(classification_report(y_test_hate, y_predicted_gauss))
#
#
# #Clasificadores basados en Árboles
#
# #Bosques Aleatorios
# model_random = RandomForestClassifier()
# model_random.fit(x_train_hate, y_train_hate)
# y_predicted_random = model_random.predict(x_test_hate)
# print("\n\n\nBASADOS EN ÁRBOLES: BOSQUES ALEATORIOS")
# print("Accuracy: ", metrics.accuracy_score(y_test_hate, y_predicted_random))
# print(confusion_matrix(y_test_hate, y_predicted_random))
# print(classification_report(y_test_hate, y_predicted_random))
#
# #Árboles de Decisión
# model_tree = DecisionTreeClassifier()
# model_tree.fit(x_train_hate, y_train_hate)
# y_predicted_tree = model_tree.predict(x_test_hate)
# print("\n\n\nBASADOS EN ÁRBOLES: ÁRBOLES DE DECISIÓN")
# print("Accuracy: ", metrics.accuracy_score(y_test_hate, y_predicted_tree))
# print(confusion_matrix(y_test_hate, y_predicted_tree))
# print(classification_report(y_test_hate, y_predicted_tree))
#
#
# #Clasificadores lineales de Máquinas Vector Soporte (SVM)
#
# #Kernel: RBF
# model_rbf = SVC(kernel='rbf')
# model_rbf.fit(x_train_hate, y_train_hate)
# y_predicted_rbf = model_rbf.predict(x_test_hate)
# print("\n\n\nSVM: KERNEL RBF")
# print("Accuracy: ", metrics.accuracy_score(y_test_hate, y_predicted_rbf))
# print(confusion_matrix(y_test_hate, y_predicted_rbf))
# print(classification_report(y_test_hate, y_predicted_rbf))
#
#
#
# #Guardamos modelo escogido (Bosques Aleatorios en este caso)
# with open('hate_model.pickle', "wb") as file:
#     pickle.dump(model_random, file)
# print("Modelo guardado")
#
#
#

#Importamos las librerías necesarias
import tweepy
from collections import Counter
import plotly.express as px
import plotly.utils
import json

from sklearn.feature_extraction.text import TfidfVectorizer

from sentimentmodels import cleaning_tweet, x_train_normal, x_train_hate
import pickle

#Establecemos las credenciales de nuestra API de Twitter
bearer_token = ""
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

#Establecemos la autenticación con dichas credenciales
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

apiv1 = tweepy.API(auth, wait_on_rate_limit=True)   #Nos autenticamos para utilizar API de la versión 1.1 de Twitter API

apiv2 = tweepy.Client(bearer_token, consumer_key, consumer_secret, access_token, access_token_secret,
                    wait_on_rate_limit=True)    ##Nos autenticamos para utilizar API de la versión 2 de Twitter API

#Establecemos la paleta de colores que vamos a utilizar para las gráficas
color_discrete_sequence = ["#FFB64D", "#2ED8B6", "#4099FF", "#FF5370", "#6F42C1", "#99FA18", "#E83E8C", "#435EE3", "#FD7E14"]

def data_collection(username):
    #Obtenemos información del usario indicando qué campos obtener utilizando la v2 de la API de Twitter
    user_lookup = apiv2.get_user(username=username, user_fields=["created_at","description","entities","id","location","name","profile_image_url","public_metrics","url","verified","protected"])
    profile_image = user_lookup.data.profile_image_url
    profile_image_resized = profile_image.replace("normal","400x400")   #Redimensionamos la imagen de perfil a 400x400 para que se vea con claridad
    user_lookup.data.created_at = cambiar_formato_fecha(user_lookup.data.created_at)    #Ponemos la fecha de una forma más amigable
    if user_lookup.data.description==None:
        user_lookup.data.description = ""   #En el caso de que no haya descripción
    if user_lookup.data.location==None:
        user_lookup.data.location = "No hay datos de geolocalización"   #En el caso de que no haya datos de geolocalización en los datos de su perfil
    #Indicamos si está verificado
    if user_lookup.data.verified==True:
        user_lookup.data.verified = ["Sí ", 9989]
    else:
        user_lookup.data.verified = ["No ", 10060]
    # En el caso de que no haya ninguna URL especificada
    if user_lookup.data.entities==None:
        user_lookup.data.url = "No hay ningún enlace"
    else:
        try:
            user_lookup.data.url = user_lookup.data.entities['url']['urls'][0]['display_url']
        except KeyError:
            user_lookup.data.url = "No hay ningún enlace"
    id = user_lookup.data.id    #Obtenemos el ID de ese usuario
    banner_image = apiv1.get_user(user_id=id).profile_banner_url    #Obtenemos su foto de fondo
    #De cada uno de los seguidores (con get_users_followers) obtenemos su nombre de usuario y foto de perfil, además de su nombre que ya nos lo da sin necesidad de especificarlo
    followers = apiv2.get_users_followers(id, max_results=1000, user_fields=["username","profile_image_url"])
    if followers.data == None:
        followers_n = 0     #En el caso de que no haya seguidores, determinamos su número a 0
    else:
        followers_n = len(followers.data)
    # De cada uno de los seguidos (con get_users_following) obtenemos su nombre de usuario y foto de perfil, además de su nombre que ya nos lo da sin necesidad de especificarlo
    following = apiv2.get_users_following(id, max_results=1000, user_fields=["username","profile_image_url"])
    if following.data == None:
        following_n = 0     #En el caso de que no haya seguidos, determinamos su número a 0
    else:
        following_n = len(following.data)
    #Obtenemos todos los tweets del usuario (inclusive sus respuestas, pero no retweets)
    tweets = apiv2.get_users_tweets(id, max_results=100, exclude="retweets", tweet_fields=["created_at","geo","id","lang","public_metrics","source","text"])
    if tweets.data == None:
        tweets_n = 0    #En el caso de que no haya tweets, determinamos su número a 0
    else:
        tweets_n = len(tweets.data)
    return user_lookup, profile_image_resized, banner_image, followers, followers_n, following, following_n, tweets, tweets_n

def cambiar_formato_fecha(date):
    meses = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
    #Obtenemos cada parámetro de la fecha
    date_day = date.day
    date_month = meses[date.month]
    date_year = date.year
    date_hour = date.strftime('%H:%M:%S %Z')
    datev1 = "{} de {} del {} ".format(date_day, date_month, date_year)
    datev1 = datev1 + date_hour     #Reformulamos la apariencia de dicha fecha
    return datev1

def stats_week(data, long):
    list_weekday = []
    list_x = []
    list_y = []
    weekdays = [["Lunes", 0], ["Martes", 0], ["Miércoles", 0], ["Jueves", 0], ["Viernes", 0], ["Sábado", 0], ["Domingo", 0]]
    #Obtenemos todos los días de la semana en la que se han publicado dichos tweets
    for Tweet in range(long):
        weekday = data[7].data[Tweet].created_at.weekday()
        list_weekday.append(weekday)
    #Contamos la frecuencia de cada día
    for i in range(len(list_weekday)):
        for j in range(len(weekdays)):
            if j == list_weekday[i]:
                weekdays[j][1] = weekdays[j][1]+1
    #Determinamos los valores de los ejes de la gráfica
    for k in range(len(weekdays)):
        list_x.append(weekdays[k][0])
        list_y.append(weekdays[k][1])
    #Obtenemos un histograma con los días de la semana en el eje X y la frecuencia de tweets en el eje Y
    fig_weekday = px.histogram(weekdays, x=list_x, y=list_y, range_x=[-1, 7], color=list_x, color_discrete_sequence=color_discrete_sequence, opacity=0.7)
    #Configuramos la apariencia de la gráfica
    fig_weekday.update_layout(
        showlegend=False,
        width=770,
        height=270,
        plot_bgcolor="rgba(229,236,246,70)",
        font_family="Roboto Medium",
        font_color="#364f6b",
        font_size=13,
        xaxis_title="Día de la semana",
        yaxis_title="Nº de tweets publicados",
        margin=dict(pad=10, b=0, l=0, t=20, r=0))
    fig_weekday.update_yaxes(showline=True, linecolor="rgb(229,236,246)")
    #Pasamos la gráfica a JSON para poder reproducirla en nuestra página web
    fig_weekday_html = json.dumps(fig_weekday, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_weekday_html

def stats_hour(data, long):
    list_hour = []
    list_x = []
    list_y = []
    hours = [["00h", 0], ["01h", 0], ["02h", 0], ["03h", 0], ["04h", 0], ["05h", 0], ["06h", 0], ["07h", 0], ["08h", 0], ["09h", 0], ["10h", 0], ["11h", 0], ["12h", 0], ["13h", 0], ["14h", 0], ["15h", 0], ["16h", 0], ["17h", 0], ["18h", 0], ["19h", 0], ["20h", 0], ["21h", 0], ["22h", 0], ["23h", 0]]
    #Obtenemos las horas en las que se han publicado los tweets
    for Tweet in range(long):
        hour = data[7].data[Tweet].created_at.hour
        list_hour.append(hour)
    #Contamos la frecuencia de cada hora
    for i in range(len(list_hour)):
        for j in range(len(hours)):
            if j == list_hour[i]:
                hours[j][1] = hours[j][1] + 1
    #Determinamos los valores de eje X e Y
    for k in range(len(hours)):
        list_x.append(hours[k][0])
        list_y.append(hours[k][1])
    # Obtenemos un histograma con las horas en el eje X y la frecuencia de tweets en el eje Y
    fig_hour = px.histogram(hours, x=list_x, y=list_y, range_x=[-1, 24], color=list_x, color_discrete_sequence=color_discrete_sequence, opacity=0.7)
    #Configuramos la apariencia de la gráfica
    fig_hour.update_layout(
        showlegend=False,
        width=770,
        height=270,
        plot_bgcolor="rgba(229,236,246,70)",
        font_family="Roboto Medium",
        font_color="#364f6b",
        font_size=13,
        xaxis_title="Hora del día (UTC+00:00)",
        yaxis_title="Nº de tweets publicados",
        margin=dict(pad=10, b=0, l=0, t=20, r=0))
    fig_hour.update_yaxes(showline=True, linecolor="rgb(229,236,246)")
    # Pasamos la gráfica a JSON para poder reproducirla en nuestra página web
    fig_hour_html = json.dumps(fig_hour, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_hour_html

def stats_pie_collection(data, long):
    list_items_source = []
    list_items_lang = []
    #Obtenemos las fuentes en las que se ha publicado cada tweet y se almacenan en una lista
    for Tweet in range(long):
        source = data[7].data[Tweet].source
        list_items_source.append(source)
    #Contamos las veces que aparece cada fuente
    count_items_source = Counter(list_items_source)
    # Obtenemos los idiomas en los que se han escrito cada tweet y se almacenan en una lista
    for Tweet in range(long):
        lang = data[7].data[Tweet].lang
        list_items_lang.append(lang)
    #Contamos las veces que hay de cada lenguaje
    count_items_lang = Counter(list_items_lang)
    return count_items_source, count_items_lang

def stats_piechart(count_items):
    #Realizamos una gráfica con el diccionario con los valores contados de cada clase
    fig_pie = px.pie(values=count_items.values(), names=count_items.keys(), hole=0.4, color=count_items.keys(), color_discrete_sequence=color_discrete_sequence, opacity=0.8)
    #Configuramos la apariencia de la gráfica
    fig_pie.update_layout(
        autosize=False,
        width=440,
        height=220,
        font_family="Roboto Medium",
        font_color="#364f6b",
        font_size=15,
        margin=dict(pad=0, b=0, l=0, t=0, r=0))
    fig_pie.update_traces(textinfo='percent', textposition='inside', hoverinfo='label+percent+name',
                          hovertemplate="%{label}<br>Nº tweets: %{value}</br>Porcentaje: %{percent}")
    #Pasamos la gráfica a JSON para poder reproducirla en nuestra página web
    fig_pie_html = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_pie_html

def obtener_marcadores(data, long):
    marcadores = []
    otros = 0
    i = 0
    #Obtenemos las coordenadas de los tweets que tengan datos de geolocalización
    for Tweet in range(long):
        if data[7].data[Tweet].geo !=None:
            marcadores.append([data[7].data[Tweet].geo['coordinates']['coordinates'][1], data[7].data[Tweet].geo['coordinates']['coordinates'][0]])
            i = i + 1
        else:
            otros = otros + 1   #Se cuentan los tweets no geolocalizados
    return marcadores, otros

def cleaned_tweets(data, long):
    cleaned_list = []
    #Limpiamos cada uno de los tweets como se ha hecho al entrenar los modelos de Machine Learning
    for Tweet in range(long):
        tweet = data[7].data[Tweet].text
        cleaned_list.append(cleaning_tweet(tweet))
    return cleaned_list

def sentiment_analysis(data, long):
    tweets = cleaned_tweets(data,long)
    sentiment_normal = []
    sentiment_hate = []
    with open('normal_model.pickle', "rb") as file:
        #Importamos el modelo guardado
        normal_model = pickle.load(file)
        #Vectorizamos los tweets preprocesados
        vec_tweet_normal = TfidfVectorizer()
        vec_tweet_normal.fit_transform(x_train_normal)
        normal_vec = vec_tweet_normal.transform(tweets).toarray()
        #Obtenemos un array con los valores de análisis de sentimiento normal
        sentiment_normal = normal_model.predict(normal_vec)
    file.close()
    with open('hate_model.pickle', "rb") as fileh:
        # Importamos el modelo guardado
        hate_model = pickle.load(fileh)
        # Vectorizamos los tweets preprocesados
        vec_tweet_hate = TfidfVectorizer()
        vec_tweet_hate.fit_transform(x_train_hate)
        hate_vec = vec_tweet_hate.transform(tweets).toarray()
        # Obtenemos un array con los valores de análisis de sentimiento de odio
        sentiment_hate = hate_model.predict(hate_vec)
    fileh.close()
    return sentiment_normal, sentiment_hate


import tweepy

#Establecemos las credenciales de la API de Twitter
bearer_token = ""
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

##Establecemos la autenticación con dichas credenciales para utilizar la versión 2 de la API de Twitter
apiv2 = tweepy.Client(bearer_token, consumer_key, consumer_secret, access_token, access_token_secret,
                    wait_on_rate_limit=True)

#Establecemos el string de consulta (Descomentar dependiendo de las palabras que se quieran buscar)
# query = "(amariconada OR amariconadas OR amariconado OR amariconados OR bollera OR bolleras OR catalufo OR catalufos OR furcia OR furcias OR maricon OR maricona OR mariconeo OR maricones OR minusvalido OR minusvalida OR minusvalidos OR minusvalidas OR minusválido OR minusválida OR minusválidos OR minusválidas OR mongolo OR moromierda OR moro de mierda OR moromierdas OR negro de mierda OR perra OR perroflauta OR retrasado mental OR sudaca OR sudacas OR travelo OR travelos OR zorra) lang:es -is:retweet"
# query = "(puta OR puto OR hijo de puta) lang:es -is:retweet"
def tweets_collection(query, until_id):
    tweets = apiv2.search_recent_tweets(query=query, tweet_fields=['id','text'], max_results=100, until_id=until_id)    #Obtenemos los ids y contenido de los 100 tweets más recientes siguiendo la consulta y desde el id seleccionado
    long = long = len(tweets.data)
    rows = []
    with open("database.txt","a", encoding='utf-8') as database:    #En un archivo de texto introducimos toda la información separada con ';||;'
        for Tweet in range(long):
            tweet = str(tweets.data[Tweet].id)+";||;"+tweets.data[Tweet].text+";||;"+";||;\n"
            database.write(tweet)
        database.close()

#Vamos ejecutando la siguiente línea indicando el string de consulta y el id desde el que se quiere consultar. Se ha ido ejecutando múltiples veces
# tweets_collection(query, 1528875768247472130)

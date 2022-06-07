#Importamos las librerías y funciones que necesitamos
from flask import Flask, render_template, request, redirect, url_for
from twitter import data_collection, stats_week, stats_hour, stats_pie_collection, stats_piechart, obtener_marcadores, sentiment_analysis


#Iniciamos la aplicación Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")    #HTML del buscador

@app.route('/search', methods=['GET', 'POST'])
def search():
    #Obtiene el nombre de usuario introducido en el buscador y te redirige a la función "user" a la que damos como parámetro el nombre de usuario
    username = request.form.get("username")
    return redirect((url_for('user', username=username)))

@app.route('/search/<username>')
def user(username):
    username = str(username)    #Pasamos el nombre de usuario obtenido a string, por si acaso
    data = data_collection(username)
    long = data[8]  #Para la ejecución de las siguientes funciones se necesita saber el número de tweets recogidos
    #Obtenemos los gráficos con las distintas funciones creadas en el archivo twitter.py
    fig_weekday_html = stats_week(data, long)
    fig_hour_html = stats_hour(data, long)
    fig_source_html = stats_piechart(stats_pie_collection(data, long)[0])
    fig_lang_html = stats_piechart(stats_pie_collection(data, long)[1])
    marcadores = obtener_marcadores(data, long)[0]
    marcadores_n = data[8] - obtener_marcadores(data, long)[1]
    #Obtenemos la clasificación de los tweets en el análisis de sentimientos normal y de odio
    if long != 0:
        sentiment_normal = sentiment_analysis(data, long)[0]
        sentiment_hate = sentiment_analysis(data, long)[1]
    if data[0].data.protected == False:     #Si el usuario no está protegido, renderiza la plantilla con toda la información
        return render_template("userpublic.html", username=username, data=data, fig_weekday_html=fig_weekday_html,
                               fig_hour_html=fig_hour_html, fig_source_html=fig_source_html, fig_lang_html=fig_lang_html,
                               marcadores=marcadores, marcadores_n=marcadores_n, sentiment_normal=sentiment_normal,
                               sentiment_hate=sentiment_hate)
    else:   #Si el usuario es privado, nos dará información restrictiva del usuario (solo información y estadísticas básicas)
        return render_template("userprotected.html", username=username, data=data)

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")  #En el caso de que ocurra error 404

@app.errorhandler(Exception)
def not_found(e):
    return render_template("500.html")  #En el caso de que ocurra error 500


#Solo se utilizan las siguientes líneas en desarrollo, son para que se actualice automáticamente la aplicación web
if __name__ == '__main__':
    app.run()

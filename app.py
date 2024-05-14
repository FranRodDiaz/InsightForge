import os.path
import shutil
from celery import Celery
from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from flask_babel import Babel, gettext  # Para internacionalizar la aplicación
from flask_mail import Mail, Message
from modelo import model as m
from io import BytesIO
import uuid
import pandas as pd
import requests
from flask_socketio import join_room, SocketIO
from datetime import datetime
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
import traceback
import ast
import re
# celery -A app.celery worker --max-tasks-per-child=1 --loglevel=info  Comando para los workers

LANGUAGES = {
    'en': 'English',
    'es': 'Spanish'
}


def get_locale():
    user_language = session.get('user_language')
    if user_language:
        return user_language
    selected_language = request.accept_languages.best_match(app.config['LANGUAGES'].keys())
    return selected_language if selected_language else 'es'


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
app.secret_key = 'tfgFrancescRodriguezDiaz2324'
app.config['LANGUAGES'] = LANGUAGES
app.config['MAIL_SERVER'] = "smtp.gmail.com"
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = "tfgfrancescrodriguezdiaz@gmail.com"
app.config['MAIL_PASSWORD'] = "igfl ptwb ytot yllm"
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
babel = Babel(app, locale_selector=get_locale)
mail = Mail(app)
s = URLSafeTimedSerializer('tfgFrancescRodriguezDiaz2324')
socketio = SocketIO(app)


@app.route('/')
def home():
    user = session.get('usuario')

    if user is None:
        return render_template('index.html', LANGUAGES=LANGUAGES, current_language=get_locale())
    else:
        datasets = m.fetchDatasetsName(user)
        from_duplicate = session.get('from_duplicate')

        if from_duplicate:
            project_id = session.get('project_id')

            session.pop('from_duplicate', None)

            project = m.getProject(project_id)

            dataset = m.getFile(project_id)

            model = m.getModel(project_id)

            dataProject = [project[i] if project[i] is not None else "" for i in [1, 4, 6, 7, 8, 9, 10]]

            column_types, valores = m.processForm(BytesIO(dataset[0]), dataProject[1])

            fields_disabled = False

            newProjectId = m.createProject(BytesIO(dataset[0]), user, dataset[1])
            # Generación de un id único para el id del proyecto
            action_id = str(uuid.uuid4())

            session[action_id] = {'idProyecto': newProjectId}
            # Actualización de datos del proyecto creado con los datos del proyecto que estamos duplicando
            m.updateData(dataProject[2], newProjectId, "tipoProblema", user)

            m.updateData(dataProject[1], newProjectId, "clase", user)

            m.updateData(model[0], newProjectId, "modelo", user)

            m.updateData(dataProject[4], newProjectId, "externo", user)

            m.updateData(dataProject[3], newProjectId, "control", user)

            m.updateData(dataProject[5], newProjectId, "validacion", user)

            m.updateData(dataProject[6], newProjectId, "clasePositiva", user)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            m.updateData(str(dataProject[0]) + "_copy" + timestamp, newProjectId, "nombreProyecto", user)

            return render_template('crearProyecto.html', loged=True,
                                   control_var=dataProject[1], problem_type=dataProject[2],
                                   checkbox_vars=dataProject[3].split(','), extern_vars=dataProject[4].split(','),
                                   validation=dataProject[5].split("-"),
                                   file_name=dataset[1], model=model[0], column_types=column_types,
                                   fields_disabled=fields_disabled, duplicate=True, action_id=action_id,
                                   datasets=datasets, positiveClass=dataProject[6],
                                   nombreProyecto=str(dataProject[0]) + "_copy" + timestamp, valores=valores.tolist(), LANGUAGES=LANGUAGES, current_language=get_locale())

        return render_template('index.html', loged=True, LANGUAGES=LANGUAGES, current_language=get_locale())

@app.route('/goIndex')
def goIndex():
    user = session.get('usuario')

    if user is None:
        return render_template('index.html', LANGUAGES=LANGUAGES, current_language=get_locale())
    else:
        return render_template('index.html', loged=True, LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route('/goMisProyectos')
def goMisProyectos():
    user = session.get('usuario')

    if user is None:
        return render_template('index.html', LANGUAGES=LANGUAGES, current_language=get_locale())
    else:
        projects = m.getProjects(user)

        return render_template('misProyectos.html', loged=True, history=projects, username=user[1], userId=user[0], LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route('/goCrearProyecto') #CAMBIOOOOO AQUIIIIIII
def goCrearProyecto():
    user = session.get('usuario')

    if user is None:
        return render_template('index.html', LANGUAGES=LANGUAGES, current_language=get_locale())
    else:
        datasets = m.fetchDatasetsName(user)
        return render_template('crearProyecto.html', loged=True, datasets=datasets, duplicate=False, LANGUAGES=LANGUAGES, current_language=get_locale())

@app.route('/goLogin')
def goLogin():
    return render_template('login.html', existUser=False, LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route('/recuperarContrasenia')
def recuperarContrasenia():
    return render_template('recuperarContrasenia.html', LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route('/mandarCorreoRecuperacion', methods=['GET', 'POST'])
def mandarCorreoRecuperacion():
    if request.method == 'POST':
        username = request.form['username'].strip()
        exist = m.checkUser(username)

        if not exist:
            return render_template('recuperarContrasenia.html', existUser=True, LANGUAGES=LANGUAGES, current_language=get_locale())

        if len(username) == 0:
            return render_template('recuperarContrasenia.html', usernameOk=False, LANGUAGES=LANGUAGES, current_language=get_locale())

        email = m.obtainEmail(username)[0]
        print(email)
        token_data = {'email': email, 'username': username}
        token = s.dumps(token_data, salt='recuperacion-email')
        msg = Message('Recuperar Contraseña', sender='tfgrodriguezdiazfrancesc@gmail.com', recipients=[email])
        link = url_for('confirmar_token', token=token, _external=True)
        msg.body = 'Tu enlace para recuperar la contraseña es {}'.format(link)
        mail.send(msg)

    return render_template('index.html', LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route('/confirmar/<token>')
def confirmar_token(token):
    try:
        token_data = s.loads(token, salt='recuperacion-email', max_age=3600)
    except SignatureExpired:
        return '<h1>El enlace ha expirado!</h1>'

    email = token_data['email']
    username = token_data['username']

    return render_template('nuevaContrasenia.html', email=email, username=username, LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route('/establecerNuevaContrasenia', methods=['POST'])
def restablecer():
    email = request.form['email'].strip()
    username = request.form['username'].strip()
    password = request.form['newPassword'].strip()
    repeatPassword = request.form['repeatPassword'].strip()

    if password != repeatPassword:
        return render_template('nuevaContrasenia.html', email=email, username=username, iguales=True, LANGUAGES=LANGUAGES, current_language=get_locale())

    if len(password) == 0 and len(repeatPassword) == 0:
        return render_template('nuevaContrasenia.html', email=email, username=username, newpass=False, repeatpass=False, LANGUAGES=LANGUAGES, current_language=get_locale())

    if len(password) == 0:
        return render_template('nuevaContrasenia.html', email=email, username=username, newpass=False, LANGUAGES=LANGUAGES, current_language=get_locale())

    if len(repeatPassword) == 0:
        return render_template('nuevaContrasenia.html', email=email, username=username, repeatpass=False, LANGUAGES=LANGUAGES, current_language=get_locale())

    m.updatePassword(username, password)

    msg = Message('Confirmación de Cambio de Contraseña', sender='tfgrodriguezdiazfrancesc@gmail.com',
                  recipients=[email])
    msg.body = 'Tu contraseña ha sido actualizada exitosamente.'
    mail.send(msg)

    return render_template('login.html', LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route("/goRegister")
def goRegister():
    return render_template('signup.html', LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route('/set-language/<lang>')
def set_language(lang):
    if lang in LANGUAGES:
        session['user_language'] = lang
    return redirect(request.referrer or url_for('home'))


@app.route("/login", methods=['GET', 'POST'])
def login():
    username = request.form['username'].strip()
    password = request.form['password'].strip()
    usuario = m.login(username, password)

    if usuario is None:
        return render_template('login.html', existUser=True, username=username, LANGUAGES=LANGUAGES, current_language=get_locale())
    else:
        session['usuario'] = usuario
        return redirect(url_for('home'))


@socketio.on('user_connected')
def handle_user_connected():
    user_id = session['usuario']['id']
    print("El usuario con id ", user_id, " va a entrar a sala")
    join_room(user_id)


@app.route('/goLogout')
def goLogout():
    user = session.get("usuario")
    if user is not None:
        session.pop('usuario')
    return redirect(url_for('home'))


@app.route("/register", methods=['GET', 'POST'])
def register():
    username = request.form['username'].strip()
    password = request.form['password'].strip()
    email = request.form['email'].strip()

    if len(username) == 0 and len(password) == 0 and len(email) == 0:
        return render_template('signup.html', usernameOk=False, passwordOk=False, emailOk=False, LANGUAGES=LANGUAGES, current_language=get_locale())

    if len(username) == 0:
        return render_template('signup.html', usernameOk=False, LANGUAGES=LANGUAGES, current_language=get_locale())

    if len(password) == 0:
        return render_template('signup.html', passwordOk=False, LANGUAGES=LANGUAGES, current_language=get_locale())

    if len(email) == 0:
        return render_template('signup.html', emailOk=False, LANGUAGES=LANGUAGES, current_language=get_locale())

    exist = m.checkUser(username)

    if exist:
        return render_template('signup.html', existUser=True)
    else:
        m.registerUser(username, password, email)
        return redirect(url_for('home'))


@app.route('/process_file', methods=['GET', 'POST'])  #CAMBIOSSS AQUÍIIII
def process_file():
    if request.method == "POST":
        file = request.files.get('uploaded_file')

        content_type = file.content_type
        if content_type != 'text/csv':
            return jsonify({"error": "El archivo no es un CSV según el tipo MIME"}), 415

        # Mover el puntero al final del archivo para obtener su tamaño
        file.seek(0, 2)  # Mover el puntero al final
        file_size = file.tell()  # Obtener la posición del puntero, que corresponde al tamaño
        file.seek(0)  # Volver el puntero al inicio para futuras operaciones

        print(file_size)
        if file_size > 25 * 1024 * 1024:
            print("ERROR EN EL TAMAÑO")
            return jsonify({"error": "El archivo excede el tamaño máximo permitido de 25 MB"}), 413

        columns, valores = m.processForm(file)

        user = session.get("usuario")

        action_id = str(uuid.uuid4())

        idProyecto = m.createProject(file, user)

        session[action_id] = {'idProyecto': idProyecto}

        return jsonify({"columns": columns, "actionId": action_id})

    else:
        return render_template('index.html', LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route("/fetchDataset", methods=["GET", "POST"])
def fetchDataset():
    idProyecto = request.form.get("idProyecto")

    user = session.get("usuario")

    dataset = m.fetchDataset(idProyecto)

    columns = m.processForm(BytesIO(dataset[0]))

    print(columns[0])

    idNewProyect = m.createProject(BytesIO(dataset[0]), user, dataset[1])

    action_id = str(uuid.uuid4())

    session[action_id] = {'idProyecto': idNewProyect}

    return jsonify({"columns": columns[0], "actionId": action_id})


@app.route('/updateData', methods=['POST'])
def updateData():
    new_value = request.form.get('value')

    action_id = request.form.get('idAction')

    origen = request.form.get('origen')

    success = True

    user = session.get("usuario")

    if action_id in session:
        idProyecto = session[action_id].get("idProyecto")

        success = m.updateData(new_value, idProyecto, origen, user)

    return jsonify({"success": success})


@app.route('/updateDataList', methods=['POST'])
def updateDataList():
    new_value = request.form.getlist('value')

    action_id = request.form.get('idAction')

    origen = request.form.get('origen')

    success = True

    user = session.get("usuario")

    if action_id in session:
        idProyecto = session[action_id].get("idProyecto")

        success = m.updateData(new_value, idProyecto, origen, user)

    return jsonify({"success": success})


@app.route("/clasePositiva", methods=['POST'])
def fetchPositiveClass():
    action_id = request.form.get('idAction')

    columnaClase = request.form.get("columnaClase")

    idProyecto = session[action_id].get("idProyecto")

    dataset = m.getFile(idProyecto)

    df = pd.read_csv(BytesIO(dataset[0]))

    df = df[columnaClase]

    valores = df.unique()

    print(valores)

    return jsonify({"valores": valores.tolist()})


@app.route("/duplicate", methods=['POST'])
def duplicateProject():
    project_id = request.form.get('project_id_duplicate')

    user = session.get('usuario')

    idProjects = m.getProjectsId(user[0])

    idProjects_flattened = [str(id[0]) for id in idProjects]

    if project_id not in idProjects_flattened:
        projects = m.getProjects(user)

        return render_template('misProyectos.html', loged=True, history=projects, username=user[1], userId=user[0],
                               LANGUAGES=LANGUAGES, current_language=get_locale(), errorIdentificador=True)

    session['from_duplicate'] = True

    session['project_id'] = project_id

    return redirect(url_for('home'))


@app.route('/eliminar_carpeta', methods=['POST'])
def eliminar_carpeta():
    action_id = request.form.get('action_id')
    print(action_id)
    if action_id == "":
        return '', 204

    ruta_carpeta = f"./static/dirs/{action_id}"

    if os.path.exists(ruta_carpeta):
        shutil.rmtree(ruta_carpeta)

    return '', 204


@app.route('/inicioProceso', methods=['GET', 'POST'])
def inicioProceso():
    # Inicializa las variables
    print("ESTAMOSSS DENTROOOO")
    id_project = None
    nombreProyecto = None

    # Comprueba el tipo de petición
    if request.method == 'POST':
        # Si es POST, obtén los datos del cuerpo de la petición
        id_project = request.form.get('idProject')
        nombreProyecto = request.form.get("nombreProyecto")

        idUser = session.get('usuario')[0]

        idProjects = m.getProjectsId(idUser)

        idProjects_flattened = [str(id[0]) for id in idProjects]

        if id_project not in idProjects_flattened:
            return jsonify({'error': 'id_project not found'}), 500
    elif request.method == 'GET':
        # Si es GET, obtén los datos de la sesión
        id_project = session.get('project_id')
        nombreProyecto = m.obtenerNombreProyecto(id_project)

    idUser = session.get('usuario')[0]
    nameUser = session.get('usuario')[1]
    mailUser = session.get('usuario')[3]
    n_cores = os.cpu_count()

    rutaArchivo = f"./static/dirs/{nameUser}/Modelos/{nombreProyecto}"

    rutaImagen = f"./static/dirs/{nameUser}/Images/{nombreProyecto}"

    if not os.path.exists(f"./static/dirs/{nameUser}/Images/{nombreProyecto}"):
        os.mkdir(f"./static/dirs/{nameUser}/Images/{nombreProyecto}")

    if not os.path.exists(f"./static/dirs/{nameUser}/Modelos/{nombreProyecto}"):
        os.mkdir(f"./static/dirs/{nameUser}/Modelos/{nombreProyecto}")

    print("VAMOS AL PROCEOS CELERY")

    result = proceso_independiente_celery.delay(id_project, nombreProyecto, idUser, n_cores, rutaArchivo, nameUser, rutaImagen,
                                          mailUser)

    task_result = result.get()

    print(task_result)
    if request.method == "POST":
        if not task_result:
            print("Entre 1")
            return '', 500
        else:
            print("ENtre 2")
            return '', 204


# Para que esto funcione bien hay que declarar la siguiente variable global: export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
@celery.task()
def proceso_independiente_celery(id_project, nombreProyecto, idUser, n_cores, rutaArchivo, nameUser, rutaImagen,
                                 mailUser):

    try:

        response = requests.get(
            f'http://localhost:3001/activarConsultas?id_project={id_project}&nombreProyecto={nombreProyecto}&idUser={idUser}&nameUser={nameUser}')

        m.actualizarEstadoProyecto("Running", id_project, idUser)

        m.actualizarPorcentajeCargaInicio(1, 25, id_project)

        dataset = m.procesarDataset(id_project)

        print(dataset)

        modelo = m.getModel(id_project)

        proyecto = m.getProject(id_project)

        validacion, valorValidacion = proyecto[9].split("-")

        if validacion == "HO":
            valorValidacion = int(valorValidacion) / 100

        columnaClase = proyecto[4]

        columnasControl = proyecto[7]

        columnasExterior = proyecto[8]

        clasePositiva = proyecto[10]

        tipoProblema = proyecto[6]

        m.actualizarPorcentajeCargaInicio(25, 50, id_project)

        m.model(dataset, validacion, valorValidacion, tipoProblema, columnaClase, modelo[0], id_project, n_cores,
                rutaArchivo, nombreProyecto, nameUser, clasePositiva)

        m.actualizarPorcentajeCargaInicio(50, 75, id_project)

        m.generarGraficas(dataset, columnaClase, columnasControl, columnasExterior, n_cores, nombreProyecto, nameUser,
                          modelo[0], rutaImagen, tipoProblema)


        rutaAnalisisError = f'./static/dirs/{nameUser}/Modelos/{nombreProyecto}/{modelo[0]}-Model-Predictions-{nameUser}-{nombreProyecto}.csv'

        m.generateAnalisisError(rutaAnalisisError, id_project)

        m.actualizarPorcentajeCargaInicio(75, 101, id_project)

        m.actualizarEstadoProyecto("Done", id_project, idUser)

        with app.app_context():
            msg = Message("Análisis finalizado",
                          sender="tfgrodriguezdiazfrancesc@gmail.com",
                          recipients=[mailUser])
            msg.body = 'Se le envia este correo para comunicarle que su proyecto ha terminado de ejecutarse. Puede acceder a "http://127.0.0.1:5000/goMisProyectos" para verlo.'
            mail.send(msg)

        return True
    except Exception as e:
        print(f"Fallo en la ejecución del proyecto con id {id_project}")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Mensaje de error: {e}")

        # Imprime la traza completa del error
        print("Traza del error:")
        traceback.print_tb(e.__traceback__)
        m.actualizarEstadoProyecto("Not started", id_project, idUser)
        m.reinicarPorcentajeCarga(id_project)
        shutil.rmtree(f"./static/dirs/{nameUser}/Modelos/{nombreProyecto}")
        shutil.rmtree(f"./static/dirs/{nameUser}/Images/{nombreProyecto}")
        return False





#Ruta para visualizar un proyecto
@app.route('/viewProject', methods=['POST', 'GET']) #CAMBIO AQUIIIII
def viewProject():
    user = session.get('usuario')

    if user is None:
        return render_template('index.html', LANGUAGES=LANGUAGES, current_language=get_locale())

    if request.method == "POST":
        idProject = request.form.get('project_id_view')  # Se obtiene el id del proyecto
        session['project_id'] = idProject

        idUser = user[0]

        idProjects = m.getProjectsId(idUser)

        idProjects_flattened = [str(id[0]) for id in idProjects]

        if idProject not in idProjects_flattened:
            projects = m.getProjects(user)

            return render_template('misProyectos.html', loged=True, history=projects, username=user[1], userId=user[0],
                                   LANGUAGES=LANGUAGES, current_language=get_locale(), errorIdentificador=True)
    else:
        idProject = session.get('project_id')

    project = m.getProject(idProject) #Obtenemos el proyecto seleccionado

    if project[9].split('-')[0] == "HO": #Construimos la cadena de validacion
        validacion = "Hold-out " + str(project[9].split('-')[1]) + "% train"
    else:
        validacion = "Cross validaction " + str(project[9].split('-')[1]) + " folds"

    user = session.get('usuario') #Obtenemos de la sesión al usuario

    rutaImagen = f'./static/dirs/{user[1]}/Images/{project[1]}' #Construimos la ruta de las imagenes
    rutaFichero = f'./static/dirs/{user[1]}/Modelos/{project[1]}' #Construimos la ruta de los ficheros

    rutaZip = m.crearZip(project[1], user[1], idProject) #Creamos el zip y obtenemos su ruta

    datos = {'nombreProyecto': project[1], #Creamos un diccionario con los datos
             'fecha': project[3],
             'clase': project[4],
             'problema': project[6],
             'clasePositiva': project[10],
             'validacion': validacion,
             'rutaImagenes': rutaImagen,
             'rutaFichero': rutaFichero,
             'rutaZip': rutaZip}

    analisisError = m.obtainAnalisisError(idProject)[0] #Obtenemos el analisis de errores realizados

    analisisError = re.sub(r"\.\n*", ".", analisisError)

    metrics = m.getMetrics(idProject, project[6]) #Obtenemos las metricas

    print(metrics)
    print(type(metrics[0]))
    if project[6] == "clasificacion": #Según el tipo de problema construimos un diccionario u otro
        metricas = {'Accuracy': metrics[0],
                    'Precision': metrics[1],
                    'Sensibilidad': metrics[2],
                    'F1 Score': metrics[3],
                    'Specificity': metrics[4],
                    'NPV': metrics[5]}

    else:
        metricas = {'MAE': metrics[0],
                    'MAPE': metrics[1],
                    'MSE': metrics[2],
                    'RMSE': metrics[3],
                    'R^2': metrics[4]}

    for key, value in metricas.items():
        metricas[key] = round(value, 4)

    mensajeMAPE = True if float(metrics[1]) > 1000 else False #Comprobación para el mensaje del MAPE

    modelo = m.getModel(idProject) #Obtenemos el modelo

    dataset = m.getFile(idProject) #Obtenemos el dataset

    df = pd.read_csv(BytesIO(dataset[0])) #Leemos el archivo

    df = df[project[4]] #Nos quedamos con las columna del target

    if project[6] == "clasificacion":
        valores = "binario" if len(df.unique()) < 3 else "multiclase" #Comprobamos si es binario o multiclase

    tipoPresentacion = 0 #Tipo de presentación

    mensajeSHAP = True

    #Escogemos el tipo de presentación
    if modelo[0] == "AML" and (project[6] == "regresion" or (project[6] == "clasificacion" and valores == "binario")):
        tipoPresentacion = 1
        mensajeSHAP = project[6] not in ["RF", "GBM"]

    elif modelo[0] == "AML" and project[6] == "clasificacion" and valores == "multiclase":
        tipoPresentacion = 2
        mensajeSHAP = False

    elif modelo[0] != "AML" and (project[6] == "regresion" or (project[6] == "clasificacion" and valores == "binario")):
        tipoPresentacion = 3
        mensajeSHAP = project[6] not in ["RF", "GBM"]

    elif modelo[0] != "AML" and project[6] == "clasificacion" and valores == "multiclase":
        tipoPresentacion = 4
        mensajeSHAP = False

    datasetContruido = m.procesarDataset(idProject) #Procesamos el dataset

    X, y = m.splitDataset(datasetContruido) #Lo dividimos

    if project[6] == "regresion": #Creamos el json del modelo Tree
        arbol_json = m.convertirArbolAJSONRegresion(X, y, datasetContruido.columns)
    else:
        clasesTarget = y.unique()

        arbol_json = m.convertirArbolAJSONClasificacion(X, y, datasetContruido.columns, clasesTarget)

    analisisError = m.insert_newlines_after_four_dots(analisisError)

    return render_template('resultados.html', datos=datos, metricas=metricas, tipoPresentacion=tipoPresentacion,
                           tree_data=arbol_json, loged=True, id_project=idProject, mensajeMAPE=mensajeMAPE, analisisError=analisisError, problema=project[6]
                           , LANGUAGES=LANGUAGES, current_language=get_locale(), mensajeSHAP=mensajeSHAP)


#Ruta para obtener la gráficas de un proyecto
@app.route("/obtenerGraficos", methods=['POST'])
def obtenerGraficas():
    ruta = request.form.get('rutaDirectorio') #Obtenemos la ruta

    pdf_files_DP = [f for f in os.listdir(ruta) if 'dependece' in f and f.endswith('.pdf')] #Obtenemos los archivos pdf del DP

    pdf_files_ICE = [f for f in os.listdir(ruta) if 'ICE plot' in f and f.endswith('.pdf')] #Obtenemos los archivos pdf del ICE

    pdf_files_NOMO = [f for f in os.listdir(ruta) if 'Nomogram' in f and f.endswith('.pdf')] #Obtenemos los archivos pdf del NOMO

    result = { #Creamos diccionario
        "dependencePlot": pdf_files_DP,
        "ICEplot": pdf_files_ICE,
        "NomogramPlot": pdf_files_NOMO
    }

    return jsonify(result)


#Método para comprobar los datos antes de iniciar la ejecución de un proyecto
@app.route("/comprobarDatos", methods=['POST'])
def comprobarDatos():
    id_project = request.form.get('idProject') #Obtenemos el id del proyecto

    proyecto = m.getProject(id_project) #Obtenemos el proyecto

    errors = m.checkData(proyecto) #Obtenemos una lista que contiene los errores

    if any(errors): #Si contiene algún error se devuelve error
        return '', 400
    else: #Sino se devuelve que está correcto
        return '', 204


#Método para modificar los datos
@app.route("/ModificarDatos", methods=['GET', 'POST'])
def modificarDatos():
    id_project = request.args.get('projectId') #Obtenemos el id del proyecto

    project = m.getProject(id_project)

    dataset = m.getFile(id_project)

    model = m.getModel(id_project)

    dataProject = [project[i] if project[i] is not None else "" for i in [1, 4, 6, 7, 8, 9, 10]]

    if dataProject[1] == '':
        column_types, valores = m.processForm(BytesIO(dataset[0]))
    else:
        column_types, valores = m.processForm(BytesIO(dataset[0]), dataProject[1])

    fields_disabled = False

    if valores is not None:
        valores = valores.tolist()

    user = session.get('usuario')

    datasets = m.fetchDatasetsName(user)

    action_id = str(uuid.uuid4())

    session[action_id] = {'idProyecto': id_project}

    return render_template('crearProyecto.html', loged=True,
                           control_var=dataProject[1], problem_type=dataProject[2],
                           checkbox_vars=dataProject[3].split(','), extern_vars=dataProject[4].split(','),
                           validation=dataProject[5].split("-"),
                           file_name=dataset[1], model=model[0], column_types=column_types,
                           fields_disabled=fields_disabled, duplicate=True, action_id=action_id,
                           datasets=datasets, positiveClass=dataProject[6],
                           nombreProyecto=str(dataProject[0]), valores=valores, actualizar=True, LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route('/receive_speech', methods=['POST'])
def receive_speech():
    texto = request.form['texto']
    print(texto)

    texto = texto.lower()

    if texto == "eliminar todos los proyectos" or texto == "borrar todos los proyectos":
        return jsonify({"borrar": True})

    if "el proyecto" not in texto:
        # La frase no sigue la estructura esperada
        print({"frase": None})
        return jsonify({"frase": None})

    inicio_nombre_proyecto = texto.find("el proyecto ") + len("el proyecto ")

    # La acción es lo que está antes del el proyecto'
    accion = texto[:inicio_nombre_proyecto - len("el proyecto ")].strip()

    # El nombre del proyecto es lo que está después del 'el proyecto'
    identificador = texto[inicio_nombre_proyecto:].strip()

    diccionario = {'accion': accion, 'identificador': identificador}

    diccionario['identificador'] = m.convertir_numeros_en_nombre(diccionario['identificador'])
    print(diccionario)

    if diccionario['identificador'] is None:
        return jsonify({'frase': None})

    idUser = session.get('usuario')[0]

    resultado = m.comprobarNombreyAccionProyecto(diccionario['identificador'], idUser, diccionario['accion'])

    print(resultado)

    if not isinstance(resultado, dict):
        print("ENTREEE AQUI DE NUEVO")
        session['project_id'] = resultado  # Guardar ID del proyecto en la sesión
        if accion == 'visualizar':
            return jsonify({'redirect': url_for('viewProject')})
        elif accion == 'copiar':
            session['from_duplicate'] = True
            return jsonify({'redirect': url_for('home')})
        elif accion == 'eliminar' or accion == 'borrar':
            return jsonify({'redirect': url_for('deleteProject')})
        elif accion == 'ejecutar':
            proyecto = m.getProject(resultado)  # Obtenemos el proyecto

            errors = m.checkData(proyecto)  # Obtenemos una lista que contiene los errores

            if any(errors):
                return jsonify({"datos": None, "idProject": resultado})
            else:
                return jsonify({'redirect': url_for('inicioProceso')})

    else:
        print(resultado)
        # Si 'resultado' es un diccionario, asumimos que contiene información del error
        error_tipo = 'accion' if 'accion' in resultado else 'identificador'
        print(error_tipo)
        return jsonify({error_tipo: None})


@app.route('/contacto', methods=['GET', 'POST'])
def contacto():
    user = session.get("usuario")
    print(request.method)
    if request.method == 'POST':

        if user is None:
            username = request.form['username'].strip()
            mensaje = request.form['mensaje'].strip()

            exist = m.checkUser(username)

            if exist:
                email = m.obtainEmail(username)

                msg = Message('Mensaje del usuario {0}'.format(username), sender='tfgrodriguezdiazfrancesc@gmail.com',
                              recipients=['frodriguezd28@educarex.es'])

                msg.body = '{0}\n\nEl correo de contacto es {1}'.format(mensaje, email[0])
                mail.send(msg)

                return render_template('index.html', LANGUAGES=LANGUAGES, current_language=get_locale())
            else:
                return render_template('index.html', existUser=True, LANGUAGES=LANGUAGES, current_language=get_locale())
        else:
            mensaje = request.form['mensaje'].strip()

            email = m.obtainEmail(user[1])

            msg = Message('Mensaje del usuario {0}'.format(user[1]), sender='tfgrodriguezdiazfrancesc@gmail.com',
                          recipients=['frodriguezd28@educarex.es'])

            msg.body = '{0}\n\nEl correo de contacto es {1}'.format(mensaje, email[0])
            mail.send(msg)

            return render_template('index.html', loged=True, LANGUAGES=LANGUAGES, current_language=get_locale())
    else:
        if user is not None:
            return render_template('index.html', loged=True, LANGUAGES=LANGUAGES, current_language=get_locale())
        else:
            return render_template('index.html', LANGUAGES=LANGUAGES, current_language=get_locale())


@app.route('/deleteProject', methods=['POST', 'GET'])
def deleteProject():

    if request.method == "POST":
        idProject = request.form.get('project_id_view')  # Se obtiene el id del proyecto

        user = session.get('usuario')

        idProjects = m.getProjectsId(user[0])

        idProjects_flattened = [str(id[0]) for id in idProjects]

        if idProject not in idProjects_flattened:
            projects = m.getProjects(user)

            return render_template('misProyectos.html', loged=True, history=projects, username=user[1], userId=user[0],
                                   LANGUAGES=LANGUAGES, current_language=get_locale(), errorIdentificador=True)
    else:
        idProject = session.get('project_id')

    nombreProyecto = m.obtenerNombreProyecto(idProject)

    m.deleteProject(idProject)

    nameUser = session.get('usuario')[1]
    if os.path.exists(f"./static/dirs/{nameUser}/Modelos/{nombreProyecto}"):
        shutil.rmtree(f"./static/dirs/{nameUser}/Modelos/{nombreProyecto}")

    if os.path.exists(f"./static/dirs/{nameUser}/Images/{nombreProyecto}"):
        shutil.rmtree(f"./static/dirs/{nameUser}/Images/{nombreProyecto}")

    if os.path.exists(f"./static/dirs/{nameUser}/{nombreProyecto}.zip"):
        os.remove(f"./static/dirs/{nameUser}/{nombreProyecto}.zip")

    return redirect(url_for('goMisProyectos'))


@app.route('/deleteProjects', methods=['POST', 'GET'])
def deleteProjects():

    idUser = session.get('usuario')[0]

    nameUser = session.get('usuario')[1]

    idProjects = m.getProjectsId(idUser)

    for i in range(len(idProjects)):
        idProject = idProjects[i]

        nombreProyecto = m.obtenerNombreProyecto(idProject)

        m.deleteProject(idProject)

        nameUser = session.get('usuario')[1]
        if os.path.exists(f"./static/dirs/{nameUser}/Modelos/{nombreProyecto}"):
            shutil.rmtree(f"./static/dirs/{nameUser}/Modelos/{nombreProyecto}")

        if os.path.exists(f"./static/dirs/{nameUser}/Images/{nombreProyecto}"):
            shutil.rmtree(f"./static/dirs/{nameUser}/Images/{nombreProyecto}")

        if os.path.exists(f"./static/dirs/{nameUser}/{nombreProyecto}.zip"):
            os.remove(f"./static/dirs/{nameUser}/{nombreProyecto}.zip")

    return redirect(url_for('goMisProyectos'))

@app.route('/receive_speech_CrearProyecto', methods=['POST'])
def receive_speech_CrearProyecto():
    texto = request.form['texto']
    print(texto)

    valores_checkboxes = request.form.getlist('valoresColumnas[]')
    print(valores_checkboxes)
    valores_checkboxes.remove('ns')
    print(valores_checkboxes)
    # Verifica si la lista de valores de los checkboxes está vacía
    if not valores_checkboxes:
        print("ENTREEEE")
        return jsonify({"valores":None})

    diccionario = m.procesarTexto(texto, valores_checkboxes)

    if diccionario == "none":
        return jsonify({"frase": None})

    diccionario = diccionario.replace("```json\n", "").replace("\n```", "")

    print(diccionario)
    print(type(diccionario))
    diccionario = ast.literal_eval(diccionario)
    print(type(diccionario))

    return jsonify(diccionario)


if __name__ == '__main__':
    app.run(debug=True)

import os
import pandas as pd
from io import BytesIO
from DBConnection import dataBaseExtractor
from flask_babel import gettext
import bcrypt
from .algoritmos import modelos, modeloArbolRegresion, modeloArbolClasificacion
from .generarGraficas import graficas, generarNomograma
import time
import requests
import zipfile
import numpy as np
from openai import OpenAI
from text_to_num import text2num
import unicodedata

#Método que recupera todos los projectos de un usuario
def getProjects(usuario):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')
    projects = db.obtainProjects(usuario[0]) #Recuperamos todos los proyectos de un usuario X
    new_projects = [] #Lista con los proyectos

    for i in range(len(projects)):  #Por cada proyecto
        errores = checkData(projects[i])     #Revisamos si tiene algún error
        dato = list(projects[i]) # Pasamos de tupla a lista

        if any(errores): #Si existe algún error en la lista entonces se añade true
            dato.append(True)
        else:
            dato.append(False)

        dato.append(i)
        new_projects.append(tuple(dato)) #Se añade el proyecto en la lista

    db.closeConnection() #Se cierra la conexión con la base de datos.

    return tuple(new_projects) #Devolvemos en forma de tupla los nuevos proyectos.

def deleteProject(idProject):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    db.deleteProject(idProject)

    db.closeConnection()

#Método que rescata un projecto a partir de su id
def getProject(idProject):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    project = db.obtainProject(idProject)

    db.closeConnection()
    return project


def getFile(idProject):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    project = db.obtainFile(idProject)

    db.closeConnection()

    return project


def getModel(idProject):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    project = db.obtainModel(idProject)

    db.closeConnection()

    return project


def getMetrics(idProject, problema):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    metrics = db.obtainMetrics(idProject, problema)

    db.closeConnection()

    return metrics


#Método para iniciar sesión
def login(username, password):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    usuario = db.obtainUser(username) #Obtenemos el usuario

    db.closeConnection()

    if usuario is None: #Comprobamos que el usuario existe
        return None

    storedHash = usuario[2]
    storedHash_bytes = storedHash.encode('utf-8')

    password = password.encode('utf-8') #Codificamos la constraseña

    if bcrypt.checkpw(password, storedHash_bytes):  #Comprobamos que la contraseña sea igual
        return usuario #Si lo es entonces devolvemos el usuario
    else:
        return None #Sino devolvemos None


def checkUser(username):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    exist = db.existUser(username)

    db.closeConnection()

    return exist


def registerUser(username, password, email):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    password = password.encode('utf-8')

    salt = bcrypt.gensalt()

    hash = bcrypt.hashpw(password, salt)

    db.addUser(username, hash, email)

    createDir(username)

    db.closeConnection()


#Método que actualiza la constraseña de un usuario
def updatePassword(username, password):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='') #Creamos la conexión con la base de datos

    password = password.encode('utf-8') #Encriptamos la contraseña

    salt = bcrypt.gensalt() #Generamos una sal

    hash = bcrypt.hashpw(password, salt) #Obtenemos el hash

    db.updatePassword(username, hash) #Actualizamos el hash en la base de datos

    db.closeConnection() #Cerramos la conexión con la base de datos.


#Método que crea un directorio para un usuario
def createDir(username):
    ruta_directorio = f"./static/dirs/{username}" #Creamos el directorio padre
    os.makedirs(ruta_directorio)

    ruta_datasets = os.path.join(ruta_directorio, "Modelos") #Creamos el primer directorio hijo
    os.makedirs(ruta_datasets)

    ruta_images = os.path.join(ruta_directorio, "Images") #Creamos el segundo directorio hijo
    os.makedirs(ruta_images)

    return ruta_directorio


# Método que devuelve un diccionario cuya clave es el nombre de la columna y el valor es el tipo de la columna,
# además devuelve los valores unicos de la clase positiva en caso de tenerla
def processForm(archivo, clase=None):
    df = pd.read_csv(archivo, low_memory=False) #Leemos el archivo
    # Construye un diccionario de columna: tipo de dato
    column_types = {col: str(df[col].dtype) for col in df.columns} #Creamos el diccionario
    if clase is not None: #Si tenemos clase positiva, entonces obtenemos sus valores
        col = df[clase]

        valores = col.unique()
    else:
        valores = None

    if not isinstance(archivo, str): #Si el archivo no es una cadena, entonces ponemos su puntero al principio del archivo.
        archivo.seek(0)

    return column_types, valores


#Método para crear un proyecto
def createProject(file, user, fileName=None):
    #Como se implementa un patrón singleton da igual cuantas veces creemos una conexión con la base de datos,
    # que solo se crea una vez. Lo mismo para cerrar la conexión solamente hace falta que se cierre de un sitio.
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    idProject = db.addProject(user[0]) #Creamos un proyecto y recuperamos su id

    db.addFiles(file, idProject, fileName) #Subimos el archivo a la base de datos y se asocia al proyecto creado

    db.addModel(idProject) #Creamos un modelo y se asocia al proyecto creado

    db.addMetrics(idProject) #Creamos las métricas y se asocia al proyecto creado

    db.closeConnection() #Cerramos la conexión a la base de datos.

    return idProject


#Método que actualiza los datos de un proyecto, en función del valor de origen se actualiza un campo
def updateData(newValue, idProyect, origen, user):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    success = True

    if origen == "tipoProblema": #Actualizamos el tipo de problema
        db.updateProblem(newValue, idProyect)

    elif origen == "clase": #Actualizamos el nombre de la clase objetivo
        db.updateNameClass(newValue, idProyect)

    elif origen == "nombreProyecto": #Actualizamos el nombre del proyecto.
        if not checkNameProject(user, newValue, db): #Comprobamos que el nombre no este ya asignado.
            db.updateNameProject(newValue, idProyect)
        else:
            print("Entre")
            success = False

    elif origen == "modelo": #Actualizamos el nombre del modelo
        db.updateNameModel(newValue, idProyect)

    elif origen == "externo": #Actualizamos las columnas externas
        db.updateColumnsExt(newValue, idProyect)

    elif origen == "control": #Actualizamos las columnas de control
        db.updateColumnsCont(newValue, idProyect)

    elif origen == "validacion": #Actualizamos el método de validación
        db.updateValidation(newValue, idProyect)

    elif origen == "clasePositiva": #Actualizamos la clase positiva.
        db.updateClasePositiva(newValue, idProyect)

    db.closeConnection() #Cerramos la conexión

    return success


def checkNameProject(user, nameProject, db):
    existName = db.existNameProject(user[0], nameProject)

    return existName


#Método que rescata los nombre únicos de los datasets de un usuario
def fetchDatasetsName(user):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    if user is not None: #Si el usuario no es None
        idProjects = db.obtainIdProjects(user[0]) #Obtenemos el id de los proyectos asociados al usuario

        if len(idProjects) != 0: #Si el tamaño de la tupla no es 0
            idProjectsList = [idProject[0] for idProject in idProjects] #Pasamos la lista a tupla

            datasets = db.obtainFiles(idProjectsList) #Obtenemos los nombre únicos de los datasets del usuario

            db.closeConnection() #Cerramos la conexión

            return datasets #Devolvemos los nombre
        else: #Si es 0
            db.closeConnection() #Cerramos la conexión

            return None #Devolvemos none
    else: #Si no existe
        db.closeConnection() #Cerramos la conexión

        return None #Devolvemos none


#Método que recupera un dataset en base al id de un proyecto
def fetchDataset(idProyecto):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    dataset = db.obtainFile(idProyecto) #Rescatamos el dataset

    db.closeConnection() #Cerramos la conexión con la base de datos.

    return dataset


#Método para comprobar que un proyecto no tenga errores
def checkData(data):
    errors = []
    print(data)
    if data[1] is None:
        errors.append(True)
    else:
        errors.append(False)

    if data[7] is None:
        errors.append(True)
    else:
        errors.append(False)

    if data[4] is None:
        errors.append(True)
    else:
        errors.append(False)

    if data[6] == "ns":

        errors.append(True)
        errors.append(True)  # Este es para la clase positiva
    else:
        errors.append(False)
        if data[6] == "clasificacion":
            if data[10] == "":
                errors.append(True)
            else:
                errors.append(False)

    return errors


#Método para construir el dataset en base a la configuración del proyecto
def buildNewDataset(resultsDataset, columnClass, columnsControl, columnsExternas):
    df = pd.read_csv(BytesIO(resultsDataset[0]))

    if columnsExternas[0] != '':
        nuevo_df = df[columnsControl + columnsExternas + [columnClass]]
    else:
        nuevo_df = df[columnsControl + [columnClass]]

    return nuevo_df


#Método para dividir el dataset en el entrenamiento y test
def splitDataset(dataset):
    return dataset.iloc[:, 0:-1], dataset.iloc[:, -1]


#Método que actualiza el estado de un proyecto
def actualizarEstadoProyecto(state, idProject, idUser):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    db.updateState(state, idProject)

    db.closeConnection()

    requests.get(f'http://localhost:3001/actualizarEstado?estado={state}&idProject={idProject}&idUser={idUser}')


#Método que actualiza el porcentaje de carga de un proyecto
def actualizarPorcentajeCargaInicio(inicio, fin, idProject):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    for i in range(inicio, fin):
        time.sleep(1)

        db.updatePorcentajeCarga(i, idProject)

    db.closeConnection()


def reinicarPorcentajeCarga(idProject):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    db.updatePorcentajeCarga(0, idProject)

    db.closeConnection()



#Método que crea el modelo seleccionado por el usuario
def model(dataset, validacion, valorValidacion, tipoProblema, columnaClase, tipoAlgoritmo, id_project, n_cores,
          rutaArchivo, nombreProyecto, nameUser, clasePositiva=None):
    print("El tipo de problema es ", tipoProblema)
    if tipoProblema == "regresion":
        m = modelos(dataset, validacion, valorValidacion, columnaClase, n_cores, nombreProyecto, nameUser, tipoProblema)

        if tipoAlgoritmo == "RF":
            m.randomForest(tipoProblema, id_project, rutaArchivo)

        elif tipoAlgoritmo == "RN":
            m.redesNeuronales(tipoProblema, id_project, rutaArchivo)

        elif tipoAlgoritmo == "RLIN":
            m.regresionLineal(tipoProblema, id_project, rutaArchivo)

        elif tipoAlgoritmo == "GBM":
            m.gradientBoostingMachine(tipoProblema, id_project, rutaArchivo)

        elif tipoAlgoritmo == "AML":
            m.autoML(tipoProblema, id_project, rutaArchivo)

        m.close()
    else:
        m = modelos(dataset, validacion, valorValidacion, columnaClase, n_cores, nombreProyecto, nameUser, tipoProblema,
                    clasePositiva)

        if tipoAlgoritmo == "RF":
            m.randomForest(tipoProblema, id_project, rutaArchivo)

        elif tipoAlgoritmo == "RN":
            m.redesNeuronales(tipoProblema, id_project, rutaArchivo)

        elif tipoAlgoritmo == "NB":
            m.naiveBayes(tipoProblema, id_project, rutaArchivo)

        elif tipoAlgoritmo == "RLOG":

            familia = "binomial" if len(dataset[columnaClase].unique()) < 3 else "multinomial"

            m.regresionLogistica(tipoProblema, id_project, familia, rutaArchivo)

        elif tipoAlgoritmo == "GBM":
            m.gradientBoostingMachine(tipoProblema, id_project, rutaArchivo)

        elif tipoAlgoritmo == "AML":
            m.autoML(tipoProblema, id_project, rutaArchivo)

        m.close()


#Método que convierte los valores numericos de la clase objetivo en string
def convert_numeric_target_values_to_string_list_pandas(dataset, target_column):
    # Extraer los valores únicos de la columna objetivo
    unique_values = dataset[target_column].unique()

    # Comprobar si la transformación es necesaria (si los valores son numéricos)
    if pd.api.types.is_numeric_dtype(dataset[target_column]):
        # Convertir cada valor numérico único a string en el formato "valor_str"
        string_values = [f"{value}_str" for value in unique_values]

        # Mapear los valores originales a los nuevos valores de string en el dataset
        mapping_dict = {value: f"{value}_str" for value in unique_values}
        dataset[target_column] = dataset[target_column].map(mapping_dict)
    else:
        # Si no es necesario transformar, usar los valores originales
        string_values = unique_values.astype(str).tolist()

    return string_values, dataset


#Método que genera las gráficas de explicabilidad de un modelo.
def generarGraficas(dataset, columnaClase, columnasControl, columnasExterior, n_cores, nombreProyecto, nombreUsuario,
                    modelo, rutaImagen, tipoProblema):

    multinominal = False if len(dataset[columnaClase].unique()) < 3 else True

    targets = []
    if multinominal and dataset[columnaClase].dtypes == np.int64:
        targets, dataset = convert_numeric_target_values_to_string_list_pandas(dataset, columnaClase)

        g = graficas(dataset, columnaClase, columnasControl, columnasExterior, n_cores, nombreProyecto, nombreUsuario,
                     modelo, tipoProblema)
    else:
        g = graficas(dataset, columnaClase, columnasControl, columnasExterior, n_cores, nombreProyecto, nombreUsuario,
                     modelo, tipoProblema)

    if len(targets) == 0:
        targets = dataset[columnaClase].unique()

    g.generarGraficoImportanciaVariables(rutaImagen, modelo)

    if modelo == ("AML"):
        if tipoProblema == "regresion":
            g.generarGraficoICEPlot(rutaImagen)

            g.generarGraficoDependencePlot(rutaImagen)

            g.generarGraficoShapValues(rutaImagen)

            g.generarGraficoVarimpHeatmap(rutaImagen)

            g.generarGraficaModelCorrelation(rutaImagen)

            g.generarGraficoCurvePlot(rutaImagen)

            g.generarDispersionResiduales(rutaImagen)
        else:
            if multinominal:
                g.generarGraficoICEPlot(rutaImagen, targets)

                g.generarGraficoDependencePlot(rutaImagen, targets)

                g.generarGraficoVarimpHeatmap(rutaImagen)

                g.generarGraficaModelCorrelation(rutaImagen)

                g.generarGraficoCurvePlot(rutaImagen)

                g.generarMatrizConfusion(rutaImagen)

                columnasControl = columnasControl.split(",")

                columnasControl.append(columnaClase)

                newDF = dataset[columnasControl]

                X, y = splitDataset(newDF)

                feature_names = [col for col in dataset.columns if col != columnaClase]

                generarNomograma(X, y, feature_names, rutaImagen)
            else:
                g.generarGraficoICEPlot(rutaImagen)

                g.generarGraficoDependencePlot(rutaImagen)

                g.generarGraficoShapValues(rutaImagen)

                g.generarGraficoVarimpHeatmap(rutaImagen)

                g.generarGraficaModelCorrelation(rutaImagen)

                g.generarGraficoCurvePlot(rutaImagen)

                g.generarMatrizConfusion(rutaImagen)
    else:
        if tipoProblema == "regresion":
            g.generarGraficoICEPlot(rutaImagen)

            g.generarGraficoDependencePlot(rutaImagen)

            g.generarGraficoShapValues(rutaImagen)

            g.generarGraficoCurvePlot(rutaImagen)

            g.generarDispersionResiduales(rutaImagen)

        else:
            if multinominal:
                g.generarGraficoICEPlot(rutaImagen, targets)

                g.generarGraficoDependencePlot(rutaImagen, targets)

                g.generarGraficoCurvePlot(rutaImagen)

                g.generarMatrizConfusion(rutaImagen)

                columnasControl = columnasControl.split(",")

                columnasControl.append(columnaClase)

                newDF = dataset[columnasControl]

                X, y = splitDataset(newDF)

                feature_names = [col for col in newDF.columns if col != columnaClase]

                generarNomograma(X, y, feature_names, rutaImagen)
            else:
                g.generarGraficoICEPlot(rutaImagen)

                g.generarGraficoDependencePlot(rutaImagen)

                g.generarGraficoShapValues(rutaImagen)

                g.generarGraficoCurvePlot(rutaImagen)

                g.generarMatrizConfusion(rutaImagen)

    g.closeH2o()

    carpeta1 = f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}"

    carpeta2 = f"./static/dirs/{nombreUsuario}/Images/{nombreProyecto}"

    archivo_zip = f"./static/dirs/{nombreUsuario}/{nombreProyecto}.zip"

    # Crear directorios intermedios si no existen
    if not os.path.exists(os.path.dirname(archivo_zip)):
        print(f"Creando directorios para {archivo_zip}")

        os.makedirs(os.path.dirname(archivo_zip), exist_ok=True)

    # Crear el archivo ZIP y agregar las carpetas
    if not os.path.exists(archivo_zip):
        print(f"Creando archivo zip {archivo_zip}")
        with zipfile.ZipFile(archivo_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.exists(carpeta1):
                print(f"Agregando carpeta {carpeta1}")
                zipdir(carpeta1, zipf)
            else:
                print(f"La carpeta {carpeta1} no existe y no se agregará al archivo zip.")

            if os.path.exists(carpeta2):
                print(f"Agregando carpeta {carpeta2}")
                zipdir(carpeta2, zipf)
            else:
                print(f"La carpeta {carpeta2} no existe y no se agregará al archivo zip.")

    else:
        print(f"El archivo {archivo_zip} ya existe.")


#Método que crea el zip
def crearZip(nombreProyecto, nombreUsuario, idProyecto):
    modelo = getModel(idProyecto)

    archivo1 = f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}/{modelo[0]}-Model-Predictions-{nombreUsuario}-{nombreProyecto}.csv"

    archivo2 = f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}/{modelo[0]}-Model-Metrics-{nombreUsuario}-{nombreProyecto}.csv"

    archivo_zip = f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}/{nombreProyecto}-{nombreUsuario}.zip"

    if not os.path.exists(archivo_zip):
        with zipfile.ZipFile(archivo_zip, 'w') as zipf:
            zipf.write(archivo1, os.path.basename(archivo1))  # El segundo argumento es el nombre del archivo en el ZIP

            zipf.write(archivo2, os.path.basename(archivo2))

    return archivo_zip


def zipdir(path, ziph):
    # ziph es el manejador del archivo ZIP
    for root, dirs, files in os.walk(path):
        for file in files:
            # Crear el path completo del archivo
            full_path = os.path.join(root, file)
            # Agregar el archivo al ZIP, incluyendo la estructura de carpetas relativa
            ziph.write(full_path, os.path.relpath(full_path, os.path.join(path, '..')))
            print(f"Agregando {full_path} al archivo zip.")


#Método que convierte modelo Tree de clasificación de sklearn en json
def convertirArbolAJSONClasificacion(X, y, feature_names=None, clasesTarget=None):
    decision_tree = modeloArbolClasificacion(X, y)

    from sklearn.tree import _tree

    tree_ = decision_tree.tree_

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    if clasesTarget is None:
        clasesTarget = [str(i) for i in decision_tree.classes_]

    def class_probabilities(values):
        total_samples = sum(values)

        probabilities = [value / total_samples for value in values]

        return probabilities

    def recurse(node, decision="Inicio"):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]

            threshold = tree_.threshold[node]

            # Aquí se crea la condición con el umbral para la característica
            name = "{} <= {:.2f}".format(name, threshold)

            left = recurse(tree_.children_left[node], decision="Yes")

            right = recurse(tree_.children_right[node], decision="No")
            # Se utiliza "children" en lugar de "left" y "right" para adaptarse al formato deseado
            node_data = {"name": name, "children": [left, right]}

            if decision is not None:
                # Agregamos la decisión como una propiedad del nodo
                node_data["decision"] = decision

            return node_data
        else:
            # Procesar las hojas para obtener las cuentas de las clases
            values = tree_.value[node][0]

            probabilities = class_probabilities(values)

            class_prob_str = [
                "{} ({}%)".format(clasesTarget[i], round(prob * 100, 2))
                for i, prob in enumerate(probabilities) if values[i] != 0
            ]
            value_str = ", ".join(class_prob_str)

            node_data = {"name": value_str, "decision": decision}

            return node_data

    return recurse(0)


#Método que convierte modelo Tree de regresión de sklearn en JSON
def convertirArbolAJSONRegresion(X, y, feature_names=None):
    decision_tree = modeloArbolRegresion(X, y)

    from sklearn.tree import _tree

    tree_ = decision_tree.tree_

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def regression_value(values):
        return values.mean()

    def recurse(node, decision="Inicio"):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]

            threshold = tree_.threshold[node]

            name = "{} <= {:.2f}".format(name, threshold)

            left = recurse(tree_.children_left[node], decision="Yes")

            right = recurse(tree_.children_right[node], decision="No")

            node_data = {"name": name, "children": [left, right], "decision": decision}

            return node_data
        else:
            # Calcula el valor promedio para la regresión
            value = regression_value(tree_.value[node][0])

            value_str = "{:.2f}".format(value)

            node_data = {"name": value_str, "decision": decision}

            return node_data

    return recurse(0)


#Método para rescartar el email de un usuario
def obtainEmail(username):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='') #Creamos la conexión para usuarios

    email = db.obtainEmail(username) #Obtenemos el email

    db.closeConnection() #Cerramos la conexión

    return email


#Método para genenrar el analisis de errores con GPT
def generateAnalisisError(ruta, idProject):
    client = OpenAI(
        api_key="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"
    )

    prompt = pd.read_csv(ruta)

    datos_texto = prompt.describe().to_string()

    prompt_texto = 'A partir de los datos que te envío, ' + datos_texto + \
                   (' haz un análisis numérico que permita explicar qué ha aprendido el modelo a partir de los datos. '
                    'Se lo más conciso posible y no pongas enumeración, guiones ni nada por el estilo, tampoco imágenes solamente texto plano. '
                    'Ten en cuenta que tu respuesta debe caber en un campo longtext de una base de datos.')
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt_texto
            }
        ],
        model="gpt-4-turbo-preview"
    )

    respuesta_texto = chat_completion.choices[0].message.content

    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    db.updateAnalisisModelo(respuesta_texto, idProject)

    db.closeConnection()


def procesarTexto(texto, valoresColumnas):
    client = OpenAI(
        api_key="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"
    )

    columnas_str = ", ".join(valoresColumnas)

    prompt_texto = ("Analiza la siguiente frase, el contenido más importante se tiene que extraer en forma de diccionario"
                    " y que se un diccionario. Los campos son: 'nombreProyecto', 'variablesControl' y 'variablesExternas'."
                    "Si el texto menciona variables de control o variables externas de manera no muy clara "
                    "(esto es a que el reconocimiento de voz está en español y el nombre de las variables en inglés"
                    ", emplea la siguientes "
                    f" columnas para seleccionar la más cercana o la que más te suene {columnas_str}. En ocasiones ten puede venir"
                    f"una frase tipo que contenga (o algo parecido) la palabra X, entonces tienes que buscar dicha subcadena X "
                    f"entre los nombre de las columnas anteriores y seleccionar aquellos que la contengan o te suene más, en "
                    f"caso de que ninguno te suene bien pon none, pero esto siempre tiene que ser el último intento. Y ya por último"
                    f" si la palabra te viene en español pero tu ves que en las que te he pasado vienen en inglés pues traduce y la que más se parezca. "
                    f"Si el texto no presenta información relevante (la antes indicada) devuelve none. El texto es: {texto}.")

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt_texto
            }
        ],
        model="gpt-4-turbo-preview"
    )
    print(chat_completion)
    respuesta = chat_completion.choices[0].message.content

    print(respuesta)

    return respuesta


#Método para obtener el análisis realizado a un modelo
def obtainAnalisisError(idProyecto):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    analisisError = db.obtainAnalisisModelo(idProyecto)

    db.closeConnection()

    return analisisError


def comprobarNombreyAccionProyecto(identificadorProyecto, idUser, accion):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    nombres = db.obtainNameProjects(idUser)

    db.closeConnection()

    if identificadorProyecto > len(nombres) or identificadorProyecto < 0:
        return {'identificador': False}
    else:
        proyecto = nombres[identificadorProyecto]

    print(proyecto)

    # Inicializamos el identificador de coincidencia como None
    id_coincidencia = None
    statusProyecto = None

    # Buscamos la cadena en las tuplas modificadas y guardamos el id si encontramos una coincidencia
    for num, nombre, status in nombres:
        if proyecto[1] == nombre:
            id_coincidencia = num
            statusProyecto = status
            break  # Detenemos el bucle si encontramos una coincidencia

    print(id_coincidencia)
    print(statusProyecto)
    # Mostramos si se encontró o no la coincidencia y el identificador asociado

    if statusProyecto == "Done":
        if accion in ["visualizar", "copiar", "eliminar", "borrar"]:
            print("ENTREEE AQUIIII")
            return id_coincidencia
        elif accion == "ejecutar":
            return {'accion': False}
    elif statusProyecto == "Not started":
        if accion == "visualizar":
            return {'accion': False}
        elif accion in ["ejecutar", "copiar", "eliminar", "borrar"]:
            return id_coincidencia
    elif statusProyecto == "Running" and accion == "copiar":
        return id_coincidencia

    return {'accion': False}


def obtenerNombreProyecto(idProject):
    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    nombre = db.obtainNameProject(idProject)

    db.closeConnection()

    return nombre[0]


def convertir_numeros_en_nombre(nombre):

    nombre = quitar_tildes(nombre)

    try:
        # Intenta convertir de texto a número
        numero = text2num(nombre, "es")
    except ValueError:
        # Si ocurre un error, asume que nombre ya es un número y trata de convertirlo a int
        try:
            numero = int(nombre)
        except ValueError:
            # Si la conversión a int falla, maneja el caso (puedes decidir cómo manejarlo)
            numero = None

    return numero


def insert_newlines_after_four_dots(text):
    # Inicializar variables para la construcción del nuevo texto, el conteo de puntos y la comprobación de puntos decimales
    new_text = ""
    temp_count = 0
    i = 0  # Índice para recorrer el texto

    # Recorrer cada carácter en el texto
    while i < len(text):
        char = text[i]

        # Agregar el carácter al nuevo texto
        new_text += char

        # Comprobar si el carácter es un punto
        if char == '.':
            # Comprobar si el punto no está rodeado por dígitos (no es parte de un número decimal)
            if (i == 0 or not text[i - 1].isdigit()) or (i == len(text) - 1 or not text[i + 1].isdigit()):
                temp_count += 1

                # Si se han encontrado 4 puntos no decimales, agregar un salto de línea y resetear el contador
                if temp_count == 4:
                    new_text += "\n\n"
                    temp_count = 0

        i += 1

    return new_text


def quitar_tildes(texto):
    texto_sin_tildes = ''.join(
        (c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    )
    return texto_sin_tildes


def getProjectsId(idUser):
    print(idUser)

    db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

    idProjects = db.obtainIdProjects(idUser)

    db.closeConnection()

    return idProjects


def procesarDataset(idProject):
    resultadoDataset = getFile(idProject)

    project = getProject(idProject)

    lista_finalColumnasControl = project[7].split(",")

    if project[8] is not None:
        lista_finalColumnasExternas = project[8].split(",")
    else:
        lista_finalColumnasExternas = ['']

    dataset = buildNewDataset(resultadoDataset, project[4], lista_finalColumnasControl,
                                lista_finalColumnasExternas)

    return dataset
import pymysql
from pymysql import converters
from datetime import datetime
from io import BytesIO
import os


class dataBaseExtractor:

    # Constructor de la clase para crear la conexión con la base de datos
    def __init__(self, host, database, user, password):

        try:
            # Código para la conexión y ejecución de consultas
            self.conn = pymysql.connect(host=host, user=user, password=password, database=database)
            self.cursor = self.conn.cursor()

            if self.conn:
                print("Se ha establecido correctamente la conexión con la base de datos");


        except pymysql.Error as e:
            # Manejo de la excepción
            print("Ocurrió un error en la base de datos:", e)
            self.cursor.close()
            self.conn.close()

    # Method to fetch a user based on their username and password
    def obtainUser(self, user):
        self.conn.converter = converters.escape_str

        consulta = "SELECT * FROM Usuario WHERE username = %s"

        self.cursor.execute(consulta, (user))

        usuario = self.cursor.fetchone()

        return usuario

    def obtainHashUser(self, user):
        self.conn.converter = converters.escape_str

        consulta = "SELECT password FROM Usuario WHERE username = %s"

        self.cursor.execute(consulta, (user))

        hash = self.cursor.fetchone()

        return hash

    def updateNameClass(self, nameClass, idProject):
        consulta = "UPDATE Proyectos SET clase = %s where idProyecto = %s"
        self.cursor.execute(consulta, (nameClass, idProject))
        self.conn.commit()

    def updateProblem(self, tipoProblema, idProject):
        consulta = "UPDATE Proyectos SET tipoProblema = %s where idProyecto = %s"
        self.cursor.execute(consulta, (tipoProblema, idProject))
        self.conn.commit()

    def updateNameProject(self, nameProject, idProject):
        consulta = "UPDATE Proyectos SET nombreProyecto = %s where idProyecto = %s"
        self.cursor.execute(consulta, (nameProject, idProject))
        self.conn.commit()

    def updateNameModel(self, nameModel, idProject):
        consulta = "UPDATE Modelos SET nombreModelo = %s where idProyecto = %s"
        self.cursor.execute(consulta, (nameModel, idProject))
        self.conn.commit()

    def updateColumnsExt(self, columns, idProject):
        consulta = "UPDATE Proyectos SET nombresColumnasExterior = %s where idProyecto = %s"
        self.cursor.execute(consulta, (columns, idProject))
        self.conn.commit()

    def updateColumnsCont(self, columns, idProject):
        consulta = "UPDATE Proyectos SET nombresColumnasControl = %s where idProyecto = %s"
        self.cursor.execute(consulta, (columns, idProject))
        self.conn.commit()

    def updateValidation(self, validation, idProject):
        consulta = "UPDATE Proyectos SET validacion = %s where idProyecto = %s"
        self.cursor.execute(consulta, (validation, idProject))
        self.conn.commit()

    def updateClasePositiva(self, clasePositiva, idProject):
        consulta = "UPDATE Proyectos SET clasePositiva = %s where idProyecto = %s"
        self.cursor.execute(consulta, (clasePositiva, idProject))
        self.conn.commit()

    def updatePorcentajeCarga(self, porcentaje, idProject):
        consulta = "UPDATE Proyectos SET porcentajeCarga= %s where idProyecto = %s"
        self.cursor.execute(consulta, (porcentaje, idProject))
        self.conn.commit()

    def updateState(self, state, idProject):
        consulta = "UPDATE Proyectos SET status = %s where idProyecto = %s"
        self.cursor.execute(consulta, (state, idProject))
        self.conn.commit()

    def updateMetricsRegression(self, MAE, MAPE, MSE, RMSE, R2, idProyecto):
        consulta = "UPDATE Metricas SET MAE = %s, MAPE = %s, MSE = %s, RMSE = %s, R2 = %s where idProyecto = %s"
        self.cursor.execute(consulta, (MAE, MAPE, MSE, RMSE, R2, idProyecto))
        self.conn.commit()

    def updateMetricsClassification(self, Accuracy, Precision, Sensibilidad, F1, Especificidad, NPV, idProyecto):
        consulta = "UPDATE Metricas SET Accuracy = %s, _Precision = %s, Sensibilidad = %s, F1 = %s, Especificidad = %s, NPV = %s where idProyecto = %s"
        self.cursor.execute(consulta, (Accuracy, Precision, Sensibilidad, F1, Especificidad, NPV, idProyecto))
        self.conn.commit()

    def obtainIdProjects(self, idUser):
        consulta = "SELECT idProyecto FROM Proyectos WHERE idUsuario = %s"

        self.cursor.execute(consulta, (idUser))

        results = self.cursor.fetchall()

        return results

    def obtainFiles(self, idProyects):

        placeholders = ', '.join(['%s'] * len(tuple(idProyects)))

        consulta = f"SELECT nombreFichero, MIN(idProyecto) FROM Datasets WHERE idProyecto IN ({placeholders}) GROUP BY nombreFichero"

        self.cursor.execute(consulta, (idProyects))

        results = self.cursor.fetchall()

        return results

    def addUser(self, username, password, email):

        self.conn.converter = converters.escape_str

        consulta = "INSERT INTO Usuario (username, password, email) VALUES (%s, %s, %s)"

        self.cursor.execute(consulta, (username, password, email))

        self.conn.commit()

    # Método que almacena la consulta del usuario
    def addProject(self, idUser):
        fecha_actual = datetime.now()
        fecha_actual_formateada = fecha_actual.strftime("%Y-%m-%d")

        consulta = "INSERT INTO Proyectos (nombreProyecto, status, fecha, clase, idUsuario, tipoProblema, nombresColumnasControl, nombresColumnasExterior, validacion, clasePositiva, porcentajeCarga) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

        self.cursor.execute(consulta,
                            (None, "Not started", fecha_actual_formateada, None, idUser, None, None, None, "HO-70", None, 0))
        id_insertado = self.cursor.lastrowid
        self.conn.commit()

        return id_insertado

    # Método para añadir los ficheros asociados a una búsqueda
    def addFiles(self, file, idProyecto, filename=None):

        try:
            # Si 'file' tiene un atributo 'stream', lo usamos
            contenido = file.stream.read()
            file.stream.seek(0)
            filename = file.filename
        except AttributeError:
            # Si no tiene 'stream', asumimos que es un objeto BytesIO
            contenido = file.read()
            file.seek(0)

        consulta = "INSERT INTO Datasets (idProyecto, fichero, nombreFichero) VALUES (%s, %s, %s)"
        self.cursor.execute(consulta, (idProyecto, contenido, filename))
        self.conn.commit()

    def addModel(self, idProyecto):
        consulta = "INSERT INTO Modelos (idProyecto, analisisModelo, nombreModelo) VALUES (%s, %s, %s)"
        self.cursor.execute(consulta, (idProyecto, None, "RF"))
        self.conn.commit()

    def addMetrics(self, idProyecto):
        consulta = "INSERT INTO Metricas (idProyecto, MAE, MAPE, MSE, RMSE, R2, Accuracy, _Precision, Sensibilidad, F1, Especificidad, NPV) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.cursor.execute(consulta, (idProyecto, None, None, None, None, None, None, None, None, None, None, None))
        self.conn.commit()

    # Método para obtener todas las búsquedas que ha hecho un usuario
    def obtainProjects(self, idUser):
        consulta = "SELECT * FROM Proyectos WHERE idUsuario = %s"

        self.cursor.execute(consulta, (idUser))

        results = self.cursor.fetchall()

        return results

    def obtainNameProjects(self, idUser):
        consulta = "SELECT idProyecto, nombreProyecto, status FROM Proyectos WHERE idUsuario = %s"

        self.cursor.execute(consulta, (idUser))

        results = self.cursor.fetchall()

        return results

    def obtainProject(self, idProject):
        consulta = "SELECT * FROM Proyectos WHERE idProyecto = %s"

        self.cursor.execute(consulta, (idProject))

        results = self.cursor.fetchone()

        return results

    def obtainFile(self, idProject):
        consulta = "SELECT fichero, nombreFichero FROM Datasets WHERE idProyecto = %s"

        self.cursor.execute(consulta, (idProject))

        results = self.cursor.fetchone()

        return results

    def obtainModel(self, idProject):
        consulta = "SELECT nombreModelo FROM Modelos WHERE idProyecto = %s"

        self.cursor.execute(consulta, (idProject))

        results = self.cursor.fetchone()

        return results


    def obtainMetrics(self, idProject, problema):
        if problema == "clasificacion":
            consulta = "SELECT Accuracy, _Precision, Sensibilidad, F1, Especificidad, NPV FROM Metricas WHERE idProyecto = %s"
        else:
            consulta = "SELECT MAE, MAPE, MSE, RMSE, R2 FROM Metricas WHERE idProyecto = %s"

        self.cursor.execute(consulta, (idProject))

        results = self.cursor.fetchone()

        return results

    def obtainEmail(self, username):
        print(username)

        consulta = "SELECT email FROM Usuario WHERE username = %s"

        self.cursor.execute(consulta, (username))

        results = self.cursor.fetchone()

        return results

    def obtainNameProject(self, idProject):
        consulta = "SELECT nombreProyecto from Proyectos where idProyecto = %s"

        self.cursor.execute(consulta, (idProject))

        results = self.cursor.fetchone()

        return results

    # Método para validar la existencia de un usuario
    def existUser(self, username):
        consulta = "SELECT * FROM Usuario WHERE username = %s"

        self.cursor.execute(consulta, (username))

        usuario = self.cursor.fetchone()

        if usuario is None:
            return False
        else:
            return True

    def existNameProject(self, idUser, nameProject):
        consulta = "SELECT * FROM Proyectos WHERE idUsuario = %s AND nombreProyecto = %s"

        self.cursor.execute(consulta, (idUser, nameProject))

        result = self.cursor.fetchone()

        if result is None:
            return False
        else:
            return True

    def updatePassword(self, username, hash):
        consulta = "UPDATE Usuario SET password= %s where username = %s"
        self.cursor.execute(consulta, (hash, username))
        self.conn.commit()

    def updateAnalisisModelo(self, analisisProyecto, idProyecto):
        consulta = "UPDATE Modelos SET analisisModelo = %s where idProyecto = %s"
        self.cursor.execute(consulta, (analisisProyecto, idProyecto))
        self.conn.commit()

    def obtainAnalisisModelo(self, idProyecto):
        consulta = "SELECT analisisModelo FROM Modelos WHERE idProyecto = %s"

        self.cursor.execute(consulta, (idProyecto))

        analisisModelo = self.cursor.fetchone()

        return analisisModelo

    def deleteProject(self, idProject):
        consulta = "DELETE FROM Proyectos WHERE idProyecto = %s"

        self.cursor.execute(consulta, (idProject,))

        self.conn.commit()

    # Método para cerrar la conexión con la base de datos
    def closeConnection(self):
        self.cursor.close()
        self.conn.close()
        print("Se ha cerrado la conexión con la BD")

import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2ONaiveBayesEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from DBConnection import dataBaseExtractor
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


class modelos:

    def __init__(self, dataset, validacion, valorValidacion, columnaClase, n_cores, nombreProyecto, nombreUsuario, tipoProblema, clasePositiva=None):
        h2o.init(ip="127.0.0.1")

        self.dataset = h2o.H2OFrame(python_obj=dataset)

        self.validacion = validacion

        self.valorValidacion = valorValidacion

        self.columnaClase = columnaClase

        self.clasePositiva = clasePositiva

        self.nombreProyecto = nombreProyecto

        self.nameUsuario = nombreUsuario

        if tipoProblema == "clasificacion":
            clase = dataset[columnaClase]

            self.valoresClase = clase.unique()

            multiclase = len(clase.unique())

            self.multiclase = multiclase

    def randomForest(self, tipoProblema, idProyecto, rutaArchivo):
        db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

        if tipoProblema == "clasificacion":
            self.dataset[self.columnaClase] = self.dataset[self.columnaClase].asfactor()

        if self.validacion == "HO":
            train, test = self.dataset.split_frame(ratios=[self.valorValidacion])

            model = H2ORandomForestEstimator()

            model.train(y=self.columnaClase, training_frame=train)

            predictions = model.predict(test)

            y_pred = h2o.as_list(predictions)['predict'].values

            # Obtén el tipo de datos de la columna de clase
            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = test.as_data_frame()

            y_true = h2o.as_list(test[self.columnaClase]).values

            y_true_flat = y_true.ravel()

            if tipoProblema == "regresion":
                MAE = mean_absolute_error(y_true_flat, y_pred)
                MSE = mean_squared_error(y_true_flat, y_pred)
                RMSE = np.sqrt(MSE)
                MAPE = mean_absolute_percentage_error(y_true_flat, y_pred)  # Ver si se calcula a mano o no.
                R2 = r2_score(y_true_flat, y_pred)  # Cálculo del R cuadrado

                df_metricas = pd.DataFrame({'MAE': [MAE],
                                            'MSE': [MSE],
                                            'RMSE': [RMSE],
                                            'MAPE': [MAPE],
                                            'R^2': [R2]})  # Agregar R^2 al DataFrame

                df_metricas.to_csv(rutaArchivo + f"/RF-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                df_resultado = pd.DataFrame({'predict': y_pred})
                error = np.abs(y_true_flat - y_pred)
                df_error = pd.DataFrame({'Error': error})
                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/RF-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                # Actualizar la base de datos con R^2 además de las otras métricas
                db.updateMetricsRegression(MAE, MAPE, MSE, RMSE, R2, idProyecto)
            else:
                acc = accuracy_score(y_true_flat, y_pred_rounded)

                recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                if self.multiclase <= 2:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded)

                    # Extrae TN, FP, FN, TP
                    TN, FP, FN, TP = cm.ravel()

                    # Calcula Especificidad (para clasificación binaria)
                    especificidad = TN / (TN + FP)

                    # Calcula NPV (para clasificación binaria)
                    NPV = TN / (TN + FN)
                else:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                    clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                    TP = cm[clase_positiva_index, clase_positiva_index]
                    FP = cm[:, clase_positiva_index].sum() - TP
                    FN = cm[clase_positiva_index, :].sum() - TP
                    TN = cm.sum() - TP - FP - FN

                    # Asegúrate de no dividir por cero
                    especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

                df_metricas = pd.DataFrame({'Accuracy': [acc],
                                            'Recall': [recall],
                                            'Precision': [precision],
                                            'F1 Score': [f1],
                                            'Specificity': [especificidad],
                                            'NPV': [NPV]})

                df_resultado = pd.DataFrame({'predict': y_pred})

                error = (y_true_flat != y_pred_rounded).astype(int)

                df_error = pd.DataFrame({'Error': error})

                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/RF-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                df_metricas.to_csv(rutaArchivo + f"/RF-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv", index=False)

                db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)
        else:
            model = H2ORandomForestEstimator(nfolds=int(self.valorValidacion), keep_cross_validation_predictions=True)

            model.train(y=self.columnaClase, training_frame=self.dataset)

            y_pred = model.cross_validation_holdout_predictions().as_data_frame().values
            y_true = self.dataset[self.columnaClase].as_data_frame().values
            y_true_flat = y_true.ravel()
            y_pred = y_pred[:, 0]

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = self.dataset.as_data_frame()

            df_resultado = pd.DataFrame({'predict': y_pred})

            df_combinado = pd.concat([df, df_resultado], axis=1)

            df_combinado.to_csv(rutaArchivo + f"/RF-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                index=False)

            if tipoProblema == "regresion":
                MAE = mean_absolute_error(y_true_flat, y_pred)
                MSE = mean_squared_error(y_true_flat, y_pred)
                RMSE = np.sqrt(MSE)
                MAPE = mean_absolute_percentage_error(y_true_flat, y_pred)  # Ver si se calcula a mano o no.
                R2 = r2_score(y_true_flat, y_pred)  # Cálculo del R cuadrado

                df_metricas = pd.DataFrame({'MAE': [MAE],
                                            'MSE': [MSE],
                                            'RMSE': [RMSE],
                                            'MAPE': [MAPE],
                                            'R^2': [R2]})  # Agregar R^2 al DataFrame

                df_metricas.to_csv(rutaArchivo + f"/RF-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                df_resultado = pd.DataFrame({'predict': y_pred})
                error = np.abs(y_true_flat - y_pred)
                df_error = pd.DataFrame({'Error': error})
                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/RF-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                # Actualizar la base de datos con R^2 además de las otras métricas
                db.updateMetricsRegression(MAE, MAPE, MSE, RMSE, R2, idProyecto)
            else:
                acc = accuracy_score(y_true_flat, y_pred_rounded)

                recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                if self.multiclase <= 2:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded)

                    # Extrae TN, FP, FN, TP
                    TN, FP, FN, TP = cm.ravel()

                    # Calcula Especificidad (para clasificación binaria)
                    especificidad = TN / (TN + FP)

                    # Calcula NPV (para clasificación binaria)
                    NPV = TN / (TN + FN)
                else:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                    clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                    TP = cm[clase_positiva_index, clase_positiva_index]
                    FP = cm[:, clase_positiva_index].sum() - TP
                    FN = cm[clase_positiva_index, :].sum() - TP
                    TN = cm.sum() - TP - FP - FN

                    # Asegúrate de no dividir por cero
                    especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

                df_metricas = pd.DataFrame({'Accuracy': [acc],
                                            'Recall': [recall],
                                            'Precision': [precision],
                                            'F1 Score': [f1],
                                            'Specificity': [especificidad],
                                            'NPV': [NPV]})

                df_resultado = pd.DataFrame({'predict': y_pred})

                error = (y_true_flat != y_pred_rounded).astype(int)

                df_error = pd.DataFrame({'Error': error})

                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/RF-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                df_metricas.to_csv(rutaArchivo + f"/RF-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)

        db.closeConnection()

    def redesNeuronales(self, tipoProblema, idProyecto, rutaArchivo):
        db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

        if tipoProblema == "clasificacion":
            self.dataset[self.columnaClase] = self.dataset[self.columnaClase].asfactor()

        if self.validacion == "HO":
            train, test = self.dataset.split_frame(ratios=[self.valorValidacion])

            model = H2ODeepLearningEstimator()

            model.train(y=self.columnaClase, training_frame=train)

            predictions = model.predict(test)

            y_pred = h2o.as_list(predictions)['predict'].values

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = test.as_data_frame()


            y_true = h2o.as_list(test[self.columnaClase]).values

            y_true_flat = y_true.ravel()

            if tipoProblema == "regresion":
                MAE = mean_absolute_error(y_true_flat, y_pred)
                MSE = mean_squared_error(y_true_flat, y_pred)
                RMSE = np.sqrt(MSE)
                MAPE = mean_absolute_percentage_error(y_true_flat, y_pred)  # Ver si se calcula a mano o no.
                R2 = r2_score(y_true_flat, y_pred)  # Cálculo del R cuadrado

                df_metricas = pd.DataFrame({'MAE': [MAE],
                                            'MSE': [MSE],
                                            'RMSE': [RMSE],
                                            'MAPE': [MAPE],
                                            'R^2': [R2]})  # Agregar R^2 al DataFrame

                df_metricas.to_csv(rutaArchivo + f"/RN-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                df_resultado = pd.DataFrame({'predict': y_pred})
                error = np.abs(y_true_flat - y_pred)
                df_error = pd.DataFrame({'Error': error})
                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/RN-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                # Actualizar la base de datos con R^2 además de las otras métricas
                db.updateMetricsRegression(MAE, MAPE, MSE, RMSE, R2, idProyecto)  # Asumiendo que db.updateMetricsRegression puede manejar R^2
            else:
                acc = accuracy_score(y_true_flat, y_pred_rounded)

                recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                if self.multiclase <= 2:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded)

                    # Extrae TN, FP, FN, TP
                    TN, FP, FN, TP = cm.ravel()

                    # Calcula Especificidad (para clasificación binaria)
                    especificidad = TN / (TN + FP)

                    # Calcula NPV (para clasificación binaria)
                    NPV = TN / (TN + FN)
                else:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                    clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                    TP = cm[clase_positiva_index, clase_positiva_index]
                    FP = cm[:, clase_positiva_index].sum() - TP
                    FN = cm[clase_positiva_index, :].sum() - TP
                    TN = cm.sum() - TP - FP - FN

                    # Asegúrate de no dividir por cero
                    especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

                df_metricas = pd.DataFrame({'Accuracy': [acc],
                                            'Recall': [recall],
                                            'Precision': [precision],
                                            'F1 Score': [f1],
                                            'Specificity': [especificidad],
                                            'NPV': [NPV]})

                df_resultado = pd.DataFrame({'predict': y_pred})

                error = (y_true_flat != y_pred_rounded).astype(int)

                df_error = pd.DataFrame({'Error': error})

                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/RN-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                df_metricas.to_csv(rutaArchivo + f"/RN-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)
        else:
            model = H2ODeepLearningEstimator(nfolds=int(self.valorValidacion), keep_cross_validation_predictions=True)

            model.train(y=self.columnaClase, training_frame=self.dataset)

            y_pred = model.cross_validation_holdout_predictions().as_data_frame().values

            y_true = self.dataset[self.columnaClase].as_data_frame().values

            y_true_flat = y_true.ravel()

            y_pred = y_pred[:, 0]

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = self.dataset.as_data_frame()

            if tipoProblema == "regresion":
                MAE = mean_absolute_error(y_true_flat, y_pred)
                MSE = mean_squared_error(y_true_flat, y_pred)
                RMSE = np.sqrt(MSE)
                MAPE = mean_absolute_percentage_error(y_true_flat, y_pred)  # Ver si se calcula a mano o no.
                R2 = r2_score(y_true_flat, y_pred)  # Cálculo del R cuadrado

                df_metricas = pd.DataFrame({'MAE': [MAE],
                                            'MSE': [MSE],
                                            'RMSE': [RMSE],
                                            'MAPE': [MAPE],
                                            'R^2': [R2]})  # Agregar R^2 al DataFrame

                df_metricas.to_csv(rutaArchivo + f"/RN-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                df_resultado = pd.DataFrame({'predict': y_pred})
                error = np.abs(y_true_flat - y_pred)
                df_error = pd.DataFrame({'Error': error})
                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/RN-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                # Actualizar la base de datos con R^2 además de las otras métricas
                db.updateMetricsRegression(MAE, MAPE, MSE, RMSE, R2, idProyecto)  # Asumiendo que db.updateMetricsRegression puede manejar R^2
            else:
                acc = accuracy_score(y_true_flat, y_pred_rounded)

                recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                if self.multiclase <= 2:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded)

                    # Extrae TN, FP, FN, TP
                    TN, FP, FN, TP = cm.ravel()

                    # Calcula Especificidad (para clasificación binaria)
                    especificidad = TN / (TN + FP)

                    # Calcula NPV (para clasificación binaria)
                    NPV = TN / (TN + FN)
                else:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                    clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                    TP = cm[clase_positiva_index, clase_positiva_index]
                    FP = cm[:, clase_positiva_index].sum() - TP
                    FN = cm[clase_positiva_index, :].sum() - TP
                    TN = cm.sum() - TP - FP - FN

                    # Asegúrate de no dividir por cero
                    especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

                df_metricas = pd.DataFrame({'Accuracy': [acc],
                                            'Recall': [recall],
                                            'Precision': [precision],
                                            'F1 Score': [f1],
                                            'Specificity': [especificidad],
                                            'NPV': [NPV]})

                df_resultado = pd.DataFrame({'predict': y_pred})

                error = (y_true_flat != y_pred_rounded).astype(int)

                df_error = pd.DataFrame({'Error': error})

                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/RN-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                df_metricas.to_csv(rutaArchivo + f"/RN-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)

        db.closeConnection()


    def naiveBayes(self,  tipoProblema, idProyecto, rutaArchivo):
        db = dataBaseExtractor(host='localhost', database='BD-TFG', user='root', password='')

        if tipoProblema == "clasificacion":
            self.dataset[self.columnaClase] = self.dataset[self.columnaClase].asfactor()

        if self.validacion == "HO":
            train, test = self.dataset.split_frame(ratios=[self.valorValidacion])

            model = H2ONaiveBayesEstimator()

            model.train(y=self.columnaClase, training_frame=train)

            predictions = model.predict(test)

            y_pred = h2o.as_list(predictions)['predict'].values

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = test.as_data_frame()

            y_true = h2o.as_list(test[self.columnaClase]).values

            y_true_flat = y_true.ravel()

            df_resultado = pd.DataFrame({'predict': y_pred})

            error = (y_true_flat != y_pred_rounded).astype(int)

            df_error = pd.DataFrame({'Error': error})

            df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

            df_combinado.to_csv(rutaArchivo + f"/NB-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                index=False)

            acc = accuracy_score(y_true_flat, y_pred_rounded)

            recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            if self.multiclase <= 2:
                cm = confusion_matrix(y_true_flat, y_pred_rounded)

                # Extrae TN, FP, FN, TP
                TN, FP, FN, TP = cm.ravel()

                # Calcula Especificidad (para clasificación binaria)
                especificidad = TN / (TN + FP)

                # Calcula NPV (para clasificación binaria)
                NPV = TN / (TN + FN)
            else:
                cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                TP = cm[clase_positiva_index, clase_positiva_index]
                FP = cm[:, clase_positiva_index].sum() - TP
                FN = cm[clase_positiva_index, :].sum() - TP
                TN = cm.sum() - TP - FP - FN

                # Asegúrate de no dividir por cero
                especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

            df_metricas = pd.DataFrame({'Accuracy': [acc],
                                        'Recall': [recall],
                                        'Precision': [precision],
                                        'F1 Score': [f1],
                                        'Specificity': [especificidad],
                                        'NPV': [NPV]})

            df_metricas.to_csv(rutaArchivo + f"/NB-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv", index=False)

            db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)
        else:
            model = H2ONaiveBayesEstimator(nfolds=int(self.valorValidacion), keep_cross_validation_predictions=True)

            model.train(y=self.columnaClase, training_frame=self.dataset)

            y_pred = model.cross_validation_holdout_predictions().as_data_frame().values

            y_true = self.dataset[self.columnaClase].as_data_frame().values

            y_true_flat = y_true.ravel()

            y_pred = y_pred[:, 0]

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = self.dataset.as_data_frame()

            df_resultado = pd.DataFrame({'predict': y_pred})

            error = (y_true_flat != y_pred_rounded).astype(int)

            df_error = pd.DataFrame({'Error': error})

            df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

            df_combinado.to_csv(rutaArchivo + f"/NB-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                index=False)

            acc = accuracy_score(y_true_flat, y_pred_rounded)

            recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            if self.multiclase <= 2:
                cm = confusion_matrix(y_true_flat, y_pred_rounded)

                # Extrae TN, FP, FN, TP
                TN, FP, FN, TP = cm.ravel()

                # Calcula Especificidad (para clasificación binaria)
                especificidad = TN / (TN + FP)

                # Calcula NPV (para clasificación binaria)
                NPV = TN / (TN + FN)
            else:
                cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                TP = cm[clase_positiva_index, clase_positiva_index]
                FP = cm[:, clase_positiva_index].sum() - TP
                FN = cm[clase_positiva_index, :].sum() - TP
                TN = cm.sum() - TP - FP - FN

                # Asegúrate de no dividir por cero
                especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                NPV = TN / (TN + FN) if (TN + FN) > 0 else 0


            df_metricas = pd.DataFrame({'Accuracy': [acc],
                                        'Recall': [recall],
                                        'Precision': [precision],
                                        'F1 Score': [f1],
                                        'Specificity': [especificidad],
                                        'NPV': [NPV]})

            df_metricas.to_csv(rutaArchivo + f"/NB-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                               index=False)

            db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)

        db.closeConnection()

    def regresionLineal(self, tipoProblema, idProyecto, rutaArchivo):

        db = dataBaseExtractor('localhost', 'BD-TFG', 'root', '')

        if self.validacion == "HO":
            train, test = self.dataset.split_frame(ratios=[self.valorValidacion])

            model = H2OGeneralizedLinearEstimator(family="gaussian")

            model.train(y=self.columnaClase, training_frame=train)

            predictions = model.predict(test)

            y_pred = h2o.as_list(predictions)['predict'].values

            df = test.as_data_frame()

            y_true = h2o.as_list(test[self.columnaClase]).values

            y_true_flat = y_true.ravel()

            df_resultado = pd.DataFrame({'predict': y_pred})

            error = np.abs(y_true_flat - y_pred)

            df_error = pd.DataFrame({'Error': error})

            df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

            df_combinado.to_csv(rutaArchivo + f"/RLIN-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                index=False)

            MAE = mean_absolute_error(y_true_flat, y_pred)

            MSE = mean_squared_error(y_true_flat, y_pred)

            RMSE = np.sqrt(MSE)

            MAPE = mean_absolute_percentage_error(y_true_flat, y_pred)

            R2 = r2_score(y_true_flat, y_pred)  # Cálculo del R cuadrado

            df_metricas = pd.DataFrame({'MAE': [MAE],
                                        'MSE': [MSE],
                                        'RMSE': [RMSE],
                                        'MAPE': [MAPE],
                                        'R^2': [R2]})  # Agregar R^2 al DataFrame

            df_metricas.to_csv(rutaArchivo + f"/RLIN-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv", index=False)

            db.updateMetricsRegression(MAE, MAPE, MSE, RMSE, R2, idProyecto)

        else:
            model = H2OGeneralizedLinearEstimator(family="gaussian", nfolds=int(self.valorValidacion), keep_cross_validation_predictions=True)

            model.train(y=self.columnaClase, training_frame=self.dataset)

            y_pred = model.cross_validation_holdout_predictions().as_data_frame().values

            y_true = self.dataset[self.columnaClase].as_data_frame().values

            y_true_flat = y_true.ravel()

            y_pred = y_pred[:, 0]

            df = self.dataset.as_data_frame()

            df_resultado = pd.DataFrame({'predict': y_pred})

            error = np.abs(y_true_flat - y_pred)

            df_error = pd.DataFrame({'Error': error})

            df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

            df_combinado.to_csv(rutaArchivo + f"/RLIN-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                index=False)

            MAE = mean_absolute_error(y_true_flat, y_pred)

            MSE = mean_squared_error(y_true_flat, y_pred)

            RMSE = np.sqrt(MSE)

            MAPE = mean_absolute_percentage_error(y_true_flat, y_pred)

            R2 = r2_score(y_true_flat, y_pred)  # Cálculo del R cuadrado

            df_metricas = pd.DataFrame({'MAE': [MAE],
                                        'MSE': [MSE],
                                        'RMSE': [RMSE],
                                        'MAPE': [MAPE],
                                        'R^2': [R2]})  # Agregar R^2 al DataFrame

            df_metricas.to_csv(rutaArchivo + f"/RLIN-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv", index=False)

            db.updateMetricsRegression(MAE, MAPE, MSE, RMSE, R2, idProyecto)

        db.closeConnection()

    def regresionLogistica(self, tipoProblema, idProyecto, familia, rutaArchivo):

        db = dataBaseExtractor('localhost', 'BD-TFG', 'root', '')

        self.dataset[self.columnaClase] = self.dataset[self.columnaClase].asfactor()

        if self.validacion == "HO":
            train, test = self.dataset.split_frame(ratios=[self.valorValidacion])

            model = H2OGeneralizedLinearEstimator(family=familia)

            model.train(y=self.columnaClase, training_frame=train)

            predictions = model.predict(test)

            y_pred = h2o.as_list(predictions)['predict'].values

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = test.as_data_frame()

            y_true = h2o.as_list(test[self.columnaClase]).values

            y_true_flat = y_true.ravel()

            df_resultado = pd.DataFrame({'predict': y_pred})

            error = (y_true_flat != y_pred_rounded).astype(int)

            df_error = pd.DataFrame({'Error': error})

            df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

            df_combinado.to_csv(rutaArchivo + f"/RLOG-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                index=False)

            acc = accuracy_score(y_true_flat, y_pred_rounded)

            recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            if self.multiclase <= 2:
                cm = confusion_matrix(y_true_flat, y_pred_rounded)

                # Extrae TN, FP, FN, TP
                TN, FP, FN, TP = cm.ravel()

                # Calcula Especificidad (para clasificación binaria)
                especificidad = TN / (TN + FP)

                # Calcula NPV (para clasificación binaria)
                NPV = TN / (TN + FN)
            else:
                cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                TP = cm[clase_positiva_index, clase_positiva_index]
                FP = cm[:, clase_positiva_index].sum() - TP
                FN = cm[clase_positiva_index, :].sum() - TP
                TN = cm.sum() - TP - FP - FN

                # Asegúrate de no dividir por cero
                especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

            df_metricas = pd.DataFrame({'Accuracy': [acc],
                                        'Recall': [recall],
                                        'Precision': [precision],
                                        'F1 Score': [f1],
                                        'Specificity': [especificidad],
                                        'NPV': [NPV]})

            df_metricas.to_csv(rutaArchivo + f"/RLOG-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv", index=False)

            db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)
        else:
            model = H2OGeneralizedLinearEstimator(family=familia, nfolds=int(self.valorValidacion), keep_cross_validation_predictions=True)

            model.train(y=self.columnaClase, training_frame=self.dataset)

            y_pred = model.cross_validation_holdout_predictions().as_data_frame().values

            y_true = self.dataset[self.columnaClase].as_data_frame().values

            y_true_flat = y_true.ravel()

            y_pred = y_pred[:, 0]

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = self.dataset.as_data_frame()

            df_resultado = pd.DataFrame({'predict': y_pred})

            error = (y_true_flat != y_pred_rounded).astype(int)

            df_error = pd.DataFrame({'Error': error})

            df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

            df_combinado.to_csv(rutaArchivo + f"/RLOG-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                index=False)

            acc = accuracy_score(y_true_flat, y_pred_rounded)

            recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

            if self.multiclase <= 2:
                cm = confusion_matrix(y_true_flat, y_pred_rounded)

                # Extrae TN, FP, FN, TP
                TN, FP, FN, TP = cm.ravel()

                # Calcula Especificidad (para clasificación binaria)
                especificidad = TN / (TN + FP)

                # Calcula NPV (para clasificación binaria)
                NPV = TN / (TN + FN)
            else:
                cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                TP = cm[clase_positiva_index, clase_positiva_index]
                FP = cm[:, clase_positiva_index].sum() - TP
                FN = cm[clase_positiva_index, :].sum() - TP
                TN = cm.sum() - TP - FP - FN

                # Asegúrate de no dividir por cero
                especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

            df_metricas = pd.DataFrame({'Accuracy': [acc],
                                        'Recall': [recall],
                                        'Precision': [precision],
                                        'F1 Score': [f1],
                                        'Specificity': [especificidad],
                                        'NPV': [NPV]})

            df_metricas.to_csv(rutaArchivo + f"/RLOG-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                               index=False)

            db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)

        db.closeConnection()


    def gradientBoostingMachine(self, tipoProblema, idProyecto, rutaArchivo):

        db = dataBaseExtractor('localhost', 'BD-TFG', 'root', '')

        if tipoProblema == "clasificacion":
            self.dataset[self.columnaClase] = self.dataset[self.columnaClase].asfactor()

        if self.validacion == "HO":
            train, test = self.dataset.split_frame(ratios=[self.valorValidacion])

            model = H2OGradientBoostingEstimator()

            model.train(y=self.columnaClase, training_frame=train)

            predictions = model.predict(test)

            y_pred = h2o.as_list(predictions)['predict'].values

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = test.as_data_frame()

            y_true = h2o.as_list(test[self.columnaClase]).values

            y_true_flat = y_true.ravel()

            if tipoProblema == "regresion":
                MAE = mean_absolute_error(y_true_flat, y_pred)
                MSE = mean_squared_error(y_true_flat, y_pred)
                RMSE = np.sqrt(MSE)
                MAPE = mean_absolute_percentage_error(y_true_flat, y_pred)  # Ver si se calcula a mano o no.
                R2 = r2_score(y_true_flat, y_pred)  # Cálculo del R cuadrado

                df_metricas = pd.DataFrame({'MAE': [MAE],
                                            'MSE': [MSE],
                                            'RMSE': [RMSE],
                                            'MAPE': [MAPE],
                                            'R^2': [R2]})  # Agregar R^2 al DataFrame

                df_metricas.to_csv(rutaArchivo + f"/GBM-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                df_resultado = pd.DataFrame({'predict': y_pred})
                error = np.abs(y_true_flat - y_pred)
                df_error = pd.DataFrame({'Error': error})
                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/GBM-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                # Actualizar la base de datos con R^2 además de las otras métricas
                db.updateMetricsRegression(MAE, MAPE, MSE, RMSE, R2,
                                           idProyecto)  # Asumiendo que db.updateMetricsRegression puede manejar R^2
            else:
                acc = accuracy_score(y_true_flat, y_pred_rounded)

                recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                if self.multiclase <= 2:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded)

                    # Extrae TN, FP, FN, TP
                    TN, FP, FN, TP = cm.ravel()

                    # Calcula Especificidad (para clasificación binaria)
                    especificidad = TN / (TN + FP)

                    # Calcula NPV (para clasificación binaria)
                    NPV = TN / (TN + FN)
                else:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                    clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                    TP = cm[clase_positiva_index, clase_positiva_index]
                    FP = cm[:, clase_positiva_index].sum() - TP
                    FN = cm[clase_positiva_index, :].sum() - TP
                    TN = cm.sum() - TP - FP - FN

                    # Asegúrate de no dividir por cero
                    especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

                df_metricas = pd.DataFrame({'Accuracy': [acc],
                                            'Recall': [recall],
                                            'Precision': [precision],
                                            'F1 Score': [f1],
                                            'Specificity': [especificidad],
                                            'NPV': [NPV]})

                df_resultado = pd.DataFrame({'predict': y_pred})

                error = (y_true_flat != y_pred_rounded).astype(int)

                df_error = pd.DataFrame({'Error': error})

                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/GBM-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                df_metricas.to_csv(rutaArchivo + f"/GBM-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)
        else:
            model = H2OGradientBoostingEstimator(nfolds=int(self.valorValidacion), keep_cross_validation_predictions=True)

            model.train(y=self.columnaClase, training_frame=self.dataset)

            y_pred = model.cross_validation_holdout_predictions().as_data_frame().values

            y_true = self.dataset[self.columnaClase].as_data_frame().values

            y_true_flat = y_true.ravel()

            y_pred = y_pred[:, 0]

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = self.dataset.as_data_frame()

            if tipoProblema == "regresion":
                MAE = mean_absolute_error(y_true_flat, y_pred)
                MSE = mean_squared_error(y_true_flat, y_pred)
                RMSE = np.sqrt(MSE)
                MAPE = mean_absolute_percentage_error(y_true_flat, y_pred)  # Ver si se calcula a mano o no.
                R2 = r2_score(y_true_flat, y_pred)  # Cálculo del R cuadrado

                df_metricas = pd.DataFrame({'MAE': [MAE],
                                            'MSE': [MSE],
                                            'RMSE': [RMSE],
                                            'MAPE': [MAPE],
                                            'R^2': [R2]})  # Agregar R^2 al DataFrame

                df_metricas.to_csv(rutaArchivo + f"/GBM-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                df_resultado = pd.DataFrame({'predict': y_pred})
                error = np.abs(y_true_flat - y_pred)
                df_error = pd.DataFrame({'Error': error})
                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/RF-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                # Actualizar la base de datos con R^2 además de las otras métricas
                db.updateMetricsRegression(MAE, MAPE, MSE, RMSE, R2,
                                           idProyecto)  # Asumiendo que db.updateMetricsRegression puede manejar R^2
            else:
                acc = accuracy_score(y_true_flat, y_pred_rounded)

                recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                if self.multiclase <= 2:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded)

                    # Extrae TN, FP, FN, TP
                    TN, FP, FN, TP = cm.ravel()

                    # Calcula Especificidad (para clasificación binaria)
                    especificidad = TN / (TN + FP)

                    # Calcula NPV (para clasificación binaria)
                    NPV = TN / (TN + FN)
                else:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                    clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                    TP = cm[clase_positiva_index, clase_positiva_index]
                    FP = cm[:, clase_positiva_index].sum() - TP
                    FN = cm[clase_positiva_index, :].sum() - TP
                    TN = cm.sum() - TP - FP - FN

                    # Asegúrate de no dividir por cero
                    especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

                df_metricas = pd.DataFrame({'Accuracy': [acc],
                                            'Recall': [recall],
                                            'Precision': [precision],
                                            'F1 Score': [f1],
                                            'Specificity': [especificidad],
                                            'NPV': [NPV]})

                df_resultado = pd.DataFrame({'predict': y_pred})

                error = (y_true_flat != y_pred_rounded).astype(int)

                df_error = pd.DataFrame({'Error': error})

                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/GBM-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                df_metricas.to_csv(rutaArchivo + f"/GBM-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)

        db.closeConnection()

    def autoML(self, tipoProblema, idProyecto, rutaArchivo):

        db = dataBaseExtractor('localhost', 'BD-TFG', 'root', '')

        if tipoProblema == "clasificacion":
            self.dataset[self.columnaClase] = self.dataset[self.columnaClase].asfactor()

        if self.validacion == "HO":
            train, test = self.dataset.split_frame(ratios=[self.valorValidacion])

            model = H2OAutoML(max_models=5)

            model.train(y=self.columnaClase, training_frame=train)

            predictions = model.predict(test)

            y_pred = h2o.as_list(predictions)['predict'].values

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = test.as_data_frame()

            y_true = h2o.as_list(test[self.columnaClase]).values

            y_true_flat = y_true.ravel()

            if tipoProblema == "regresion":
                MAE = mean_absolute_error(y_true_flat, y_pred)
                MSE = mean_squared_error(y_true_flat, y_pred)
                RMSE = np.sqrt(MSE)
                MAPE = mean_absolute_percentage_error(y_true_flat, y_pred)  # Ver si se calcula a mano o no.
                R2 = r2_score(y_true_flat, y_pred)  # Cálculo del R cuadrado

                df_metricas = pd.DataFrame({'MAE': [MAE],
                                            'MSE': [MSE],
                                            'RMSE': [RMSE],
                                            'MAPE': [MAPE],
                                            'R^2': [R2]})  # Agregar R^2 al DataFrame

                df_metricas.to_csv(rutaArchivo + f"/AML-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                df_resultado = pd.DataFrame({'predict': y_pred})
                error = np.abs(y_true_flat - y_pred)
                df_error = pd.DataFrame({'Error': error})
                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/AML-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                # Actualizar la base de datos con R^2 además de las otras métricas
                db.updateMetricsRegression(MAE, MAPE, MSE, RMSE, R2,
                                           idProyecto)  # Asumiendo que db.updateMetricsRegression puede manejar R^2
            else:
                acc = accuracy_score(y_true_flat, y_pred_rounded)

                recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                if self.multiclase <= 2:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded)

                    # Extrae TN, FP, FN, TP
                    TN, FP, FN, TP = cm.ravel()

                    # Calcula Especificidad (para clasificación binaria)
                    especificidad = TN / (TN + FP)

                    # Calcula NPV (para clasificación binaria)
                    NPV = TN / (TN + FN)
                else:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                    clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                    TP = cm[clase_positiva_index, clase_positiva_index]
                    FP = cm[:, clase_positiva_index].sum() - TP
                    FN = cm[clase_positiva_index, :].sum() - TP
                    TN = cm.sum() - TP - FP - FN

                    # Asegúrate de no dividir por cero
                    especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

                df_metricas = pd.DataFrame({'Accuracy': [acc],
                                            'Recall': [recall],
                                            'Precision': [precision],
                                            'F1 Score': [f1],
                                            'Specificity': [especificidad],
                                            'NPV': [NPV]})

                df_resultado = pd.DataFrame({'predict': y_pred})

                error = (y_true_flat != y_pred_rounded).astype(int)

                df_error = pd.DataFrame({'Error': error})

                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/AML-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                df_metricas.to_csv(rutaArchivo + f"/AML-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)
        else:
            model = H2OAutoML(max_models=5, nfolds=int(self.valorValidacion), keep_cross_validation_predictions=True, exclude_algos=["StackedEnsemble"])

            model.train(y=self.columnaClase, training_frame=self.dataset)

            leader = model.leader

            y_pred = leader.cross_validation_holdout_predictions().as_data_frame().values

            y_true = self.dataset[self.columnaClase].as_data_frame().values

            y_true_flat = y_true.ravel()

            y_pred = y_pred[:, 0]

            columna_clase_tipo = self.dataset[self.columnaClase].dtype

            # Si la columna de clase es entera, redondea y convierte y_pred a enteros
            if np.issubdtype(columna_clase_tipo, np.integer):
                y_pred_rounded = np.around(y_pred).astype(int)
                positiva = int(self.clasePositiva)
            else:
                y_pred_rounded = y_pred  # Mantén y_pred como está si la columna de clase es categórica
                positiva = self.clasePositiva

            df = self.dataset.as_data_frame()

            if tipoProblema == "regresion":
                MAE = mean_absolute_error(y_true_flat, y_pred)
                MSE = mean_squared_error(y_true_flat, y_pred)
                RMSE = np.sqrt(MSE)
                MAPE = mean_absolute_percentage_error(y_true_flat, y_pred)  # Ver si se calcula a mano o no.
                R2 = r2_score(y_true_flat, y_pred)  # Cálculo del R cuadrado

                df_metricas = pd.DataFrame({'MAE': [MAE],
                                            'MSE': [MSE],
                                            'RMSE': [RMSE],
                                            'MAPE': [MAPE],
                                            'R^2': [R2]})  # Agregar R^2 al DataFrame

                df_metricas.to_csv(rutaArchivo + f"/AML-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                df_resultado = pd.DataFrame({'predict': y_pred})
                error = np.abs(y_true_flat - y_pred)
                df_error = pd.DataFrame({'Error': error})
                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/AML-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                # Actualizar la base de datos con R^2 además de las otras métricas
                db.updateMetricsRegression(MAE, MAPE, MSE, RMSE, R2,
                                           idProyecto)  # Asumiendo que db.updateMetricsRegression puede manejar R^2
            else:
                acc = accuracy_score(y_true_flat, y_pred_rounded)

                recall = recall_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                precision = precision_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                f1 = f1_score(y_true_flat, y_pred_rounded, average='weighted', labels=[positiva])

                if self.multiclase <= 2:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded)

                    # Extrae TN, FP, FN, TP
                    TN, FP, FN, TP = cm.ravel()

                    # Calcula Especificidad (para clasificación binaria)
                    especificidad = TN / (TN + FP)

                    # Calcula NPV (para clasificación binaria)
                    NPV = TN / (TN + FN)
                else:
                    cm = confusion_matrix(y_true_flat, y_pred_rounded, labels=self.valoresClase)

                    clase_positiva_index = self.valoresClase.tolist().index(self.clasePositiva)

                    TP = cm[clase_positiva_index, clase_positiva_index]
                    FP = cm[:, clase_positiva_index].sum() - TP
                    FN = cm[clase_positiva_index, :].sum() - TP
                    TN = cm.sum() - TP - FP - FN

                    # Asegúrate de no dividir por cero
                    especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
                    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0

                df_metricas = pd.DataFrame({'Accuracy': [acc],
                                            'Recall': [recall],
                                            'Precision': [precision],
                                            'F1 Score': [f1],
                                            'Specificity': [especificidad],
                                            'NPV': [NPV]})

                df_resultado = pd.DataFrame({'predict': y_pred})

                error = (y_true_flat != y_pred_rounded).astype(int)

                df_error = pd.DataFrame({'Error': error})

                df_combinado = pd.concat([df, df_resultado, df_error], axis=1)

                df_combinado.to_csv(rutaArchivo + f"/AML-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                    index=False)

                df_metricas.to_csv(rutaArchivo + f"/AML-Model-Metrics-{self.nameUsuario}-{self.nombreProyecto}.csv",
                                   index=False)

                db.updateMetricsClassification(acc, precision, recall, f1, especificidad, NPV, idProyecto)

        db.closeConnection()


    def close(self):
        h2o.cluster().shutdown()


def modeloArbolRegresion(X, y):
    tree_reg = DecisionTreeRegressor(max_depth=7)
    tree_reg.fit(X, y)

    return tree_reg


def modeloArbolClasificacion(X, y):
    tree_clf = DecisionTreeClassifier(max_depth=7)
    tree_clf.fit(X, y)

    return tree_clf
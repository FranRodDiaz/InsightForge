import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2ONaiveBayesEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML
from .algoritmos import modelos, modeloArbolRegresion, modeloArbolClasificacion
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .nomogram import nomogram
import pickle
from sklearn.metrics import confusion_matrix

class graficas:

    def __init__(self, dataset, columnaClase, columnasControl, columnasExterior, n_cores, nombreProyecto, nombreUsuario, modelo, tipoProblema):
        h2o.init(ip="127.0.0.1", max_mem_size="3G", nthreads=2)
        self.dataset = h2o.H2OFrame(python_obj=dataset)
        self.columnaClase = columnaClase
        self.columnasControl = columnasControl
        self.columnasExterior = columnasExterior
        self.nombreProyecto = nombreProyecto
        self.tipoProblema = tipoProblema
        self.nameUsuario = nombreUsuario
        self.modelo = modelo

        if modelo == "RF":
            model = H2ORandomForestEstimator()

            model.train(y=self.columnaClase, training_frame=self.dataset)

            h2o.save_model(model=model, path=f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}")

            self.model = model

        elif modelo == "RN":
            model = H2ODeepLearningEstimator()

            model.train(y=self.columnaClase, training_frame=self.dataset)

            h2o.save_model(model=model, path=f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}")

            self.model = model

        elif modelo == "RLIN":
            model = H2OGeneralizedLinearEstimator(family="gaussian")

            model.train(y=self.columnaClase, training_frame=self.dataset)

            self.model = model

        elif modelo == "SVM":
            model = H2OSupportVectorMachineEstimator()

            model.train(y=self.columnaClase, training_frame=self.dataset)

            h2o.save_model(model=model, path=f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}")

            self.model = model

        elif modelo == "GBM":
            model = H2OGradientBoostingEstimator()

            model.train(y=self.columnaClase, training_frame=self.dataset)

            h2o.save_model(model=model, path=f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}")

            self.model = model

        elif modelo == "AML":
            model = H2OAutoML(max_models=5, exclude_algos=["StackedEnsemble"])

            model.train(y=self.columnaClase, training_frame=self.dataset)

            h2o.save_model(model=model.leader, path=f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}")

            self.model = model

        elif modelo == "NB":
            model = H2ONaiveBayesEstimator()

            model.train(y=self.columnaClase, training_frame=self.dataset)

            h2o.save_model(model=model, path=f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}")

            self.model = model

        elif modelo == "RLOG":

            familia = "binomial" if len(self.dataset[self.columnaClase].unique()) < 3 else "multinomial"

            model = H2OGeneralizedLinearEstimator(family=familia)

            model.train(y=self.columnaClase, training_frame=self.dataset)

            h2o.save_model(model=model, path=f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}")

            self.model = model

        ruta = f"./static/dirs/{nombreUsuario}/Modelos/{nombreProyecto}/decision_tree_model.pkl"
        X = dataset.iloc[:, 0:-1]
        y = dataset.iloc[:, -1]
        if tipoProblema == "regresion":
            arbol = modeloArbolRegresion(X, y)
        else:
            arbol = modeloArbolClasificacion(X, y)

        with open(ruta, 'wb') as file:
            pickle.dump(arbol, file)

    def generarGraficoImportanciaVariables(self, ruta, modelo):

        if modelo == "AML":
            importance_data = self.model.leader.varimp(use_pandas=True)
        else:
            importance_data = self.model.varimp(use_pandas=True)

        num_variables = len(importance_data['variable'])
        fig_height = max(10, num_variables * 0.5)

        plt.figure(figsize=(8, fig_height))

        if self.columnasExterior is not None:
            importance_data['color_group'] = ['selected' if var in self.columnasExterior else 'other' for var in importance_data['variable']]

        # Crear la visualización usando seaborn
            sns.barplot(x='scaled_importance', y='variable', hue='color_group', data=importance_data,
                    palette={'selected': 'tomato', 'other': 'lightseagreen'}, dodge=False)

            plt.xlabel('')
            plt.ylabel('')
            plt.title('Importancia de las Variables')
            legend_labels = [mpatches.Patch(color='tomato', label='Variables externas'),
                         mpatches.Patch(color='lightseagreen', label='Variables de control')]

            legend = plt.legend(handles=legend_labels, title='Variables ', loc='center left', bbox_to_anchor=(1, 0.5))

            plt.tight_layout()
            plt.savefig(f"{ruta}/importanciaVariables.pdf", format='pdf', bbox_extra_artists=(legend,),
                        bbox_inches='tight')
            plt.close()
        else:
            sns.barplot(x='scaled_importance', y='variable', color="lightseagreen",data=importance_data, dodge=False)

            # Configurar el título y los límites del eje x
            plt.xlabel('')
            plt.ylabel('')
            plt.title('Importancia de las Variables')

            plt.tight_layout()
            plt.savefig(f"{ruta}/importanciaVariables.pdf", format='pdf', bbox_inches='tight')
            plt.close()

    def generarGraficoICEPlot(self, ruta, targets=None):
            if self.modelo == "AML":
                model = self.model.leader

                listaVariables = list(self.columnasControl.split(","))

                for i in range(len(listaVariables)):
                    col_type = self.dataset.type(i)
                    if col_type == "string":
                        continue

                    if targets is not None:
                        for j in range(len(targets)):
                            dependence = model.ice_plot(self.dataset, listaVariables[i], target=targets[j])
                            plt.savefig(f"{ruta}/ICE plot-{listaVariables[i]}-{targets[j]}.pdf", format='pdf',
                                        bbox_inches='tight')
                            plt.close()
                    else:
                        dependence = model.ice_plot(self.dataset, listaVariables[i])
                        plt.savefig(f"{ruta}/ICE plot-{listaVariables[i]}.pdf", format='pdf', bbox_inches='tight')
                        plt.close()

            else:
                listaVariables = list(self.columnasControl.split(","))

                for i in range(len(listaVariables)):
                    col_type = self.dataset.type(i)
                    if col_type == "string":
                        continue

                    if targets is not None:
                        for j in range(len(targets)):
                            dependence = self.model.ice_plot(self.dataset, listaVariables[i], target=targets[j])
                            plt.savefig(f"{ruta}/ICE plot-{listaVariables[i]}-{targets[j]}.pdf", format='pdf', bbox_inches='tight')
                            plt.close()
                    else:
                        dependence = self.model.ice_plot(self.dataset, listaVariables[i])
                        plt.savefig(f"{ruta}/ICE plot-{listaVariables[i]}.pdf", format='pdf', bbox_inches='tight')
                        plt.close()

    def generarGraficoDependencePlot(self, ruta, targets=None):

        if self.modelo == "AML":
            listaVariables = list(self.columnasControl.split(","))

            for i in range(len(listaVariables)):
                col_type = self.dataset.type(i)
                if col_type == "string":
                    continue

                if targets is not None:
                    for j in range(len(targets)):
                        dependence = self.model.pd_multi_plot(self.dataset, listaVariables[i], target=targets[j])
                        plt.savefig(f"{ruta}/dependece plot-{listaVariables[i]}-{targets[j]}.pdf", format='pdf',
                                    bbox_inches='tight')
                        plt.close()
                else:
                    dependence = self.model.pd_multi_plot(self.dataset, listaVariables[i])
                    plt.savefig(f"{ruta}/dependece plot-{listaVariables[i]}.pdf", format='pdf', bbox_inches='tight')
                    plt.close()

        else:
            listaVariables = list(self.columnasControl.split(","))

            for i in range(len(listaVariables)):
                col_type = self.dataset.type(i)
                if col_type == "string":
                    continue

                if targets is not None:
                    for j in range(len(targets)):
                        dependence = self.model.pd_plot(self.dataset, listaVariables[i], target=targets[j])
                        plt.savefig(f"{ruta}/dependecePlot-{listaVariables[i]}-{targets[j]}.pdf", format='pdf',
                                bbox_inches='tight')
                        plt.close()
                else:
                    dependence = self.model.pd_plot(self.dataset, listaVariables[i])
                    plt.savefig(f"{ruta}/dependecePlot-{listaVariables[i]}.pdf", format='pdf', bbox_inches='tight')
                    plt.close()

    def generarGraficoShapValues(self, ruta):
        listaVariables = list(self.columnasControl.split(","))

        if self.modelo == "AML":
            shap = self.model.leader.shap_summary_plot(self.dataset, columns=listaVariables)
        else:

            if self.tipoProblema == "clasificacion":
                model = H2ORandomForestEstimator()

                model.train(y=self.columnaClase, training_frame=self.dataset)

                shap = model.shap_summary_plot(self.dataset, columns=listaVariables)
            else:
                shap = self.model.shap_summary_plot(self.dataset, columns=listaVariables)

        plt.savefig(f"{ruta}/shapValues.pdf", format='pdf',
                        bbox_inches='tight')
        plt.close()


    def generarGraficoVarimpHeatmap(self, ruta):

        varimpHeatmap = self.model.varimp_heatmap()

        plt.savefig(f"{ruta}/variableImportanceHeatmap.pdf", format='pdf',
                        bbox_inches='tight')
        plt.close()

    def generarGraficaModelCorrelation(self, ruta):

        modelCorrelation = self.model.model_correlation_heatmap(self.dataset)

        plt.savefig(f"{ruta}/modelCorrelation.pdf", format='pdf',
                        bbox_inches='tight')
        plt.close()

    def generarGraficoCurvePlot(self, ruta):

        if self.modelo == "AML":
            model = self.model.leader
            curve = model.learning_curve_plot()
        else:
            curve = self.model.learning_curve_plot()

        plt.savefig(f"{ruta}/learningCurvePlot.pdf", format='pdf',
                    bbox_inches='tight')
        plt.close()

    def generarMatrizConfusion(self, ruta):
        rutaArchivo = f'./static/dirs/{self.nameUsuario}/Modelos/{self.nombreProyecto}/{self.modelo}-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv'

        print(rutaArchivo)

        data = pd.read_csv(rutaArchivo)

        conf_matrix = confusion_matrix(data[self.columnaClase], data['predict'], labels=data[self.columnaClase].unique())

        axis_labels = data[self.columnaClase].unique()

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=axis_labels, yticklabels=axis_labels)
        plt.title('Matriz de Confusión')
        plt.xlabel('Clase Real')
        plt.ylabel('Clase Predicha')
        plt.savefig(f"{ruta}/matrizConfusion.pdf", format='pdf', bbox_inches='tight')
        plt.close()

    def generarDispersionResiduales(self, ruta):
        rutaArchivo = f'./static/dirs/{self.nameUsuario}/Modelos/{self.nombreProyecto}/{self.modelo}-Model-Predictions-{self.nameUsuario}-{self.nombreProyecto}.csv'

        data = pd.read_csv(rutaArchivo)

        plt.figure(figsize=(10, 6))
        plt.scatter(data[self.columnaClase], data['predict'], color='blue', alpha=0.5)
        plt.plot([data[self.columnaClase].min(), data[self.columnaClase].max()], [data[self.columnaClase].min(), data[self.columnaClase].max()], color='red', linestyle='--')
        plt.title('Gráfica de valores reales vs predicciones')
        plt.xlabel('Valores Reales')
        plt.ylabel('Valores Predichos')
        plt.savefig(f"{ruta}/graficaValoresResiduales.pdf", format='pdf', bbox_inches='tight')
        plt.close()

    def closeH2o(self):
        h2o.cluster().shutdown()


def generarNomograma(X, y, feature_names, rutaImagen):
    # Identificar columnas no numéricas (categóricas)
    cols_no_numericas = X.select_dtypes(include=['object', 'category']).columns

    X = X.drop(columns=cols_no_numericas)

    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X, y)

    threshold_value = 0.5  # Valor específico del threshold
    templates = []
    class_names = log_reg.classes_
    # Determinar los tipos de las características (nominal o continuous)
    feature_types = []
    for feature in feature_names:
        if X[feature].dtype == 'int' or X[feature].dtype == 'object':
            feature_types.append("nominal")
        else:
            feature_types.append("continuous")

    for class_index in range(log_reg.coef_.shape[0]):
        # Creamos el DataFrame con las características y el intercepto
        df = pd.DataFrame({
            "feature": feature_names + ["intercept"],
            "coef": list(log_reg.coef_[class_index]) + [log_reg.intercept_[class_index]],
            "min": list(X.min(axis=0)) + [""],
            "max": list(X.max(axis=0)) + [""],
            "type": feature_types + ["intercept"],
            "position": [""] * (len(feature_names) + 1)
        })

        # Agregamos la fila del threshold al DataFrame
        df = pd.concat([df, pd.DataFrame({
            "feature": ["threshold"],
            "coef": [threshold_value],
            "min": [""],
            "max": [""],
            "type": ["threshold"],
            "position": [""]
        })], ignore_index=True)

        # Añadir el DataFrame a la lista de plantillas
        templates.append(df)

    for i in range(len(templates)):
        df = templates[i]

        df['feature'] = df['feature'].str.replace('_', ' ', regex=False)

        df.to_excel(rutaImagen + '/nomogramaArchivo-' + class_names[i] + ".xlsx", index=False)

        nomo = nomogram(
            rutaImagen + '/nomogramaArchivo-' + class_names[i] + ".xlsx",
            result_title=class_names[i],
            fig_width=20,
            single_height=0.65,
            dpi=400,
            ax_para={"c": "black", "linewidth": 1.3, "linestyle": "-"},
            tick_para={"direction": 'in', "length": 3, "width": 1.5, },
            xtick_para={"fontsize": 10, "fontfamily": "Arial", "fontweight": "bold"},  # Cambiado a Arial
            ylabel_para={"fontsize": 12, "fontname": "Arial", "labelpad": 50,  # Cambiado a Arial
                         "loc": "center", "color": "black", "rotation": "horizontal"},
            total_point=100
        )
        nomo.savefig(f"{rutaImagen}/Nomogram-{class_names[i]}.pdf", format='pdf',
                    bbox_inches='tight')


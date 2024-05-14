$(document).ready(function() {

    $(document).on('click', '#pagination-dependence a, #pagination-ice a', function(event) {
        event.preventDefault();
    });

    $(document).on('click', '#descargarArbol', function (event){
        prepararBoton()
    });

    var itemsPerPage = 1; // Número de PDFs por página

    var tipoPresentacion = $("#tipoPresentacion").val();

    var problema = $("#problema").val();
    console.log(problema)
    console.log(tipoPresentacion)
    var pdfPath = $('#rutaImagen').val();
    pdfPathIV = pdfPath + "/importanciaVariables.pdf";
    pdfPathLC = pdfPath + "/learningCurvePlot.pdf";

    pdfjsLib.getDocument(pdfPathIV).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.5;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-vi');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
    });

    pdfjsLib.getDocument(pdfPathLC).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-lc');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
    });

    const formData = new FormData();
    formData.append("rutaDirectorio", pdfPath);

    fetch("/obtenerGraficos",{
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                var dependenceFiles = data.dependencePlot;
                var iceFiles = data.ICEplot;

                $('#pagination-dependence').twbsPagination({
                    totalPages: Math.ceil(dependenceFiles.length / itemsPerPage),
                    visiblePages: 5,
                    onPageClick: function (event, page) {
                        event.preventDefault();
                        displayPDFs(dependenceFiles, 'pdf-container-dependence', page, pdfPath);
                    }
                });

                $('#pagination-ice').twbsPagination({
                    totalPages: Math.ceil(iceFiles.length / itemsPerPage),
                    visiblePages: 5,
                    onPageClick: function (event, page) {
                        event.preventDefault();
                        displayPDFs(iceFiles, 'pdf-container-ice', page, pdfPath);
                    }
                });
                console.log(tipoPresentacion)
                if (tipoPresentacion == 2 || tipoPresentacion == 4){
                    console.log("ENTREEEEE")
                     var NomogramFiles = data.NomogramPlot;
                     $('#pagination-nomogram').twbsPagination({
                        totalPages: Math.ceil(NomogramFiles.length / itemsPerPage),
                        visiblePages: 5,
                        onPageClick: function (event, page) {
                            event.preventDefault();
                            displayPDFs(NomogramFiles, 'pdf-container-nomogram', page, pdfPath);
                        }
                    });

                    displayPDFs(NomogramFiles, 'pdf-container-nomogram', 1, pdfPath)
                }


                // Cargar la primera página por defecto
                displayPDFs(dependenceFiles, 'pdf-container-dependence', 1, pdfPath);
                displayPDFs(iceFiles, 'pdf-container-ice', 1, pdfPath);

            })


    if (tipoPresentacion == 1){
        pdfPathMC = pdfPath + "/modelCorrelation.pdf";
        pdfPathShap= pdfPath + "/shapValues.pdf";
        pdfPathVIH = pdfPath + "/variableImportanceHeatmap.pdf";

        if(problema === "regresion"){
            pdfPathAr = pdfPath + "/graficaValoresResiduales.pdf";
        }
        else{
            pdfPathAr = pdfPath + "/matrizConfusion.pdf";
        }

        pdfjsLib.getDocument(pdfPathMC).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-mc-1');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
        });

         pdfjsLib.getDocument(pdfPathShap).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-shap-1');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
        });

         pdfjsLib.getDocument(pdfPathVIH).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-vih-1');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
        });

         pdfjsLib.getDocument(pdfPathAr).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-ar');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
        });
    }
    else if (tipoPresentacion == 2){
        pdfPathMC = pdfPath + "/modelCorrelation.pdf";
        pdfPathVIH = pdfPath + "/variableImportanceHeatmap.pdf";
        pdfPathAr = pdfPath + "/matrizConfusion.pdf";

        pdfjsLib.getDocument(pdfPathMC).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-mc-2');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
        });

        pdfjsLib.getDocument(pdfPathVIH).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-vih-2');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
        });

         pdfjsLib.getDocument(pdfPathAr).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-ar');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
        });
    }
    else if(tipoPresentacion == 3){
        pdfPathShap= pdfPath + "/shapValues.pdf";

        if(problema === "regresion"){
            pdfPathAr = pdfPath + "/graficaValoresResiduales.pdf";
        }
        else{
            pdfPathAr = pdfPath + "/matrizConfusion.pdf";
        }

         pdfjsLib.getDocument(pdfPathShap).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-shap-3');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
        });

        pdfjsLib.getDocument(pdfPathAr).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-ar');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
        });
    }
    else{
        pdfPathAr = pdfPath + "/matrizConfusion.pdf";

        pdfjsLib.getDocument(pdfPathAr).promise.then(function (pdf) {
        // Muestra la primera página (puedes modificar esto para mostrar todas las páginas o una específica)
        pdf.getPage(1).then(function (page) {
            var scale = 1.0 ;
            var viewport = page.getViewport({scale: scale});

            // Obtén el canvas donde se mostrará el PDF
            var canvas = document.getElementById('pdf-viewer-ar');
            var context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Renderiza el PDF
            var renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
        });
    }


function displayPDFs(pdfFiles, containerId, page, ruta) {
    var startIndex = (page - 1) * itemsPerPage;
    var endIndex = startIndex + itemsPerPage;
    $('#' + containerId).empty(); // Limpiar el contenedor de PDFs

    pdfFiles.slice(startIndex, endIndex).forEach((file, index) => {
        let pdfFullPath = ruta + '/' + file; // Asegúrate de que 'ruta' esté definida correctamente
        // Crear un nuevo canvas para este PDF
        var canvas = document.createElement('canvas');
        canvas.id = 'pdf-viewer-' + containerId + '-' + index;
        $('#' + containerId).append(canvas);

        pdfjsLib.getDocument(pdfFullPath).promise.then(function (pdf) {
            pdf.getPage(1).then(function (page) {
                var scale = 1.0;
                var viewport = page.getViewport({ scale: scale });

                var context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;

                var renderContext = {
                    canvasContext: context,
                    viewport: viewport
                };
                page.render(renderContext);
            });
        });
    });
}
});





function prepararBoton() {
    const container = document.querySelector('#tree-container-arbol2');
    var svg = container.querySelector('svg');
    // Asegúrate de que las dimensiones del SVG sean correctas
    var svgData = new XMLSerializer().serializeToString(svg);
    var svgBase64 = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));

    var img = new Image();
    img.src = svgBase64;
    img.onload = function() {
        // Crea un canvas con las dimensiones correctas
        var canvas = document.createElement('canvas');
        canvas.width = img.width; // Usa el ancho real de la imagen
        canvas.height = img.height; // Usa el alto real de la imagen
        var context = canvas.getContext('2d');

        // Dibuja la imagen SVG en el canvas sin ajustes negativos
        context.drawImage(img, 0, 0, img.width, img.height);

        // Convierte el canvas a Blob y crea el enlace de descarga
        canvas.toBlob(function(blob) {
            var url = URL.createObjectURL(blob);
            var downloadLink = document.querySelector('#descargarArbol');
            downloadLink.href = url;
            downloadLink.download = "Tree.png";
        });
    }
}

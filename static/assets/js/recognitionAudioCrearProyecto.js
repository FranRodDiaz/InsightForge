    $(document).ready(function() {
        var artyom = new Artyom();
        var isListening = false;
        var recognizedText = ""; // Almacena el texto final reconocido

        // Configuración inicial de Artyom
        artyom.initialize({
            lang: "es-ES",
            continuous: true,
            listen: false, // Comienza a escuchar cuando se active el dictado
            debug: true,
            speed: 1
        });

        // Configuración del Dictado de Artyom
        var dictado = artyom.newDictation({
            continuous: true,
            onResult: function(text, isFinal) {
                if(isFinal){
                    console.log("Es la ultima vez que recogo texto")
                }
                else{
                    recognizedText = text
                }
            },
            onStart: function() {
                recognizedText = ""; // Reinicia el texto reconocido al iniciar
            },
            onEnd: function() {
                console.log(recognizedText)
                enviarTextoAlServidor(recognizedText)
            }
        });

        // Función para alternar el reconocimiento de voz
        function toggleVoiceRecognition() {
            if (isListening) {
                dictado.stop(); // Detiene el dictado
                isListening = false;
                textoMicro = document.getElementById("textoMicro")
                textoMicro.classList.add("invisible")
            } else {
                dictado.start(); // Inicia el dictado
                isListening = true;
                textoMicro = document.getElementById("textoMicro")
                textoMicro.classList.remove("invisible")
            }
        }

        // Función para enviar el texto al servidor
        function enviarTextoAlServidor(texto) {
            // Obtiene el elemento <select>
            var selectElement = document.getElementById('selectListClase');

            // Crea un array para almacenar los valores de las opciones
            var selectValues = [];

            // Itera sobre las opciones del <select> para recoger sus valores
            for (var i = 0; i < selectElement.options.length; i++) {
                selectValues.push(selectElement.options[i].value);
            }

            // Ahora 'selectValues' contiene todos los valores de las opciones del <select>
            console.log(selectValues);

            if (texto) {
                $.ajax({
                    url: '/receive_speech_CrearProyecto',
                    type: 'POST',
                    data: { texto: texto , valoresColumnas: selectValues},
                    dataType: "json",
                    success: function(response) {

                        console.log("Respuesta recibida del servidor: ", response);

                        if (response.frase === null){
                            const myToast = new bootstrap.Toast(document.querySelector("#mensajeErrorFraseMicro"));

                            myToast.show();
                        }
                        else if(response.valores === null){
                            const myToast = new bootstrap.Toast(document.querySelector("#mensajeErrorDatosMicro"));

                            myToast.show();
                        }
                        else{

                            console.log(response['nombreProyecto'])
                                    // Verifica si la propiedad existe antes de intentar acceder a ella
                            console.log(Object.keys(response));
                            if (response["nombreProyecto"] !== undefined) {
                                $('#nombreProyecto').val(response["nombreProyecto"]);

                                let formData = new FormData();

                                formData.append("value", response["nombreProyecto"]);

                                action_id = $("#action_id").val();

                                if(action_id !== ""){
                                    formData.append("idAction", action_id);
                                }
                                else{
                                    formData.append("idAction", window.actionId);
                                }

                                formData.append("origen", "nombreProyecto");

                                fetch("/updateData", {
                                    method: "POST",
                                    body: formData
                                })
                                .then(response => response.json())
                                .then(data => {
                                    console.log(data)
                                    if(data.success !== true){
                                        const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorNombreRepetido'));
                                        myToast.show();
                                    }
                                });
                            }
                            if(response['variablesControl'] !== undefined){
                                console.log("Entreee");
                                var selectedValues = []
                                $('#checkboxListControl input[type="checkbox"]').each(function() {
                                    var nombreCheckbox = $(this).attr('value'); // O 'id', dependiendo de qué atributo estés usando para comparar.
                                    console.log(nombreCheckbox)
                                    // Verificar si el nombreCheckbox está en la listaDeVariables
                                    if(response['variablesControl'].includes(nombreCheckbox)) {
                                        console.log("Coincide y se marcará: " + nombreCheckbox);
                                        // Marca el checkbox
                                        $(this).prop('checked', true);
                                        selectedValues.push(nombreCheckbox)
                                    }
                                });

                                let formData = new FormData();

                                formData.append("value", selectedValues);

                                action_id = $("#action_id").val();

                                if(action_id !== ""){
                                    formData.append("idAction", action_id);
                                }
                                else{
                                    formData.append("idAction", window.actionId);
                                }

                                formData.append("origen", "control");

                                fetch("/updateDataList", {
                                    method: "POST",
                                    body: formData
                                })
                                .then(response => response.json())
                                .then(data => {
                                    if(data.success){
                                        let selectedCount = $("#checkboxListControl input[type='checkbox']:checked").length;
                                        let totalCount = $("#checkboxListControl input[type='checkbox']").length; // Cuenta todas las casillas de verificación
                                        $("#selectedCountControl").text(`${selectedCount} / ${totalCount} seleccionadas`);
                                    }
                                });
                            }
                            if(response['variablesExternas'] !== undefined){
                                console.log("Entreee");
                                var selectedValues = []
                                $('#checkboxListExterno input[type="checkbox"]').each(function() {
                                    var nombreCheckbox = $(this).attr('value'); // O 'id', dependiendo de qué atributo estés usando para comparar.
                                    console.log(nombreCheckbox)
                                    // Verificar si el nombreCheckbox está en la listaDeVariables
                                    if(response['variablesExternas'].includes(nombreCheckbox)) {
                                        console.log("Coincide y se marcará: " + nombreCheckbox);
                                        // Marca el checkbox
                                        $(this).prop('checked', true);
                                        selectedValues.push(nombreCheckbox)
                                    }
                                });

                                let formData = new FormData();

                                formData.append("value", selectedValues);

                                action_id = $("#action_id").val();

                                if(action_id !== ""){
                                    formData.append("idAction", action_id);
                                }
                                else{
                                    formData.append("idAction", window.actionId);
                                }

                                formData.append("origen", "externo");

                                fetch("/updateDataList", {
                                    method: "POST",
                                    body: formData
                                })
                                .then(response => response.json())
                                .then(data => {
                                    if(data.success){
                                        let selectedCount = $("#checkboxListExterno input[type='checkbox']:checked").length;
                                        let totalCount = $("#checkboxListExterno input[type='checkbox']").length; // Cuenta todas las casillas de verificación
                                        $("#selectedCountExtern").text(`${selectedCount} / ${totalCount} seleccionadas`);
                                    }
                                });
                            }
                        }
                    },
                     error: function(error) {
                        console.log("Error al enviar texto al servidor.");
                        // Manejar el error adecuadamente
                     }
                });
            } else {
                console.log("No hay texto para enviar.");
            }
        }

        // Evento del botón para activar/desactivar el reconocimiento de voz
        document.getElementById('start-stop-btn').addEventListener('click', toggleVoiceRecognition);
    });

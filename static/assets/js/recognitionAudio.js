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
                console.log("Texto reconocido (provisional/final): ", text);
            },
            onStart: function() {
                console.log("Dictado iniciado");
                recognizedText = ""; // Reinicia el texto reconocido al iniciar
            },
            onEnd: function() {
                console.log("Dictado detenido");
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
            if (texto) {
                $.ajax({
                    url: '/receive_speech',
                    type: 'POST',
                    data: { texto: texto },
                    success: function(response) {
                        console.log("Texto enviado al servidor: ", texto);
                        console.log(response)
                          if (response.frase === null) {
                            const myToast = new bootstrap.Toast(document.querySelector("#mensajeErrorFraseMicro"));

                            myToast.show();
                          }
                          else if(response.accion === null){
                            const myToast = new bootstrap.Toast(document.querySelector("#mensajeErrorAccionMicro"));

                            myToast.show();
                          }
                          else if(response.identificador === null){
                            const myToast = new bootstrap.Toast(document.querySelector("#mensajeErrorIdentificador"));

                            myToast.show();
                          }
                          else if(response.borrar){
                              $('#modalParaEliminar').modal('show');
                          }
                          else if(response.datos === null){
                              projectId = response.idProject;
                              $('#AceptaModificar').data('project-id', projectId);
                              $('#modalParaError').modal('show');
                          }
                          else if (response.redirect) {
                              // El servidor indica redirección
                              window.location.href = response.redirect; // Redirige

                              setTimeout(function() {
                                  window.location.reload();
                              }, 2000); // 1000 milisegundos equivalen a 1 segundos
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


$(document).ready(function () {
    $(document).on('click', "#AceptaEliminar", function (){
        window.location.href = `/deleteProjects`
    });
});

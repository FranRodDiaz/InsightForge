$(window).on("load", function () {

    console.log(localStorage)
    let proyectosCargando = JSON.parse(localStorage.getItem('proyectosCargando') || "[]");
    proyectosCargando.forEach(proyecto => {
        var newRow = `
            <div class="row">
                <div class="col">
                    <h5>${proyecto.nombreProyecto}</h5>
                </div>
                <div class="col">
                    <div class="progress" id="barraCarga-${proyecto.projectId}">
                        <div class="progress-bar" role="progressbar" aria-valuenow="${proyecto.carga}" aria-valuemin="0" aria-valuemax="100">${proyecto.carga}%</div>
                    </div>
                </div>
            </div>
        `;
        $('#mostrarBarras').prepend(newRow);
    });

    $(document).on('click', '.botonStart', function() {


        // Tu código a ejecutar cuando se hace clic en un botón con la clase "botonStart"
         var $row = $(this).closest('tr');

        // Dentro de esa fila, encuentra el input con la clase 'projectId'
        var projectId = $row.find('.projectId').val();

        $(this).prop('disabled', true);

        var projectName = $(this).closest('tr').find('td').eq(1).text();

        const formData1 = new FormData();
        formData1.append("idProject", projectId);
        formData1.append('nombreProyecto', projectName)

        console.log(formData1)

        fetch("/comprobarDatos", {
            method: "POST",
            body: formData1
        })
        .then(response => {
            if (response.ok) {
                console.log("BIEN");
                var newRow = `
                <div class="row">
                    <div class="col">
                        <h5>${projectName}</h5>
                    </div>
                    <div class="col">
                        <div class="progress" id="barraCarga-${projectId}">
                            <div class="progress-bar" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                        </div>
                    </div>
                </div>`;

                $('#mostrarBarras').append(newRow);



                //localStorage.setItem('cargaEnProgreso', 'true');
                //localStorage.setItem('projectIdEnProgreso', projectId);

                const formData = new FormData();
                formData.append("idProject", projectId);
                formData.append('nombreProyecto', projectName)

                console.log(formData)

                fetch("/inicioProceso", {
                    method: "POST",
                    body: formData
                })
                .then(response => {
                    // Clonar la respuesta para poder leer el stream de respuesta más de una vez si es necesario.
                    const responseClone = response.clone();
                    // Primero, verifica el estado de la respuesta
                    if (response.status === 204) {
                        localStorage.removeItem('disabled-' + projectId);
                    } else if(response.status === 500) {
                        // Si hay un error, transforma la respuesta en JSON para leer los detalles del error
                        return responseClone.json().then(data => {
                            // Aquí ya puedes acceder a data.error
                            if (data.error === 'id_project not found') {
                                // Acciones específicas para este error
                                const myToastErrorIdentificador = new bootstrap.Toast(document.querySelector("#mensajeErrorIdentificador2"));
                                myToastErrorIdentificador.show();
                                localStorage.removeItem('proyectosCargando');
                                let progressBarId = '#barraCarga-' + projectId;
                                $(progressBarId).closest('.row').remove();
                                $(this).prop('disabled', false);
                            } else {
                                // Acciones para otros tipos de error 500
                                const myToastErrorProyecto = new bootstrap.Toast(document.querySelector("#mensajeErrorProyecto"));
                                myToastErrorProyecto.show();
                                socket.emit('detenerConsultas', {projectId: projectId});
                                localStorage.removeItem('proyectosCargando');
                                let progressBarId = '#barraCarga-' + projectId;
                                $(progressBarId).closest('.row').remove();
                                $(this).prop('disabled', false);
                            }
                        });
                    } else {
                    // Manejar otros códigos de estado
                    throw new Error('Respuesta no es OK');
                    }
                })
            }
            else{
                console.log("MAL");
                $('#AceptaModificar').data('project-id', projectId);
                $('#modalParaError').modal('show');
                $(this).prop('disabled', false);
            }
        });
    });
        const valorId = $("#valor").val()

        const socket = io.connect('http://localhost:3001');
            socket.on('connect', function() {
            socket.emit('user_connected', valorId);
        });

        socket.on('actualizarEstado', (data) => {
           const estado = data.estado;
           const id_project = data.id_project;
           console.log("SE HA RECIBIDO EL ID PROJECTO ", id_project)
           document.getElementById('estado-' +  id_project).textContent = estado;
        });


        socket.on('cargaProyecto', (cargaPorcentaje) => {
            let carga = cargaPorcentaje.porcentaje

            const projectId = cargaPorcentaje.projectId

            const nombreProyecto = cargaPorcentaje.nombreProyecto

            const nameUser = cargaPorcentaje.nameUser


            let proyectosCargando = JSON.parse(localStorage.getItem('proyectosCargando') || "[]");

            // Verifica si el proyecto con el mismo nombre ya existe en el array
            const proyectoYaExiste = proyectosCargando.some(proyecto => proyecto.nombreProyecto === nombreProyecto);

            if (!proyectoYaExiste) {
                // Si el proyecto no existe, lo añade a proyectosCargando
                proyectosCargando.push({ projectId, nombreProyecto, carga });
                localStorage.setItem('proyectosCargando', JSON.stringify(proyectosCargando));
            }

            let progressBarId = '#barraCarga-' + projectId + ' .progress-bar';

            $(progressBarId).css('width', carga + '%');

            $(progressBarId).text(carga + '%');
            console.log(carga)
            if (carga === 100) {
                $(progressBarId).closest('.row').remove();
                $('.btn-to-enable-' + projectId).prop('disabled', false);
                var downloadButton = $(".btn-to-download-" + projectId);
                //downloadButton.attr('data-download-url', `./static/dirs/${nameUser}/${nombreProyecto}.zip`);
                let proyectosCargando = JSON.parse(localStorage.getItem('proyectosCargando') || "[]");
                proyectosCargando = proyectosCargando.filter(proyecto => proyecto.projectId !== projectId);
                localStorage.setItem('proyectosCargando', JSON.stringify(proyectosCargando));
            }
        });

});



$(document).ready(function () {
    $(document).on('click', "#AceptaModificar", function (){
        var projectId = $(this).data('project-id');
        window.location.href = `/ModificarDatos?projectId=${projectId}`
    });
});

$(document).ready(function() {

    function filterList(inputId, listId) {
        const searchValue = document.getElementById(inputId).value.toLowerCase();
        const checkboxList = document.getElementById(listId);
        const items = checkboxList.getElementsByTagName("li");

        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            const label = item.textContent || item.innerText;

            if (label.toLowerCase().indexOf(searchValue) > -1) {
            item.style.display = "";
          } else {
            item.style.display = "none";
          }
        }
      }

    document.getElementById("searchInput1").addEventListener("input", function() {
        filterList("searchInput1", "checkboxListControl");
    });
      
    document.getElementById("searchInput2").addEventListener("input", function() {
        filterList("searchInput2", "checkboxListExterno");
    });



});


$(document).ready(function(){

     new TomSelect("#selectDataset")


    document.getElementById("file-upload").addEventListener("change", function() {
        if (this.files.length > 0) {
            processFile(this.files[0]);
        }
    });

    function processFile(file) {
        let formData = new FormData();
        formData.append("uploaded_file", file);
        let nombreArchivo = file.name;
        $("#file-count").text(nombreArchivo)
        fetch("/process_file", {     //CAMBIOS AQUIIIIIIII
            method: "POST",
            body: formData
        })
        .then(response =>{
            if(!response.ok){
                response.json().then(data => {
                    $("#file-count").text("Ningún archivo seleccionado");

                    // Mostrar mensajes basados en el código de estado HTTP
                    switch (response.status) {
                        case 413:
                            var myToast = new bootstrap.Toast(document.querySelector("#mensajeErrorTamanioArchivo"));
                            myToast.show();
                            break;
                        case 415:
                            var myToast = new bootstrap.Toast(document.querySelector("#mensajeErrorExtensionArchivo"));
                            myToast.show();
                            break;
                        default:
                            alert("Error en el servidor: " + data.error);  // Otro tipo de error
                    }
                });
                throw new Error('Hubo un error en el servidor');
            }
            return response.json();
        })
        .then(data => {
            console.log(data);

            $("#botonEnvio").prop("disabled", false);
            $("#nombreProyecto").prop("disabled", false);
            $('#porcentajeTrain').prop('disabled', false);
            $('#kFolds').prop('disabled', true);
            $("#mensajeModelos").removeClass("d-none");

            updateCheckboxesControl(Object.keys(data.columns));

            updateCheckboxesExternos(Object.keys(data.columns));

            updateCheckboxesClase(Object.keys(data.columns));

            new TomSelect("#selectListClase")
            new TomSelect("#selectTipoProblema")


            $("#action_id").val(data.actionId);
            sessionStorage.setItem('columnDataTypes', JSON.stringify(data.columns));
            window.actionId = data.actionId;

            if ($('#holdOutRadio').is(':checked')) { // Si el radiobutton de "Hold-out" está seleccionado
                $('#porcentajeTrain').prop('disabled', false); // Habilita el campo correspondiente
                $('#kFolds').prop('disabled', true); // Deshabilita el campo de "Cross Validation"
                $("#validationInfo").text("Hold-out 70% training");
            } else {
                $('#porcentajeTrain').prop('disabled', true); // Deshabilita el campo de "Hold-out"
                $('#kFolds').prop('disabled', false); // Habilita el campo correspondiente
                $("#validationInfo").text("Cross Validation 10 Folds");
            }
        })
        .catch(error => {
            console.error("Error:", error);
        });
    }

    function updateCheckboxesControl(columns) {
        const checkboxList = document.getElementById("checkboxListControl");
        checkboxList.innerHTML = ""; // Limpiar la lista actual
    
        columns.forEach(column => {
            let listItem = document.createElement("li");
            listItem.className = "list-group-item";
    
            let checkboxDiv = document.createElement("div");
            checkboxDiv.className = "form-check";
    
            let checkbox = document.createElement("input");
            checkbox.className = "form-check-input";
            checkbox.type = "checkbox";
            checkbox.name = "controlColumns";
            checkbox.value = column;
    
            let label = document.createElement("label");
            label.className = "form-check-label";
            label.innerText = column;
    
            checkboxDiv.appendChild(checkbox);
            checkboxDiv.appendChild(label);
            listItem.appendChild(checkboxDiv);
            checkboxList.appendChild(listItem);
        });
        let totalCount = $("#checkboxListControl input[type='checkbox']").length; // Cuenta todas las casillas de verificación
        $("#selectedCountControl").text(`0 / ${totalCount} seleccionadas`);
    }

    function updateCheckboxesExternos(columns) {
        const checkboxList = document.getElementById("checkboxListExterno");
        checkboxList.innerHTML = ""; // Limpiar la lista actual
        
        columns.forEach(column => {
            let listItem = document.createElement("li");
            listItem.className = "list-group-item";
        
            let checkboxDiv = document.createElement("div");
            checkboxDiv.className = "form-check";
        
            let checkbox = document.createElement("input");
            checkbox.className = "form-check-input";
            checkbox.type = "checkbox";
            checkbox.name = "externalColumns";
            checkbox.value = column;
        
            let label = document.createElement("label");
            label.className = "form-check-label";
            label.innerText = column;
        
            checkboxDiv.appendChild(checkbox);
            checkboxDiv.appendChild(label);
            listItem.appendChild(checkboxDiv);
            checkboxList.appendChild(listItem);
        });

        let totalCount = $("#checkboxListExterno input[type='checkbox']").length; // Cuenta todas las casillas de verificación
        $("#selectedCountExtern").text(`0 / ${totalCount} seleccionadas`);
    }
            

        function updateCheckboxesClase(columns) {
        const selectList = document.getElementById("selectListClase");  // Agregar esta línea para obtener el select.
        
        while (selectList.options.length > 1) {
            selectList.remove(1);
        }
        
        columns.forEach(column => {
            let option = new Option(column, column);
            selectList.add(option);
        });
        // Habilitar campos desactivados
        document.querySelectorAll("[disabled]:not(#selectPositiveClase)").forEach(el => {
            el.removeAttribute("disabled");
        });

        document.querySelectorAll("input[type='radio'][name='modelos']").forEach(radioButton => {
            radioButton.setAttribute("disabled", "true");
        });

    }



    const modelosRegresion = ["RLIN", "RN", "RF", "AML", "GBM"];
    const modelosClasificacion = ["RN", "RLOG", "RF", "NB", "AML", "GBM"];

    // Función para actualizar la visibilidad de los modelos
    function updateModelVisibility(tipo) {
        // Obtener todos los elementos li de la lista de modelos
            const divItems = document.querySelectorAll(".algoritmos");
                divItems.forEach(div => {
        const input = div.querySelector("input");
        if (input) {
            const value = input.value;

            // Inicialmente, ocultar y deshabilitar todos
            div.style.display = "none";
            input.disabled = true;

            if (tipo === "regresion" && modelosRegresion.includes(value)) {
                div.style.display = ""; // Mostrar
                input.disabled = false;
            } else if (tipo === "clasificacion" && modelosClasificacion.includes(value)) {
                div.style.display = ""; // Mostrar
                input.disabled = false;
            } else if (tipo === "ns") {
                div.style.display = ""; // Mostrar todos
                input.disabled = true;  // Pero deshabilitar todos
            }
        }
    });
    }
// Escuchar el evento change en el select de tipoProblema
$("select[name='tipoProblema']").on("change", function() {
    const selectedValue = $(this).val();

    updateModelVisibility(selectedValue);

    action_id = $("#action_id").val()   

    let formData = new FormData();

    formData.append("value", selectedValue);

    if(action_id !== ""){
        formData.append("idAction", action_id);
    }
    else{
        formData.append("idAction", window.actionId);
    }
    
    if(selectedValue !== "ns"){
        $("#problemaVacio").addClass("d-none");
    }

    formData.append("origen", "tipoProblema");


    fetch("/updateData", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())  
    .then(data => {
        if(data.success){
            console.log("Actualización realizada");
        }
    });

    console.log(selectedValue)
    if(selectedValue === "clasificacion"){
        console.log("Estoy con clasificacion");
        let formData = new FormData();

        if(action_id !== ""){
            formData.append("idAction", action_id);
        }
        else{
            formData.append("idAction", window.actionId);
        }

        const columnaSeleccionada = $("#selectListClase").val()

        const nombreArchivo = $("#file-count").text()

        formData.append("columnaClase", columnaSeleccionada)

        formData.append("nombreArchivo", nombreArchivo)

        fetch("/clasePositiva", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {

            var tom = $("#selectPositiveClase")[0].tomselect;
            if (tom) {
                tom.clearOptions();
                tom.refreshOptions(true);
                tom.destroy();
            }

            $.each(data.valores, function(index, valor) {
                $('#selectPositiveClase').append($('<option>', {
                    value: valor,
                    text: valor
                }));
            });

            $("#selectPositiveClase").prop("disabled", false);
            new TomSelect("#selectPositiveClase")
        });
    }
    else{
        var selectElement = $("#selectPositiveClase");

        var tom = $("#selectPositiveClase")[0].tomselect;
        if (tom) {
            tom.clearOptions();
            tom.refreshOptions(true);
            tom.destroy();
        }

        selectElement.empty();

        selectElement.append($('<option>', {
            value: "ns",
            text: ""
        }));

        $("#selectPositiveClase").prop("disabled", true);
    }
});

    $("select[name='clasePositiva']").on("change", function() {
        const selectedValue = $(this).val();

         action_id = $("#action_id").val()

        let formData = new FormData();

        formData.append("value", selectedValue);

        if(action_id !== ""){
            formData.append("idAction", action_id);
        }
        else{
            formData.append("idAction", window.actionId);
        }

        formData.append("origen", "clasePositiva");

        fetch("/updateData", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if(data.success){
                console.log("Actualización realizada");
            }
        });
    })


    $("select[name='columnaClase']").on("change", function(){
         // Obtiene el nombre de la columna seleccionada
        let selectedColumn = this.value;
        const estaEnControl = $(`#checkboxListControlcheckboxListControl input[value="${selectedColumn}"]:checked`).length > 0;
        const estaEnExterno = $(`#checkboxListExterno input[value="${selectedColumn}"]:checked`).length > 0;

        if (estaEnControl) {
            const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorNombreExternaControl'));
            myToast.show();
            let tsInstance = $("#selectListClase")[0].tomselect;
            tsInstance.setValue("ns")
            let tsInstance2 = $("#selectTipoProblema")[0].tomselect;
            tsInstance2.setValue("ns")
        }
        else if(estaEnExterno){
            const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorNombreControlExterna'));
            myToast.show();
            let tsInstance = $("#selectListClase")[0].tomselect;
            tsInstance.setValue("ns")
            let tsInstance2 = $("#selectTipoProblema")[0].tomselect;
            tsInstance2.setValue("ns")
        }
        else{

            action_id = $("#action_id").val()

            let formData = new FormData();

            formData.append("value", selectedColumn);

            if(action_id !== ""){
                formData.append("idAction", action_id);
            }
            else{
                formData.append("idAction", window.actionId);
            }

            if(selectedColumn !== "ns"){
                $("#claseVacia").addClass("d-none");
            }
            console.log(action_id)
            formData.append("origen", "clase");
            
            fetch("/updateData", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())  
            .then(data => {
                if(data.success){
                    console.log("Actualización realizada");
                    $("#mensajeModelos").addClass("d-none");
                }
            });

            // Encuentra el tipo de datos de la columna seleccionada
            const columnDataTypes = JSON.parse(sessionStorage.getItem('columnDataTypes'));
            console.log(columnDataTypes)
            console.log(selectedColumn)
            let columnType = columnDataTypes[selectedColumn];
            // Selecciona el combo de tipo de problema
            let problemTypeSelect = document.querySelector("select[name='tipoProblema']");
            let tsInstance = $("#selectTipoProblema")[0].tomselect;
            // Establece el valor del combo en función del tipo de datos
            if (columnType === "float64") {
                tsInstance.setValue("regresion");
                tsInstance.disable()
                $("input[name='modelos'][value='RF']").prop("checked", true);
                $("input[name='modelos']").prop("disabled", false);
            } else if (columnType === "object") {
                tsInstance.setValue("clasificacion");
                tsInstance.disable()
                $("input[name='modelos'][value='RF']").prop("checked", true);
                $("input[name='modelos']").prop("disabled", false);
            } else if(columnType === "int64"){
                tsInstance.setValue("ns");
                $("input[name='modelos'][value='RF']").prop("checked", false);
                $("input[name='modelos']").prop("disabled", true);
            } else {
                tsInstance.setValue("ns");
            }
        }
    });

    document.querySelector("input[name='nameProject']").addEventListener("blur", function() {
        console.log("entre")
        const projectNameValue = this.value;
        console.log(projectNameValue)
        let formData = new FormData();

        formData.append("value", projectNameValue);

        action_id = $("#action_id").val();
        console.log(action_id)
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
    });

    // Obtener todos los radio buttons con el nombre "interestColumns"
    const radioButtons = document.querySelectorAll("input[type='radio'][name='modelos']");

    // Añadir un evento a cada radio button
    radioButtons.forEach(radioButton => {
        radioButton.addEventListener("change", function() {
            if (this.checked) {
                // Obtener el valor del radio button seleccionado
                const selectedModel = this.value;
               
                let formData = new FormData();

                formData.append("value", selectedModel);

                action_id = $("#action_id").val();

                if(action_id !== ""){
                    formData.append("idAction", action_id);
                }
                else{
                    formData.append("idAction", window.actionId);
                }
                
                formData.append("origen", "modelo");

                fetch("/updateData", {
                    method: "POST",
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if(data.success){
                        console.log("Actualización realizada");
                    }
                });
            }
        }); 
    });

    $('#checkboxListExterno').on('change', 'input[type="checkbox"]', function() {

        const seleccionado = $(this).val();
        const columnaClase = $('#selectListClase').val();
        const estaEnLaOtraLista = $(`#checkboxListControl input[value="${seleccionado}"]:checked`).length > 0;

        if (seleccionado === columnaClase) {
            $("#errorExterno").show()
            const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorControlClase'));
            myToast.show();
            $(this).prop('checked', false);
        }
        else if(estaEnLaOtraLista){
            $("#errorExterno").show()
            const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorNombreExternaControl'));
            myToast.show();
            $(this).prop('checked', false);
        }
        else{
            $("#errorExterno").hide()
            const selectedCheckboxes = $('#checkboxListExterno input[type="checkbox"]:checked');

            // Si solo deseas los valores de los checkboxes seleccionados en un array
            const selectedValues = [];
            selectedCheckboxes.each(function() {
                selectedValues.push($(this).val());
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
    });

    $('#checkboxListControl').on('change', 'input[type="checkbox"]', function() {
        const seleccionado = $(this).val();
        const columnaClase = $('#selectListClase').val();
        const estaEnLaOtraLista = $(`#checkboxListExterno input[value="${seleccionado}"]:checked`).length > 0;

        if (seleccionado === columnaClase) {
            $("#errorControl").show()
            const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorControlClase'));
            myToast.show();
            $(this).prop('checked', false);
        }
        else if(estaEnLaOtraLista){
            $("#errorControl").show()
            const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorNombreControlExterna'));
            myToast.show();
            $(this).prop('checked', false);
        }
        else{
            $("#errorControl").hide()
            const selectedCheckboxes = $('#checkboxListControl input[type="checkbox"]:checked');


            const selectedValues = [];
            selectedCheckboxes.each(function() {
                selectedValues.push($(this).val());
            });

            if(selectedValues.length !== 0){
                $("#controlVacio").addClass("d-none");
            }
    
           
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
    });

    $("#selectDataset").on("change", function(){
        const selectDataset = $(this).val();

        let formData = new FormData()

        formData.append("idProyecto", selectDataset)

        fetch("/fetchDataset",{
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            $("#nombreProyecto").prop("disabled", "false");
            console.log(data);
            updateCheckboxesControl(Object.keys(data.columns));
            updateCheckboxesExternos(Object.keys(data.columns));
            updateCheckboxesClase(Object.keys(data.columns));      
            sessionStorage.setItem('columnDataTypes', JSON.stringify(data.columns));
            window.actionId = data.actionId;
            $("#action_id").val(data.actionId);
            $("#mensajeModelos").removeClass("d-none");

            new TomSelect("#selectListClase")
            new TomSelect("#selectTipoProblema")
            new TomSelect("#selectPositiveClase")

            if ($('#holdOutRadio').is(':checked')) { // Si el radiobutton de "Hold-out" está seleccionado
                $('#porcentajeTrain').prop('disabled', false); // Habilita el campo correspondiente
                $('#kFolds').prop('disabled', true); // Deshabilita el campo de "Cross Validation"
                $("#validationInfo").text("Hold-out 70% training");
            } else {
                $('#porcentajeTrain').prop('disabled', true); // Deshabilita el campo de "Hold-out"
                $('#kFolds').prop('disabled', false); // Habilita el campo correspondiente
                $("#validationInfo").text("Cross Validation 10 Folds");
            }
        })
    });

    $("#botonActualizar").click(function (){
        window.location.href = `/goMisProyectos`;
    })

    $("#botonEnvio").click(function(){
        const nombreProyecto = $("#nombreProyecto").val();
        const nombreDataset = $("#file-count").text();
        const columnasControl = $('#checkboxListControl input[type="checkbox"]:checked');
        const selectedValuesControl = [];

        columnasControl.each(function() {
            selectedValuesControl.push($(this).val());
        });

        const columnasExternas = $('#checkboxListExterno input[type="checkbox"]:checked');

        const selectedValuesExternal = [];
        columnasExternas.each(function() {
            selectedValuesExternal.push($(this).val());
        });

        const modelo = $('input[type=radio][name=modelos]:checked').val();
        const columnaClase = $("select[name='columnaClase']").val();
        const tipoProblema = $("select[name='tipoProblema']").val();
        const clasePositiva = $("select[name='clasePositiva']").val();

        let validacion;
        let valor;
        if ($('#holdOutRadio').is(':checked')) { // Si el radiobutton de "Hold-out" está seleccionado
            validacion = "HO";
            valor = $("#porcentajeTrain").val();
        } else {
            validacion = "CV";
            valor = $("#kFolds").val();
        }

        const error = [false, false, false, false, false];

        if(nombreProyecto === ""){
            const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorNombreVacio'));
            myToast.show();
            error[0] = true;
        }

        if(selectedValuesControl.length === 0){
            const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorControl'));
            myToast.show();
            error[1] = true;
        }

        if(columnaClase === "ns"){
            const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorClase'));
            myToast.show();
            error[2] = true;
        }

        if(tipoProblema === "ns"){
            const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorTipoProblema'));
            myToast.show();
            error[3] = true;
        }
        else{
            if(tipoProblema === "clasificacion"){
                if(clasePositiva === "ns"){
                    const myToast = new bootstrap.Toast(document.querySelector('#mensajeErrorClasePositiva'));
                    myToast.show();
                    error[4] = true;
                }
            }
        }

        const action_id = $("#action_id").val();

        let tieneTrue = error.some(valor => valor);

        if(!tieneTrue){
            sessionStorage.setItem('botonPresionado', 'true');
            sessionStorage.setItem('carga', 'true');
            sessionStorage.setItem('nombre', nombreProyecto);

            window.location.href = `/goMisProyectos`

        }
    });
    
    
    $('#holdOutRadio').change(function(){
        if($(this).is(':checked')){
            action_id = $("#action_id").val()

            let formData = new FormData();
    
            if(action_id !== ""){
                formData.append("idAction", action_id);
            }
            else{
                formData.append("idAction", window.actionId);
            }

            const porcentaje = $("#porcentajeTrain").val();

            formData.append("value", `HO-${porcentaje}`);
    
            formData.append("origen", "validacion");
    
            fetch("/updateData", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if(data.success){
                    $('#porcentajeTrain').prop('disabled', false);
                    $('#kFolds').prop('disabled', true);                 
                    $("#validationInfo").text(`Hold out - ${porcentaje}% training`);
                }
            });

        }
    });
    

    $('#crossValidationRadio').change(function(){
        if($(this).is(':checked')){
            action_id = $("#action_id").val()

            let formData = new FormData();

            if(action_id !== ""){
                formData.append("idAction", action_id);
            }
            else{
                formData.append("idAction", window.actionId);
            }

            const kfolds = $("#kFolds").val();

            formData.append("value", `CV-${kfolds}`);

            formData.append("origen", "validacion");

            fetch("/updateData", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if(data.success){
                    $('#kFolds').prop('disabled', false);
                    $('#porcentajeTrain').prop('disabled', true);                  
                    $("#validationInfo").text(`Cross Validation - ${kfolds} Folds`);
                }
            });
        }
    });

    $("#porcentajeTrain").change(function(){
        action_id = $("#action_id").val()

        let formData = new FormData();

        if(action_id !== ""){
            formData.append("idAction", action_id);
        }
        else{
            formData.append("idAction", window.actionId);
        }

        const porcentaje = $(this).val();

        formData.append("value", `HO-${porcentaje}`);

        formData.append("origen", "validacion");

        fetch("/updateData", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if(data.success){
                $("#validationInfo").text(`Hold out - ${porcentaje}% training`);
            }
        });
        
        
    });

    $("#kFolds").change(function(){
        let action_id = $("#action_id").val()

        let formData = new FormData();

        if(action_id !== ""){
            formData.append("idAction", action_id);
        }
        else{
            formData.append("idAction", window.actionId);
        }

        const kfolds = $(this).val();

        formData.append("value", `CV-${kfolds}`);

        formData.append("origen", "validacion");

        fetch("/updateData", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if(data.success){
                $("#validationInfo").text(`Cross Validation - ${kfolds} Folds`);
            }
        });    
    });

    window.onbeforeunload = function() {
       if (sessionStorage.getItem('botonPresionado') === 'true') {

            sessionStorage.setItem('botonPresionado', 'false');
        return;
    }

        // Aquí va tu código existente para enviar la solicitud fetch
        let action_id = $("#action_id").val();
        let formData = new FormData();
        formData.append('action_id', action_id);

        fetch('/eliminar_carpeta', {
            method: 'POST',
            body: formData,
            keepalive: true
        }).then(response => {
            if (response.status === 204) {
                console.log("Eliminado con éxito");
            }
        });
    }
});

    function presentarDatos(controlVar, problemType, file_name, model, column_types, datasets, positiveClass, nombreProyecto, valores){
        console.log("He recibido todos los datos");
        $("#nombreProyecto").val(nombreProyecto);
        $("#file-count").text(file_name);

        console.log(column_types)

        const selectList = document.getElementById("selectListClase");  // Agregar esta línea para obtener el select.

        while (selectList.options.length > 1) {
            selectList.remove(1);
        }
         sessionStorage.setItem('columnDataTypes', JSON.stringify(column_types));

        columns = Object.keys(column_types)

        columns.forEach(column => {
            let option = new Option(column, column);
            if (column === controlVar) {
                option.selected = true;
            }
            selectList.add(option);
        });
        // Habilitar campos desactivados
        document.querySelectorAll("[disabled]:not(#selectPositiveClase)").forEach(el => {
            el.removeAttribute("disabled");
        });


        new TomSelect("#selectTipoProblema");
        new TomSelect("#selectListClase");

        if(problemType === "regresion"){
            const selectListPositiveClass = document.getElementById("selectPositiveClase");
            selectListPositiveClass.disabled = true;
        }
        else{
            const selectListPositiveClass = document.getElementById("selectPositiveClase");
            while (selectListPositiveClass.options.length > 1) {
                selectListPositiveClass.remove(1);
            }

            valores.forEach(valor => {
                let option = new Option(valor, valor);
                if (valor === positiveClass) {
                    option.selected = true;
                }
                selectListPositiveClass.add(option);
            });

            new TomSelect("#selectPositiveClase");
        }

        const modelosRegresion = ["RLIN", "RN", "SVM", "RF", "AML", "GBM"];
        const modelosClasificacion = ["SVM", "RN", "RLOG", "RF", "NB", "AML", "GBM"];

        // Obtener todos los elementos li de la lista de modelos
        const divItems = document.querySelectorAll(".algoritmos");
        divItems.forEach(div => {
            const input = div.querySelector("input");
            if (input) {
                const value = input.value;

            // Inicialmente, ocultar y deshabilitar todos
                div.style.display = "none";
                input.disabled = true;

                if (problemType === "regresion" && modelosRegresion.includes(value)) {
                    div.style.display = ""; // Mostrar
                    input.disabled = false;
                    if(value === model){
                        input.checked = true; // Marcar el radio input
                    }
                } else if (problemType === "clasificacion" && modelosClasificacion.includes(value)) {
                    div.style.display = ""; // Mostrar
                    input.disabled = false;
                    if(value === model){
                        input.checked = true; // Marcar el radio input
                    }
                } else if (problemType === "ns") {
                    div.style.display = ""; // Mostrar todos
                    input.disabled = true;  // Pero deshabilitar todos
                }
            }
        });
    }

function updateButtonLabels() {
  if ($(window).width() < 1200) {
    $('#btnVariablesControl').text('V. Control');
    $('#btnValidacion').text('Validación');
    $('#btnVariablesExternas').text('V. Externas');
  } else {
    // Texto para pantallas más grandes
    $('#btnVariablesControl').text('Variables control');
    $('#btnValidacion').text('Validación');
    $('#btnVariablesExternas').text('Variables externas');
  }
}

// Ejecutar al cargar la página
updateButtonLabels();

// Ejecutar cada vez que se cambia el tamaño de la ventana
$(window).resize(updateButtonLabels);
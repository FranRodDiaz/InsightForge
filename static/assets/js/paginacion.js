var historyData;    //Global variable to save the records

//Function to format the date "yyyy-MM-dd"
function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0'); 
    const day = String(date.getDate()).padStart(2, '0'); 
  
    return `${year}-${month}-${day}`;
}

//Function to search a record by its name
function searchTable() {
    var searchText = document.getElementById("search-input").value.toLowerCase();
    var filteredData = [];

    for (var i = 0; i < historyData.length; i++) {
        if (historyData[i][1].toLowerCase().includes(searchText)) {
          filteredData.push(historyData[i]);
        }
      }
    //Update the pagination with the filtered data
    paginar(filteredData, true);
}

//Function to paginate the records
function paginar(data, search, username){

    if(!search){            //Check if we returned from the 'searchTable' function
        historyData = data;
    }

    recPerPage = 7,     //Define the records per page
    page = 1,           //Define the number of page where it started
    totalRecords = data.length, //Define the total number of records
    totalPages = Math.ceil(totalRecords / recPerPage);  //Define de total pages

    if (totalRecords === 0){    //If the totalRecords is 0, display an error message
        var html = '';
        html += '<tr>';
        html += '<td colspan="5" style="text-align:center;">There are no records</td>';
        html += "</tr>"
        $("#table-data").html(html);
        $('#pagination').twbsPagination('destroy');
    }
    else{    //Otherwise, we destroy the previous instance of twbsPagination and create a new one
        $('#pagination').twbsPagination('destroy');
        $('#pagination').twbsPagination({
            totalPages : totalPages,
            visiblePages: 10,
            onPageClick: function(event, page){
                displayRecordsIndex = Math.max(page - 1, 0) * recPerPage;
                endRec = (displayRecordsIndex) + recPerPage;
                displayRecords = data.slice(displayRecordsIndex, endRec);
                insertData(displayRecords, username);
            }
        });
    }
}

//Function to insert the records into the table
function insertData(displayRecords, username){
    var html = '';
    var idioma = document.getElementById('idioma').value

    let borrar = ""
    let visualizar = ""
    let ejecutar = ""
    let copiar = ""
    let descargar = ""

    if (idioma === "es"){
        borrar = "Borrar";
        visualizar = "Visualizar";
        ejecutar = "Ejecutar";
        copiar = "Copiar";
        descargar = "Descargar";
    }
    else{
        borrar = "Delete";
        visualizar = "View";
        ejecutar = "Start";
        copiar = "Copy";
        descargar = "dwnld";
    }

    $.each(displayRecords, function(index, row){ //For each element in displayRecords, display it in the table
        console.log(displayRecords)
        html += '<tr>';
        html += '<td>' + row[13] + '</td>';
        html += '<td>' + row[1] + '</td>';
        const date = new Date(row[3]);
        html += '<td>' + formatDate(date) + '</td>';

        if(!row[12]){
            html += '<td id="estado-' + row[0] + '">' + row[2] + '</td>';
        }
        else{
            html += '<td id="estado-' + row[0] + '">Not Configured</td>';
        }



    //<i className="far fa-trash-alt"></i>
        html += "<td class='text-center'>";
        html += "<div class='row'>";

        html += "<div class=\"col d-xl-flex justify-content-xl-center align-items-xl-center\">";

        html += "<button type='button' class=\"btn-clase-1 btn btn-primary pull-right btn-to-enable-" + row[0] + "\"  onclick=\"document.getElementById('deleteProject_" + row[0] + "').submit();\">";


        html += "<svg xmlns=\"http://www.w3.org/2000/svg\" x=\"0px\" y=\"0px\" width=\"1em\" height=\"1em\" viewBox=\"0 0 26 26\">\n" +
            "<path d=\"M 10 2 L 9 3 L 5 3 C 4.4 3 4 3.4 4 4 C 4 4.6 4.4 5 5 5 L 7 5 L 17 5 L 19 5 C 19.6 5 20 4.6 20 4 C 20 3.4 19.6 3 19 3 L 15 3 L 14 2 L 10 2 z M 5 7 L 5 20 C 5 21.1 5.9 22 7 22 L 17 22 C 18.1 22 19 21.1 19 20 L 19 7 L 5 7 z M 9 9 C 9.6 9 10 9.4 10 10 L 10 19 C 10 19.6 9.6 20 9 20 C 8.4 20 8 19.6 8 19 L 8 10 C 8 9.4 8.4 9 9 9 z M 15 9 C 15.6 9 16 9.4 16 10 L 16 19 C 16 19.6 15.6 20 15 20 C 14.4 20 14 19.6 14 19 L 14 10 C 14 9.4 14.4 9 15 9 z\"></path>\n" +
            "</svg><span class='button-text'>"+ borrar +"</span></button></div>";

        html += "<form id='deleteProject_" + row[0] + "' action='/deleteProject' method='post' style='display: none;'>";
        html += "<input type='hidden' name='project_id_view' value='" + row[0] + "'>"; // Suponiendo que row[4] es el ID del proyecto
        html += "</form>";


        //First icon

        html += "<div class=\"col d-xl-flex justify-content-xl-center align-items-xl-center\">";
        if(row[2] === "Not started" || row[2] === "Running"){
            html += "<button class=\"btn-clase-1 btn btn-primary pull-right btn-to-enable-" + row[0] + "\" type=\"button\" disabled onclick=\"document.getElementById('viewProject_" + row[0] + "').submit();\">";

        }
        else{
           html += "<button type='button' class=\"btn-clase-1 btn btn-primary pull-right btn-to-enable-" + row[0] + "\"  onclick=\"document.getElementById('viewProject_" + row[0] + "').submit();\">";
        }

        html += "<i class=\"la la-eye iconos\"></i>&nbsp;<span class='button-text'>" + visualizar + "</span></button></div>";

        html += "<form id='viewProject_" + row[0] + "' action='/viewProject' method='post' style='display: none;'>";
        html += "<input type='hidden' name='project_id_view' value='" + row[0] + "'>"; // Suponiendo que row[4] es el ID del proyecto
        html += "</form>";

        //Second icon
        html += "<div class=\"col d-xl-flex justify-content-xl-center align-items-xl-center\">";
        if(row[2] === "Not started"){
            html += '<button id="inicio-' + row[0] + '" class="btn-clase-1 btn btn-primary pull-right botonStart"  type="button">';
        }
        else{
            html += "<button  class=\"btn-clase-1 btn btn-primary pull-right botonStart\" type=\"button\" disabled>";
        }

        html += "<svg class=\"bi bi-fast-forward-btn iconos\" xmlns=\"http://www.w3.org/2000/svg\" width=\"1em\" height=\"1em\" fill=\"currentColor\" viewBox=\"0 0 16 16\">\n" +
            "                                                    <path d=\"M8.79 5.093A.5.5 0 0 0 8 5.5v1.886L4.79 5.093A.5.5 0 0 0 4 5.5v5a.5.5 0 0 0 .79.407L8 8.614V10.5a.5.5 0 0 0 .79.407l3.5-2.5a.5.5 0 0 0 0-.814l-3.5-2.5Z\"></path>\n" +
            "                                                    <path d=\"M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V4Zm15 0a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V4Z\"></path>\n" +
            "                                                </svg>&nbsp;<span class='button-text'>" + ejecutar + "</span></button></div>";
        html += "<input type='hidden' class='projectId' name='project_id' value='" + row[0] + "'>";
        //Third icon
        html += "<div class=\"col d-xl-flex justify-content-xl-center align-items-xl-center\">";
        html += "<button class=\"btn btn-primary pull-right\" onclick=\"document.getElementById('duplicateForm_" + row[0] + "').submit();\" type=\"button\">";
        html += "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1em\" height=\"1em\" viewBox=\"0 0 24 24\" fill=\"none\" class=\"iconos\">\n" +
            "                                                            <path d=\"M19 5H7V3H21V17H19V5Z\" fill=\"currentColor\"></path>\n" +
            "                                                            <path d=\"M9 13V11H11V13H13V15H11V17H9V15H7V13H9Z\" fill=\"currentColor\"></path>\n" +
            "                                                            <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M3 7H17V21H3V7ZM5 9H15V19H5V9Z\" fill=\"currentColor\"></path>\n" +
            "                                                        </svg><span class='button-text'>" + copiar + "</span></button></div>";


        html += "<form id='duplicateForm_" + row[0] + "' action='/duplicate' method='post' style='display: none;'>";
        html += "<input type='hidden' name='project_id_duplicate' value='" + row[0] + "'>";
        html += "</form>";

        


        html += "<div class=\"col d-md-flex justify-content-md-center align-items-md-center\">";
        if(row[2] === "Not started" || row[2] === "Running"){
            html += "<button aria-hidden='true' class=\"btn btn-primary d-md-flex justify-content-md-center align-items-md-center pull-right btn-to-enable-" + row[0] +" btn-to-download-" + row[0] + "\" type=\"button\" disabled onclick=\"descargarArchivo('" + row[4] + "', '" + username + "', '" + row[0] + "')\">";
        }
        else{
            html += "<button aria-hidden='true' class=\"btn btn-primary pull-right btn-to-download-" + row[0] + "\" onclick=\"descargarArchivo('" + row[0] + "', '" + username + "', '" + row[1] + "')\">";
        }

        html += "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1em\" height=\"1em\" fill=\"currentColor\" viewBox=\"0 0 16 16\" class=\"bi bi-download iconos\">\n" +
            "                                                            <path d=\"M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z\"></path>\n" +
            "                                                            <path d=\"M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z\"></path>\n" +
            "                                                        </svg>&nbsp;<span class='button-text'>" + descargar + "</span></button></div>";

        html += "</div>"
        html += "</td>"

        //html += "<td>"
        //html += "<div class=\"progress blue\">"
        //html += "<span class=\"progress-left\">"
        //html += "<span class=\"progress-bar\"></span>"
       //html += "</span>"
        //html += "<span class=\"progress-right\">"
       // html += "<span class=\"progress-bar\"></span>"
       // html += "</span>"
       // html += "<div class=\"progress-value\">10%</div>"
       // html += "</div>"
      //  html += "</td>"

        html += "</tr>"
    });
    $("#table-data").html(html);
}
  
function descargarArchivo(id, username, filename) {
  var link = document.createElement('a');
  link.href = `./static/dirs/${username}/${filename}.zip`;
  link.download = `${filename}.zip`;
  document.body.appendChild(link); // Agrega el enlace al cuerpo del documento
  link.click(); // Simula un clic en el enlace
  document.body.removeChild(link); // Elimina el enlace del cuerpo del documento
}

  
  

const express = require("express");
const http = require('http');
const socketIo = require('socket.io')
const mysql = require('mysql2')
const cors = require('cors');

const app = express();
app.use(cors({
    origin: 'http://127.0.0.1:5000',
    methods: ['GET', 'POST'],
    credentials: true
}));
const server = http.createServer(app);
const io = socketIo(server, {
    cors:{
        origin: 'http://127.0.0.1:5000',
        methods: ["GET", "POST"],
        credentials: true
    }
});

const connection = mysql.createConnection({
   host: 'localhost',
   user: 'root',
   database: 'BD-TFG',
   password : ''
});

let intervals = {};

io.on('connection', (socket) => {
    console.log('Usuario conectado');

    // Cuando el cliente emite 'user_connected', únete a la sala específica
    socket.on('user_connected', (userId) => {
        console.log('Usuario con ID', userId, 'se ha unido a su sala privada.');
        socket.join(userId);  // El userId debería ser único para cada usuario
    });

    socket.on('detenerConsultas', (data) => {
        const projectId = data.projectId;
        console.log("Intentando detener consultas para: ", projectId);
        if (intervals[projectId]) {
            console.log("Deteniendo consultas para: ", projectId);
            clearInterval(intervals[projectId]);
            delete intervals[projectId];
        } else {
            console.log("No se encontró proceso para: ", projectId);
        }
    });
});

app.get('/actualizarEstado', (req, res) => {
    const estado = req.query.estado;
    const id_project = req.query.idProject;
    const userId = req.query.idUser;

    console.log("Se va a enviar esto al usuario con id ", userId)
    io.to(userId).emit('actualizarEstado', {estado, id_project});
    console.log("Estado actualizado " , estado)
    res.send('Estado actualizado');
});

app.get('/activarConsultas', (req, res) => {
    const projectId = req.query.id_project;  // Recibimos el parámetro aquí
    const nombreProyecto = req.query.nombreProyecto;
    const userId = req.query.idUser;
    const nameUser = req.query.nameUser;
    console.log("Entró a /activarConsultas con projectId:", projectId);
    console.log(nombreProyecto)

    // Iniciamos un intervalo para ejecutar la consulta cada 2 segundos
    const intervalId = setInterval(() => {
        connection.query('SELECT porcentajeCarga FROM proyectos WHERE idProyecto = ?', [projectId], (err, result) => {
            if (err) {
                console.error("Error ejecutando la consulta:", err);
                clearInterval(intervalId); // Detenemos el intervalo si hay un error
                return;
            }
            console.log(result);


            var porcentaje = result[0].porcentajeCarga

            io.to(userId).emit('cargaProyecto', {porcentaje, projectId, nombreProyecto, nameUser});

            // Si el porcentaje de carga es 99 o mayor, detenemos el intervalo
            if (result[0].porcentajeCarga >= 100) {
                clearInterval(intervalId);
                console.log("Porcentaje de carga alcanzó o superó el 99%. Intervalo detenido.");
                return;
            }


        });
    }, 2000); // 2000 milisegundos equivalen a 2 segundos

    intervals[projectId] = intervalId

    res.send('Consultas activadas para el projectId: ' + projectId);
});

server.listen(3001,  () => {
    console.log("Node.js server running on http://localhost:3001");
});

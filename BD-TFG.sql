-- phpMyAdmin SQL Dump
-- version 5.0.4
-- https://www.phpmyadmin.net/
--
-- Servidor: localhost
-- Tiempo de generación: 14-05-2024 a las 11:27:32
-- Versión del servidor: 10.4.17-MariaDB
-- Versión de PHP: 8.0.0

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `BD-TFG`
--
CREATE DATABASE IF NOT EXISTS `BD-TFG` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE `BD-TFG`;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `Datasets`
--

DROP TABLE IF EXISTS `Datasets`;
CREATE TABLE `Datasets` (
  `idDataset` int(11) NOT NULL,
  `idProyecto` int(11) NOT NULL,
  `fichero` longblob NOT NULL,
  `nombreFichero` varchar(250) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `Metricas`
--

DROP TABLE IF EXISTS `Metricas`;
CREATE TABLE `Metricas` (
  `idMetrica` int(11) NOT NULL,
  `idProyecto` int(11) NOT NULL,
  `MAE` double DEFAULT NULL,
  `MAPE` varchar(250) DEFAULT NULL,
  `MSE` double DEFAULT NULL,
  `RMSE` double DEFAULT NULL,
  `R2` double DEFAULT NULL,
  `Accuracy` double DEFAULT NULL,
  `_Precision` double DEFAULT NULL,
  `Sensibilidad` double DEFAULT NULL,
  `F1` double DEFAULT NULL,
  `Especificidad` double DEFAULT NULL,
  `NPV` double DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `Modelos`
--

DROP TABLE IF EXISTS `Modelos`;
CREATE TABLE `Modelos` (
  `idModelo` int(11) NOT NULL,
  `idProyecto` int(11) NOT NULL,
  `analisisModelo` longtext DEFAULT NULL,
  `nombreModelo` varchar(250) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `Proyectos`
--

DROP TABLE IF EXISTS `Proyectos`;
CREATE TABLE `Proyectos` (
  `idProyecto` int(11) NOT NULL,
  `nombreProyecto` varchar(250) DEFAULT NULL,
  `status` varchar(250) NOT NULL,
  `fecha` date NOT NULL,
  `clase` varchar(250) DEFAULT NULL,
  `idUsuario` int(11) NOT NULL,
  `tipoProblema` varchar(250) DEFAULT NULL,
  `nombresColumnasControl` longtext DEFAULT NULL,
  `nombresColumnasExterior` longtext DEFAULT NULL,
  `validacion` varchar(250) DEFAULT NULL,
  `clasePositiva` longtext DEFAULT NULL,
  `porcentajeCarga` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `Usuario`
--

DROP TABLE IF EXISTS `Usuario`;
CREATE TABLE `Usuario` (
  `id` int(11) NOT NULL,
  `username` varchar(150) NOT NULL,
  `password` longtext NOT NULL,
  `email` varchar(150) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Índices para tablas volcadas
--

--
-- Indices de la tabla `Datasets`
--
ALTER TABLE `Datasets`
  ADD PRIMARY KEY (`idDataset`),
  ADD KEY `idProyecto` (`idProyecto`);

--
-- Indices de la tabla `Metricas`
--
ALTER TABLE `Metricas`
  ADD PRIMARY KEY (`idMetrica`),
  ADD KEY `idProyecto` (`idProyecto`);

--
-- Indices de la tabla `Modelos`
--
ALTER TABLE `Modelos`
  ADD PRIMARY KEY (`idModelo`),
  ADD KEY `idProyecto` (`idProyecto`);

--
-- Indices de la tabla `Proyectos`
--
ALTER TABLE `Proyectos`
  ADD PRIMARY KEY (`idProyecto`),
  ADD KEY `idUsuario` (`idUsuario`);

--
-- Indices de la tabla `Usuario`
--
ALTER TABLE `Usuario`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT de las tablas volcadas
--

--
-- AUTO_INCREMENT de la tabla `Datasets`
--
ALTER TABLE `Datasets`
  MODIFY `idDataset` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT de la tabla `Metricas`
--
ALTER TABLE `Metricas`
  MODIFY `idMetrica` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT de la tabla `Modelos`
--
ALTER TABLE `Modelos`
  MODIFY `idModelo` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT de la tabla `Proyectos`
--
ALTER TABLE `Proyectos`
  MODIFY `idProyecto` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT de la tabla `Usuario`
--
ALTER TABLE `Usuario`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- Restricciones para tablas volcadas
--

--
-- Filtros para la tabla `Datasets`
--
ALTER TABLE `Datasets`
  ADD CONSTRAINT `datasets_ibfk_1` FOREIGN KEY (`idProyecto`) REFERENCES `Proyectos` (`idProyecto`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Filtros para la tabla `Metricas`
--
ALTER TABLE `Metricas`
  ADD CONSTRAINT `metricas_ibfk_1` FOREIGN KEY (`idProyecto`) REFERENCES `Proyectos` (`idProyecto`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Filtros para la tabla `Modelos`
--
ALTER TABLE `Modelos`
  ADD CONSTRAINT `modelos_ibfk_1` FOREIGN KEY (`idProyecto`) REFERENCES `Proyectos` (`idProyecto`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Filtros para la tabla `Proyectos`
--
ALTER TABLE `Proyectos`
  ADD CONSTRAINT `proyectos_ibfk_1` FOREIGN KEY (`idUsuario`) REFERENCES `Usuario` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;

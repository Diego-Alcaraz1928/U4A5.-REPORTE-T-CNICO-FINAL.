# Proyecto Final Gemelo Digital para Robot Seguidor de Línea con IA

## Descripción

Este proyecto implementa un Sistema Ciberfísico (CPS) que integra un robot seguidor de línea físico, un gemelo digital desarrollado en Python, comunicación IoT mediante MQTT y un modelo de Inteligencia Artificial basado en Random Forest.

El objetivo principal es analizar el comportamiento del robot, reducir las oscilaciones durante el seguimiento de la línea y validar mejoras mediante simulación antes de aplicarlas al sistema físico.

## Objetivos

- Capturar telemetría del robot en tiempo real.
- Almacenar y procesar datos mediante Node-RED.
- Entrenar un modelo de Machine Learning para identificar estados operativos.
- Implementar un gemelo digital capaz de replicar el comportamiento del robot físico.
- Evaluar configuraciones de zona muerta para reducir oscilaciones.
- Comparar resultados entre el sistema físico y el entorno virtual.

## Tecnologías Utilizadas

- Python
- PyQt5
- ESP32
- MQTT
- Node-RED
- Pandas
- Scikit-Learn
- Random Forest
- CSV para almacenamiento de telemetría

## Arquitectura del Sistema

Robot Físico → MQTT → Node-RED → Dataset CSV → Entrenamiento IA → Modelo Random Forest → Gemelo Digital

## Variables Registradas

Durante la operación se registran las siguientes variables:

- s0, s1, s2, s3: Lecturas de sensores QTR.
- error: Error de seguimiento.
- corr: Corrección PID.
- left: Velocidad motor izquierdo.
- right: Velocidad motor derecho.
- kp: Ganancia proporcional.
- kd: Ganancia derivativa.
- estado: Estado operativo del robot.

## Estados Clasificados

El modelo identifica los siguientes estados:

- centrado
- desviado
- vuelta_derecha
- vuelta_izquierda
- recuperacion
- fuera_linea

## Inteligencia Artificial

Se utilizó un modelo de clasificación Random Forest para identificar el estado operativo del robot a partir de la telemetría capturada.

### Métricas evaluadas

- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de Confusión

## Gemelo Digital

El gemelo digital fue desarrollado en Python utilizando PyQt5.

Características principales:

- Simulación de la pista.
- Sensores virtuales.
- Control PID.
- Modelo físico del robot.
- Visualización en tiempo real.
- Registro de telemetría.
- Comparación con datos reales.

## Resultados

Se logró una representación digital capaz de reproducir de forma aceptable el comportamiento del robot físico.

Las mejoras obtenidas mediante el entrenamiento permitieron:

- Reducir oscilaciones.
- Mejorar la estabilidad.
- Disminuir el tiempo de recorrido.
- Evaluar configuraciones antes de aplicarlas al sistema real.

## Estructura del Proyecto

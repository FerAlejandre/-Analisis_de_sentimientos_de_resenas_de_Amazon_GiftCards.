# Análisis de Sentimientos de Reseñas de Amazon Gift Cards

Este proyecto fue desarrollado por **Fernando Alejandre** como parte del curso de **Certificación en Python Data Engineering**.

## Descripción del Proyecto

El objetivo del proyecto es realizar un análisis de sentimientos en las reseñas de productos (Gift Cards) de Amazon utilizando técnicas de procesamiento de lenguaje natural (NLP) y el modelo preentrenado BERT. Los resultados se visualizan mediante gráficos que clasifican las reseñas en categorías como satisfecho, neutral e insatisfecho.

## Documentación del Proyecto

#### Version 1
   ##### Crear el ambiente virtual 
    Para no afectar el ambiente global python, se necesita crear un virtual env e instalar las librerías contenidas en el archivo "requirements.txt"

    1. python -m venv env
    2. env\Scripts\activate
    3. pip install -r requirements.txt

    ##### Descargar el modelo Bert por primera vez

    El archivo Bert.py tiene un módulo de pruebas que se pueden correr, ejecutando este módulo directo en el Python REPL. Al ejecutarlo, se generará una carpeta local llamada nlptown donde se guardará el modelo y el tokenizer.

    ##### Ejecutar el módulo main.py

    1. Desde VS Code, puedes abrir el archivo main.py y ejecutarlo con el run and debug.
    2. Desde linea de comandos, puedes correr con:
    
    > python main.py

#### Flujo del Proceso

El análisis de sentimientos sigue los siguientes pasos:

   1. Carga de datos:
   Se lee un archivo JSON que contiene comentarios de productos de Amazon.

   2. Inicialización de objetos:
   Los datos se convierten en una lista de objetos Review, donde se almacenan las propiedades relevantes de cada reseña.

   3. Limpieza de datos:

      Normalización del texto (minúsculas, eliminación de puntuación y dígitos).
      Eliminación de palabras irrelevantes (stopwords).
      Tokenización del texto.

   4. Análisis de sentimientos:
   Se utiliza el modelo preentrenado BERT para evaluar cada reseña y asignarle una puntuación de sentimiento.

   5. Visualización:
   Se generan gráficos de barras que clasifican las reseñas en tres categorías:
      Insatisfecho: Puntuaciones entre 0 y 0.3.
      Neutral: Puntuaciones entre 0.3 y 0.5.
      Satisfecho: Puntuaciones superiores a 0.5.

#### Dependencias

El proyecto utiliza las siguientes librerías y herramientas clave:

   Transformers: Para la integración con el modelo BERT.
   NLTK: Procesamiento de lenguaje natural (tokenización, stopwords).
   Matplotlib: Generación de gráficos.
   PyTorch: Ejecución del modelo de aprendizaje profundo.
   
   Consulta requirements.txt para más detalles sobre las versiones específicas.

#### Resultados

Al finalizar la ejecución, se generarán gráficos que reflejan el análisis de sentimientos de las reseñas de Gift Cards de Amazon. 
Estos gráficos ayudan a identificar patrones de satisfacción, neutralidad e insatisfacción entre los usuarios.

# Demo Streamlit

Demo para desplegar un producto MVP del modelo de machine learning creado para detectar imágenes

## :boom: Desplegar la aplicación

Para desplegar la aplicación, desde la carpeta `/frontend` corremos el siguiente comando:

```
$ streamlit run main.py
```

## :racehorse: Funcionamiento

La parte de API no está implementada, por lo que los únicos archivos que se utilizan para el funcionamiento actual de la aplicación son:

`frontend/`
1.  `main.py` : Contiene los estilos de página y acciona la ejecución de las subpáginas
2.  `functions.py` : Contiene todas las funciones utilizadas para el deploy de la aplicación
3.  `page_inicio.py` : Acciona la ejecución de la carga de imágenes y posterior predicción de características

`weights/` : Almacena todos los pesos de los modelos pre-entrenados

`images/` : Contiene las imágenes utilizadas en la aplicación

`files/` : Contiene el esquema de la página y el url de la API (que no se utiliza por el momento)

## :warning: Consideraciones
La parte de predicción requiere de la instalación de `tensorflow` y `keras` pero la aplicación se cae al importar los packages

# MovieLab · aplicación de escritorio

MovieLab es un programa Python de escritorio para experimentar con sistemas de recomendación de películas. La interfaz está construida con Tkinter y se abre como una ventana normal del sistema: no utiliza navegador ni servidor web.

## Algoritmos

1. Modelo base con media global y sesgos regularizados.
2. K-NN usuario-usuario.
3. K-NN ítem-ítem.
4. Modelo basado en contenido mediante TF-IDF de géneros y, opcionalmente, títulos.
5. Modelo híbrido que combina las cuatro predicciones anteriores.

## Funciones de la aplicación

- Base SQLite de demostración con 60 películas, 90 usuarios y 1.842 valoraciones.
- Apertura de carpetas MovieLens, parejas de CSV y bases SQLite.
- Guardado del catálogo activo como una nueva base SQLite.
- Parámetros independientes para ambos K-NN: vecinos, similitud mínima y regularización.
- Regularización e iteraciones configurables para el modelo base.
- Inclusión opcional del título y peso de los géneros en el modelo de contenido.
- Pesos editables y normalizados automáticamente para el híbrido.
- Optimizador automático de los pesos híbridos según RMSE o MAE.
- Comparación mediante RMSE, MAE, precisión@K, recall@K, NDCG@K, cobertura, diversidad, novedad y tiempos.
- Gráfico intercambiable para cualquiera de las métricas.
- Recomendaciones filtradas por género, intervalo de años y número mínimo de votos.
- Control para equilibrar afinidad y diversidad de la lista.
- Explicación de las recomendaciones y de la aportación de cada componente híbrido.
- Exportación de comparaciones y recomendaciones a CSV.
- Buscador estadístico de películas por título, género, votos y nota.
- Análisis individual de usuarios, distribución de notas y género favorito.
- Comparación de géneros por catálogo, valoraciones, usuarios y nota media.
- Informe de calidad: ausentes, duplicados, esparsidad, casos con pocos votos y desigualdad de popularidad.
- Curva de concentración para analizar la cola larga del catálogo.
- Formulario para crear un usuario, registrar sus valoraciones y géneros preferidos.
- Dos formularios de alta: uno sencillo para uso cotidiano y otro avanzado para experimentar.
- Recomendación inmediata para usuarios nuevos y opción de guardarlos en la base activa.

## Iniciar

Desde la carpeta del proyecto:

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

También se puede iniciar con:

```bash
python3 desktop_app.py
```

## Cargar MovieLens

En la pestaña **Datos** o en el menú **Archivo**, selecciona una carpeta que contenga:

- `movies.csv`: `movieId`, `title`, `genres`;
- `ratings.csv`: `userId`, `movieId`, `rating` y, opcionalmente, `timestamp`.

También se pueden elegir los dos archivos por separado. Cuando el conjunto supera 2.500 películas, MovieLab ofrece limitarlo a los elementos y usuarios más activos para evitar que las matrices K-NN consuman demasiada memoria.

## Comparación correcta

Todos los algoritmos utilizan exactamente la misma partición de entrenamiento y prueba. La partición se realiza por usuario y deja al menos dos valoraciones para entrenar. La proporción, semilla, nota considerada relevante y tamaño del top-K se pueden modificar.

El optimizador de pesos utiliza la partición reservada como validación. En un experimento académico conviene separar además una prueba final que no se use durante la optimización.

## Introducir un usuario nuevo

Abre la pestaña **Nuevo usuario** y elige uno de los dos modos:

- **Sencillo:** introduce un nombre, marca géneros, elige al menos tres películas y responde con “No me gustó”, “Regular”, “Me gustó” o “Me encantó”. MovieLab configura automáticamente el híbrido.
- **Avanzado:** permite introducir notas de 0,5 a 5 y controlar algoritmo, cantidad, impulso de géneros y diversidad.

**Recomendar** crea un perfil temporal y entrena el algoritmo elegido. **Guardar usuario** incorpora sus valoraciones a la base activa. Para conservarlo después de cerrar el programa, utiliza **Archivo → Guardar base como SQLite**.

## Pruebas

```bash
python3 -m unittest discover -s tests -v
```

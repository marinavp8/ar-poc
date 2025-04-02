# Proyecto de Detección de Poses en AR

Este proyecto implementa detección de poses en tiempo real usando dos enfoques diferentes: MediaPipe y MoveNet. Captura video de tu webcam y muestra el seguimiento esquelético en tiempo real.


## Configuración Paso a Paso

### 1. Crear un Entorno Virtual
```bash
# En macOS/Linux:
python3 -m venv myenv
source myenv/bin/activate

# En Windows:
python -m venv myenv
myenv\Scripts\activate
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```
Este comando lee el archivo requirements.txt e instala:
- opencv-python: para manejo de la cámara y video
- mediapipe: para la detección de poses (versión 1)
- tensorflow y tensorflow-hub: para MoveNet (versión 2)
- numpy

### 3. Ejecutar el Programa
```bash
# Para la versión con MediaPipe:
python p1.py

# Para la versión con MoveNet:
python p2.py
```

## Detalles Técnicos

### Modelos
- **MediaPipe**: Biblioteca de Google que incluye modelos pre-entrenados para detección de poses
- **MoveNet**: Modelo de TensorFlow Hub, se descarga automáticamente la primera vez que ejecutas el programa

### Diferencias entre Versiones
1. **MediaPipe (p1.py)**:
   - Más fácil de usar
   - Mejor para comenzar
   - Conexiones entre puntos automáticas

2. **MoveNet (p2.py)**:
   - Supuestamente es más rápido
   - Más personalizable
   - Mejor para desarrollo avanzado



## Uso del Programa

1. Activa el entorno virtual
2. Ejecuta el programa (p1.py o p2.py)
3. Verás:
   - Puntos en las articulaciones detectadas
   - Líneas conectando los puntos

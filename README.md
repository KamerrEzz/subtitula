# YouTube Whisper Studio

Aplicación de escritorio para Windows 11 que descarga videos de YouTube y los transcribe **localmente**
con [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No usa APIs de pago ni envía tus
archivos a ningún servicio externo.

Flujo principal: **pegar enlace → añadir a la cola → iniciar → obtener la transcripción**.

## Qué hace

- Descarga solo el audio de un video y lo transcribe (lo más rápido).
- Descarga el video completo (eligiendo la calidad) y lo transcribe.
- Solo descarga, sin transcribir.
- Transcribe archivos locales: MP4, MKV, MOV, AVI, WEBM, MP3, WAV, M4A, FLAC (también OGG y OPUS).
  Puedes seleccionar archivos, una carpeta completa o arrastrarlos a la ventana.
- Detecta listas de reproducción y añade cada video como un trabajo independiente.
- Procesa la cola en orden, sin bloquear la interfaz; si un trabajo falla, sigue con el siguiente.
- Muestra el texto mientras se transcribe y permite copiarlo o abrirlo desde la propia aplicación.
- Exporta TXT, SRT, VTT, Markdown y JSON.
- Usa la GPU NVIDIA cuando funciona de verdad; si no, avisa y continúa con la CPU.

## Requisitos

| Componente | ¿Hace falta instalarlo aparte? |
|------------|--------------------------------|
| [uv](https://docs.astral.sh/uv/) | Sí: `winget install astral-sh.uv` |
| Python 3.12 | No, uv lo descarga solo |
| Dependencias de Python, cuBLAS y cuDNN de NVIDIA, Deno (para YouTube) | No, las instala `uv sync` |
| FFmpeg | **Recomendado**: `winget install Gyan.FFmpeg` |
| Controlador NVIDIA | Solo para usar la GPU (ver abajo) |

### FFmpeg

Sin FFmpeg la aplicación funciona, pero con limitaciones:

- Los videos en modo "Video + transcripción" se descargan en un único archivo de calidad limitada,
  porque no se pueden unir pistas separadas de video y audio.
- No se puede conocer la duración de los archivos de antemano, así que la barra de progreso de la
  transcripción se muestra como indeterminada.

Después de instalarlo, cierra y vuelve a abrir la terminal para que se detecte.

### GPU NVIDIA

Solo necesitas un **controlador NVIDIA actualizado** compatible con CUDA 12 (descárgalo desde
[nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)). **No** hace falta instalar el
CUDA Toolkit ni cuDNN: las bibliotecas necesarias (`cublas64_12.dll`, `cudnn64_9.dll`) vienen en
paquetes de Python que instala `uv sync`, y la aplicación las registra solo para su propio proceso.
No modifica el PATH del sistema ni la configuración de Windows.

Al cargar el modelo, la aplicación ejecuta una inferencia real en la GPU para comprobar que CUDA
funciona. Detectar una tarjeta no basta: si la prueba falla (bibliotecas, memoria, controlador),
verás un aviso con el motivo y el trabajo continuará en CPU con el **mismo modelo** y precisión
`int8`. Si eliges "GPU (CUDA)" en vez de "Automático", la aplicación no recurre a la CPU y muestra
el error.

## Instalación y uso

```powershell
uv sync
uv run python main.py
```

O haz doble clic en `start.bat`, que instala lo necesario y abre la aplicación.

La primera vez que uses un modelo se descarga automáticamente (de `tiny`, unos 75 MB, a
`large-v3`, unos 3 GB) en `data/models`.

## Elegir modelo

| Modelo | Calidad | Uso recomendado |
|--------|---------|-----------------|
| tiny / base | Baja | Pruebas rápidas |
| small | Buena | CPU (valor predeterminado sin GPU) |
| medium | Muy buena | GPU (valor predeterminado con CUDA) |
| large-v3 | La mejor | GPU con 6 GB de VRAM o más |

Con una GPU de 4 GB, `medium` en `int8_float16` funciona bien. Si aparece un error de memoria
insuficiente, elige un modelo más pequeño o cambia a CPU.

## Dónde quedan los resultados

Cada video o archivo tiene su propia carpeta dentro de la carpeta de salida (por defecto `outputs/`):

```text
outputs/
  Mi video-[dQw4w9WgXcQ]/
    transcripcion.txt
    transcripcion.srt
    transcripcion.vtt
    transcripcion.md
    metadata.json
```

- El audio descargado para transcribir se borra al terminar, salvo que actives
  "Conservar archivos multimedia". En "Video + transcripción" y "Solo descargar" el archivo
  descargado siempre se conserva.
- Los archivos locales nunca se modifican ni se mueven.
- Si ya existe una transcripción completa, se omite o se regenera según la opción
  "Si ya existe una transcripción".
- Si cancelas un trabajo, se borran solo sus archivos parciales; las transcripciones anteriores
  quedan intactas.

## Configuración y registros

- Preferencias: `data/settings.json` (se guardan automáticamente al cambiar cualquier opción).
- Registro técnico de errores: `data/logs/app.log`.
- El botón **Diagnóstico** comprueba Python, FFmpeg, FFprobe, faster-whisper, CUDA, cuBLAS,
  cuDNN, permisos de escritura y la carpeta de modelos, y explica cómo resolver cada problema.

## Limitaciones

- Solo descarga contenido accesible públicamente. No elude DRM, inicios de sesión ni
  restricciones de edad o región.
- El progreso de la transcripción se calcula con la duración real del archivo; con el filtro de
  voz (VAD) activado avanza a saltos, porque los silencios se omiten.

## Desarrollo

```powershell
uv run pytest
```

El código está en `whisper_studio/`: descargas (`downloader.py`), motor de transcripción
(`transcriber.py`), exportación (`exporters.py`), cola de trabajos (`job_queue.py`),
configuración (`config.py`), diagnóstico (`diagnostics.py`) e interfaz (`ui/`).

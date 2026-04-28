Esta es una primera versión de la API REST

============================
API
============================
- como tal el código de la API es "api_sira.py"
- IMPORTANTE: La API NO se lanza con el comando estándar de Python. Al ser asíncrona, requiere su propio servidor web (Uvicorn).
- El comando exacto a ejecutar en la terminal (con el entorno virtual activado) es:
  uvicorn api_sira:app --host 0.0.0.0 --port 8000
- El parámetro "--host 0.0.0.0" es vital para que la UI pueda conectarse desde una tablet o pantalla externa. Si se omite, solo funcionará en la propia máquina (localhost).
- La API ya tiene programado un "Watchdog" (Perro guardián): Si el script del micrófono crashea o se desconecta el cable, la API lo detecta a los 2 segundos y resetea las alertas a cero. El equipo de UI NO necesita programar lógicas de caducidad de datos, la API ya les entrega datos seguros.
- comenzará su ejecución y mostrará las peticiones HTTP que vaya recibiendo en todo momento 
- para visualizar los cambios hay que abrir el código "display.html"
- probé su funcionamiento con el código de detección en tiempo real y conseguí visualizar las actualizaciones de la probabilidad de sirena.
============================
detección sirena tiempo real
============================
- se ejecuta el código "piloto_micro_api.py"
- es igual al que teníamos hasta ahora de la CNN pero añadiendo la instruccion de publicar cada cierto tiempo las actualizaciones de los outputs que genera (booleano sirena, probabilidad sirena, threshold establecido, timestamps para calcular la latencia del sistema...)
- para ello hace llamadas POST a la API con un JSON con los datos en cada momento
===============================
Como continuar a partir de esto
===============================
- con ayuda de la IA presentandole el codigo de la API adaptar el codigo del DOA para que haga los POST correspondientes también
- modificar/añadir cualquier cosa necesaria
- esto tiene que ir a la par con el desarrollo de la UI porque es de donde va a sacar los datos para mostrar todo
- en la UI sería necesario hacer las llamadas correspondientes (GET) a la API para mostrar todo refrescandolo cada cierto tiempo

- la idea que tengo ejecución del sistema completo sería:
	1. lanzar la API
	2. lanzar detección en tiempo real
	3. lanzar DOA
	4. lanzar la UI / visualizarla
- sería interesante crear un sript para automatizarlo al encender el minipc
- dejé un documento con una respuesta que me dió gemini con un ejemplo de script para nuestro caso, habría que adaptarlo

=============================================================
Comunicación detección - DOA para lanzarlo cuando haya sirena
=============================================================
- me dice la IA que lo ideal sería que el código de detección avise al de DOA cuando detecte sirena
- sino el código de DOA tendría que estar haciendo llamadas GET todo el rato a la API para identificarlo con la mayor brevedad posible lo que saturaría la API
- esta comunicación sería de la siguiente manera: 2 alternativas

	
  	1. (Alternativa si se separan los scripts): Si por arquitectura se decide mantener Detección y DoA en dos archivos .py físicamente distintos, la comunicación NO se hará por la 	  API REST, sino mediante memoria compartida o un socket interno ultrarrápido (como ZeroMQ) enviando un simple flag booleano (True/False) de un script a otro.

	2. Integración en un solo Script Maestro ("Piloto Micro"): Ambos algoritmos (la red neuronal CNN y el cálculo DoA) conviven en el mismo programa de Python 
	para no pelear por el acceso a la tarjeta de sonido.
  	* Ejecución Condicional (Trigger Local): El algoritmo de detección procesa el audio en bucle. Únicamente en el milisegundo en que la probabilidad de 
	sirena supera el umbral configurado (ej. > 0.65), llama directamente a la función matemática del DoA en la memoria local.
  	* Ahorro masivo de recursos: El algoritmo DoA se mantiene "dormido" el 99% del tiempo. Solo gasta ciclos de CPU analizando los 4 canales de audio 
	cuando la red neuronal le da luz verde.
  	* Publicación (Push): Una vez el DoA calcula el ángulo de procedencia, este mismo script maestro lanza las peticiones POST asíncronas 
	hacia la API REST para actualizar el display, evitando cualquier tipo de llamada GET de "pregunta" entre los módulos de inteligencia artificial.

- ADVERTENCIA SOBRE EL HARDWARE (TARJETA DE SONIDO): 
- En la secuencia de arranque (pasos 2 y 3), si ejecutamos Detección y DoA como dos scripts separados (.py), el Sistema Operativo (Linux/Windows) bloqueará el micrófono. Un dispositivo de audio normalmente no permite ser abierto y leído por dos programas distintos al mismo tiempo ("Device or resource busy").
- Por este motivo, se recomienda encarecidamente la ruta del "Script Maestro" (punto 2 de la sección de comunicación), donde un solo programa abre el micrófono (pidiendo los 4 canales a la vez), hace la inferencia y, si hay sirena, pasa esa misma matriz de audio en RAM a la función DoA.

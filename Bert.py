# Desarrollado por "Fernando Alejandre"
# Este script utiliza un modelo preentrenado de BERT para realizar análisis de sentimiento en texto.

# Importar las bibliotecas necesarias de transformers y torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Para cargar el modelo y el tokenizador
import torch  # Para manejar tensores y realizar cálculos con el modelo

# Definir el modelo preentrenado que se utilizará para el análisis de sentimiento
MODEL = f"nlptown/bert-base-multilingual-uncased-sentiment"

# Cargar el tokenizador asociado al modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Cargar el modelo de clasificación de secuencias preentrenado
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analyze(txt):
    '''
    Función para analizar el sentimiento de un texto dado.

    Args:
        txt (str): El texto que se desea analizar.

    Returns:
        int: Un valor numérico que representa el sentimiento (1 a 5).
    '''
    # Codificar el texto en tokens utilizando el tokenizador preentrenado
    tokens = tokenizer.encode(txt, return_tensors='pt')  # Convertir texto en tensores para el modelo

    # Pasar los tokens al modelo y obtener los resultados (logits)
    result = model(tokens)

    # Determinar la categoría de sentimiento (1 a 5) basada en el índice con mayor probabilidad
    resultado = int(torch.argmax(result.logits)) + 1  # Convertir índice a rango 1-5
    return resultado

# Punto de entrada principal del script
if __name__ == '__main__':
    '''
    Punto de entrada principal del programa.

    Se analiza un texto de ejemplo y se imprime el resultado de su sentimiento.
    '''
    # Analizar el sentimiento de un texto de prueba
    resultado = analyze("ok")  # Texto de ejemplo "ok"

    # Imprimir el resultado del análisis de sentimiento
    print(resultado)  # Salida: un número del 1 al 5 que indica el sentimiento
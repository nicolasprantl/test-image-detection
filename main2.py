import numpy as np
import cv2
import matplotlib.pyplot as plt

def detectar_tarjeta(imagen):
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral adaptativo para resaltar las esquinas negras
    _, umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar los contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande (asumiendo que es la tarjeta)
    contorno_tarjeta = max(contornos, key=cv2.contourArea)

    # Aproximar el contorno con un polígono (posible trapecio)
    epsilon = 0.1 * cv2.arcLength(contorno_tarjeta, True)
    approx = cv2.approxPolyDP(contorno_tarjeta, epsilon, True)

    # Si el contorno aproximado no tiene 4 vértices, no se puede determinar un trapecio
    if len(approx) != 4:
        return None, None

    # Crear una máscara en blanco del tamaño de la imagen original
    mascara = np.zeros_like(gris)

    # Dibujar el contorno aproximado en la máscara
    cv2.drawContours(mascara, [approx], -1, (255), cv2.FILLED)

    # Aplicar la máscara a la imagen original
    imagen_mascarada = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Recortar la región de interés (sección detectada)
    seccion_detectada = imagen_mascarada.copy()

    return seccion_detectada

# Ruta de la imagen de la tarjeta "DKK Digital Kolor Kard"
ruta_imagen = "/Users/nprantl/Desktop/Evaluacion_camara/AlgunasCapturas/Color.jpg"

# Cargar la imagen
imagen = cv2.imread(ruta_imagen)

# Detectar la tarjeta y obtener la sección detectada (recorte no rectangular)
seccion_detectada = detectar_tarjeta(imagen)

# Mostrar la sección detectada
plt.imshow(cv2.cvtColor(seccion_detectada, cv2.COLOR_BGR2RGB))
plt.title("Sección detectada")
plt.axis("off")

plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt


def recortar_cuadro_colores(imagen_tarjeta):
    # Cargar la imagen de la tarjeta
    img_tarjeta = cv2.imread(imagen_tarjeta)

    # Convertir la imagen a escala de grises
    img_gray = cv2.cvtColor(img_tarjeta, cv2.COLOR_BGR2GRAY)

    # Aplicar el algoritmo de detección de esquinas de Harris
    corners = cv2.cornerHarris(img_gray, 2, 3, 0.04)

    # Obtener las coordenadas de las esquinas
    corners = cv2.dilate(corners, None)
    _, corners_binary = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)
    corners_binary = np.uint8(corners_binary)
    contours, _ = cv2.findContours(corners_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Calcular las dimensiones del cuadro de colores en función del contenido de la imagen
    min_x = np.min(box[:, 0])
    max_x = np.max(box[:, 0])
    min_y = np.min(box[:, 1])
    max_y = np.max(box[:, 1])
    width = max_x - min_x
    height = max_y - min_y

    # Ajustar las dimensiones del rectángulo para tener una proporción adecuada
    aspect_ratio = width / height
    target_ratio = 5 / 7
    if aspect_ratio > target_ratio:
        new_height = width / target_ratio
        offset = int((height - new_height) // 2)
        min_y += offset
        max_y -= offset
    else:
        new_width = height * target_ratio
        offset = int((width - new_width) // 2)
        min_x += offset
        max_x -= offset

    # Recortar la región de interés (cuadro de colores)
    roi_tarjeta = img_tarjeta[min_y:max_y, min_x:max_x]

    # Desplegar la imagen recortada utilizando plt
    plt.imshow(cv2.cvtColor(roi_tarjeta, cv2.COLOR_BGR2RGB))
    plt.title('Cuadro de Colores Recortado')
    plt.axis('off')
    plt.show()

    # Devolver la imagen recortada
    return roi_tarjeta


def calcular_fidelidad_color(imagen_referencia, imagen_tarjeta):
    # Cargar las imágenes
    img_referencia = cv2.imread(imagen_referencia)
    img_tarjeta = cv2.imread(imagen_tarjeta)

    # Recortar la región de interés (cuadro de colores)
    roi_tarjeta = recortar_cuadro_colores(imagen_tarjeta)

    # Redimensionar la imagen de referencia para que coincida con el tamaño del cuadro de colores de la tarjeta
    img_referencia = cv2.resize(img_referencia, (roi_tarjeta.shape[1], roi_tarjeta.shape[0]))

    # Calcular la diferencia absoluta entre los píxeles de las imágenes
    diferencia = cv2.absdiff(roi_tarjeta, img_referencia)

    # Calcular el promedio de la diferencia de color para obtener la fidelidad de color
    fidelidad_color = diferencia.mean()

    return fidelidad_color


def calcular_cdi(imagen_referencia, imagen_tarjeta):
    # Cargar las imágenes
    img_referencia = cv2.imread(imagen_referencia)
    img_tarjeta = cv2.imread(imagen_tarjeta)

    # Recortar la región de interés (cuadro de colores)
    roi_tarjeta = recortar_cuadro_colores(imagen_tarjeta)

    # Redimensionar la imagen de la tarjeta para que tenga las mismas dimensiones que la imagen de referencia
    img_tarjeta = cv2.resize(img_tarjeta, (img_referencia.shape[1], img_referencia.shape[0]))

    # Convertir las imágenes a formato CIELAB
    img_referencia_lab = cv2.cvtColor(img_referencia, cv2.COLOR_BGR2LAB)
    img_tarjeta_lab = cv2.cvtColor(img_tarjeta, cv2.COLOR_BGR2LAB)

    # Obtener los componentes L, A y B de las imágenes
    l_referencia, a_referencia, b_referencia = cv2.split(img_referencia_lab)
    l_tarjeta, a_tarjeta, b_tarjeta = cv2.split(img_tarjeta_lab)

    # Calcular la diferencia promedio entre los componentes de color recortados
    cdi = np.mean(np.abs(l_referencia - l_tarjeta) +
                  np.abs(a_referencia - a_tarjeta) +
                  np.abs(b_referencia - b_tarjeta))

    # Mostrar las imágenes y los pasos intermedios
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Imagen de la tarjeta con el cuadro de colores recortado
    axes[0, 0].imshow(cv2.cvtColor(roi_tarjeta, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Región de Interés (Cuadro de Colores)')

    # Imagen de referencia
    axes[0, 1].imshow(cv2.cvtColor(img_referencia, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Imagen de Referencia')

    # Imagen de la tarjeta
    axes[1, 0].imshow(cv2.cvtColor(img_tarjeta, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Imagen de la Tarjeta')

    # Mostrar el CDI
    fig.suptitle(f'Índice de Diferencia de Color (CDI): {cdi:.2f}', fontsize=12)
    plt.tight_layout()
    plt.show()

    return cdi


def detectar_tarjeta(imagen):
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para separar la tarjeta del fondo
    _, umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar los contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande (asumiendo que es la tarjeta)
    contorno_tarjeta = max(contornos, key=cv2.contourArea)

    # Calcular el rectángulo delimitador de la tarjeta
    x, y, w, h = cv2.boundingRect(contorno_tarjeta)

    # Recortar la región de interés (tarjeta) de la imagen original
    tarjeta = imagen[y:y + h, x:x + w]

    # Dibujar el rectángulo delimitador en la imagen original
    cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return imagen, tarjeta

# Ruta de la imagen TIFF de referencia y la imagen JPEG de la tarjeta
imagen_referencia = '/Users/nprantl/Desktop/Evaluacion_camara/AlgunasCapturas/referencia_color_rgb_a.tif'
imagen_tarjeta = '/Users/nprantl/Desktop/Evaluacion_camara/AlgunasCapturas/Color_1.jpg'

imagen = cv2.imread(imagen_tarjeta)

# Detectar la tarjeta en la imagen y obtener la tarjeta recortada
imagen_con_tarjeta, tarjeta_recortada = detectar_tarjeta(imagen)

# Mostrar la imagen resultante con la tarjeta detectada
cv2.imshow('Tarjeta detectada', imagen_con_tarjeta)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar la tarjeta recortada
cv2.imshow('Tarjeta recortada', tarjeta_recortada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Calcular la fidelidad de color
# fidelidad = calcular_fidelidad_color(imagen_referencia, imagen_tarjeta)
#
# # Calcular el Índice de Diferencia de Color (CDI)
# cdi = calcular_cdi(imagen_referencia, imagen_tarjeta)
#
# print(f'La fidelidad de color es: {fidelidad}')
# print(f'El Índice de Diferencia de Color (CDI) es: {cdi}')

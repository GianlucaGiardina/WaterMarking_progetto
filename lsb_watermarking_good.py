from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def binarize(image_to_transform, threshold):
    output_image = image_to_transform.convert("L")
    for x in range(output_image.width):
        for y in range(output_image.height):
            if output_image.getpixel((x, y)) < threshold:
                output_image.putpixel((x, y), 0)
            else:
                output_image.putpixel((x, y), 255)
    return output_image

def watermark_image(original_image_path, watermark_path, threshold=128):
    original_image = Image.open(original_image_path).convert("L")
    new_size = (1000, 1000)
    original_image = original_image.resize(new_size)

    watermark_img = Image.open(watermark_path).convert("L")
    watermark_img = watermark_img.resize(new_size)
    watermark_img = binarize(watermark_img, threshold)

    array_original = np.array(list(original_image.getdata()))
    array_watermark = np.array(list(watermark_img.getdata()))

    for p in range(1000000):
        bin_original = bin(array_original[p])[2:-1]
        x = bin(array_watermark[p])[2]
        bin_original += x
        array_original[p] = int(bin_original, 2)

    array_original = array_original.reshape(1000, 1000)
    encoded_image = Image.fromarray(array_original.astype('uint8'), original_image.mode)
    return encoded_image

def extract_watermark(watermarked_image):
    array_watermarked = np.array(list(watermarked_image.getdata()))
    for p in range(1000000):
        if bin(array_watermarked[p])[-1] == '1':
            array_watermarked[p] = 255
        else:
            array_watermarked[p] = 0

    array_watermarked = array_watermarked.reshape(1000, 1000)
    extracted_watermark = Image.fromarray(array_watermarked.astype('uint8'), watermarked_image.mode)
    return extracted_watermark


# Esempio di utilizzo
original_path = 'images/cover2.png'
watermark_path = 'images/watermark.png'
threshold_value = 128

watermarked_image = watermark_image(original_path, watermark_path, threshold_value)
watermarked_image = np.array(watermarked_image.convert("L"))

cover_marked_blur= cv2.GaussianBlur(watermarked_image, (5, 5), 0)
cover_marked_rotated = cv2.rotate(watermarked_image, cv2.ROTATE_180)
temp_jpeg_path = 'temp_compressed_image.jpg'
jpeg_compression_level = 233
cv2.imwrite(temp_jpeg_path, watermarked_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_compression_level])
cover_marked_compressed = cv2.imread(temp_jpeg_path, cv2.IMREAD_GRAYSCALE)

watermark_estratto= extract_watermark(Image.fromarray(watermarked_image))
watermark_estratto_blur= extract_watermark(Image.fromarray(cover_marked_blur))
watermark_estratto_rotated= extract_watermark(Image.fromarray(cover_marked_rotated))
watermark_estratto_compressed= extract_watermark(Image.fromarray(cover_marked_compressed))

import matplotlib.pyplot as plt

# Immagini originali e watermark estratti
watermarked_image = Image.fromarray(watermarked_image)
cover_marked_blur = Image.fromarray(cover_marked_blur)
cover_marked_compressed = Image.fromarray(cover_marked_compressed)

# Configurazione della visualizzazione
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Mostra le immagini
axs[0].imshow(watermarked_image, cmap='gray')
axs[0].set_title('Watermarked Image')

axs[1].imshow(cover_marked_blur, cmap='gray')
axs[1].set_title('Cover Marked (Blur)')

axs[2].imshow(cover_marked_compressed, cmap='gray')
axs[2].set_title('Cover Marked (Compressed)')

# Nascondi gli assi per una migliore visualizzazione
for ax in axs:
    ax.axis('off')

# Mostra il plot
plt.show()

plt.figure(figsize=(4, 3))
plt.subplot(221), plt.imshow(watermark_estratto, cmap='gray'), plt.title('Watermark estratto'), plt.axis('off')
plt.subplot(222), plt.imshow(watermark_estratto_blur, cmap='gray'), plt.title('Watermark estr. BLUR'), plt.axis('off')
plt.subplot(223), plt.imshow(watermark_estratto_rotated, cmap='gray'), plt.title('Watermark estr. RUOTATO'), plt.axis('off')
plt.subplot(224), plt.imshow(watermark_estratto_compressed, cmap='gray'), plt.title('Watermark estr. COMPRESSO'), plt.axis('off')
plt.tight_layout()
plt.show()



plt.figure(figsize=(4, 3))
plt.subplot(221), plt.imshow(Image.open(original_path).convert("L"), cmap='gray'), plt.title('Cover'), plt.axis('off')
plt.subplot(222), plt.imshow(watermarked_image, cmap='gray'), plt.title('Cover con Watermark'), plt.axis('off')
plt.subplot(223), plt.imshow(Image.open(watermark_path).convert("L"), cmap='gray'), plt.title('Watermark'), plt.axis('off')
plt.subplot(224), plt.imshow(watermark_estratto, cmap='gray'), plt.title('Watermark Estratto'), plt.axis('off')
plt.tight_layout()
plt.show()


from PIL import Image
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import matplotlib.pyplot as plt

# Supponiamo che le variabili watermark_estratto, watermark_estratto_blur, watermark_estratto_rotated,
# watermark_estratto_compressed siano già definite come oggetti di tipo PIL.Image.Image

# Converte le immagini in array NumPy
# Ridimensiona l'immagine originale alle dimensioni di watermark_estratto
array_watermark = np.array(Image.open(original_path).convert("L"))

# Assicurati che le due immagini abbiano le stesse dimensioni
array_watermark_estratto = np.array(watermark_estratto)
array_watermark_estratto_blur = np.array(watermark_estratto_blur)
array_watermark_estratto_rotated = np.array(watermark_estratto_rotated)
array_watermark_estratto_compressed = np.array(watermark_estratto_compressed)

# Calcola MSE e PSNR tra watermark_estratto e le altre immagini
mse_std = mean_squared_error(array_watermark.flatten(), array_watermark_estratto.flatten())
psnr_std= peak_signal_noise_ratio(array_watermark, array_watermark_estratto)


mse_blur = mean_squared_error(array_watermark.flatten(), array_watermark_estratto_blur.flatten())
psnr_blur = peak_signal_noise_ratio(array_watermark, array_watermark_estratto_blur)

mse_rotated = mean_squared_error(array_watermark.flatten(), array_watermark_estratto_rotated.flatten())
psnr_rotated = peak_signal_noise_ratio(array_watermark, array_watermark_estratto_rotated)

mse_compressed = mean_squared_error(array_watermark.flatten(), array_watermark_estratto_compressed.flatten())
psnr_compressed = peak_signal_noise_ratio(array_watermark, array_watermark_estratto_compressed)

# Visualizza i risultati
print(f'MSE e PSNR tra Watermark   e Watermark Estratto : {mse_std}, {psnr_std}')
print(f'MSE e PSNR tra Watermark  e Watermark Estratto BLUR: {mse_blur}, {psnr_blur}')
print(f'MSE e PSNR tra Watermark  e Watermark Estratto RUOTATO: {mse_rotated}, {psnr_rotated}')
print(f'MSE e PSNR tra Watermark  e Watermark Estratto COMPRESSO: {mse_compressed}, {psnr_compressed}')
import pandas as pd
import matplotlib.pyplot as plt

# Crea un DataFrame con i risultati
data = {
    'Metodo': ['Watermark Estratto', 'Watermark Estratto BLUR', 'Watermark Estratto RUOTATO', 'Watermark Estratto COMPRESSO'],
    'MSE': [mse_std, mse_blur, mse_rotated, mse_compressed],
    'PSNR': [psnr_std, psnr_blur, psnr_rotated, psnr_compressed]
}

df = pd.DataFrame(data)

# Crea una figura e un asse
fig, ax = plt.subplots()

# Nascondi assi
ax.axis('off')

# Crea la tabella
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Imposta lo stile della tabella
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Aumenta la scala per migliorare la leggibilità

# Salva l'immagine della tabella
plt.savefig('risultati_tabella.png', bbox_inches='tight', pad_inches=0.5)

# Mostra l'immagine della tabella
plt.show()

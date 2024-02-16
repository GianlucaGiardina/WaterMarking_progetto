import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error

def dwt_embed(input_image, watermark_image, seed=2024):
    if len(input_image.shape) > 2 or len(watermark_image.shape) > 2:
        print("Parameter input_image should be grayscale")
        return input_image

    # Step 1: DWT in level 2 Haar coefficients cH_l2 and cV_l2
    cA_l1, (cH_l1, cV_l1, cD_l1) = pywt.dwt2(input_image.astype(np.float32), 'haar')
    cA_l2, (cH_l2, cV_l2, cD_l2) = pywt.dwt2(cA_l1, 'haar')

    # Step 2: Embed
    height, width = input_image.shape
    watermark_image = cv2.resize(watermark_image, (width >> 2, height >> 2))
    watermark_image = watermark_image.astype(np.float32)

    # change 0 to -1
    # watermark_image[watermark_image < 1] = -1
    alpha = 3  # The strength of watermark
    cH_l2 = alpha * watermark_image
    cV_l2 = alpha * watermark_image

    # Step 3: IDWT
    cA_l1 = pywt.idwt2((cA_l2, (cH_l2, cV_l2, cD_l2)), 'haar')
    marked_image = pywt.idwt2((cA_l1, (cH_l1, cV_l1, cD_l1)), 'haar')

    return marked_image.astype(np.uint8)



# Non-blind detection, requires the original watermark
def dwt_extract(marked_image, original_watermark, seed=2024):
    if len(marked_image.shape) > 2:
        print("Parameter marked_image should be grayscale")
        return marked_image

    # Step 1: DWT in level 2 Haar coefficients cH_l2 and cV_l2
    cA_l1, (cH_l1, cV_l1, cD_l1) = pywt.dwt2(marked_image.astype(np.float32), 'haar')
    cA_l2, (cH_l2, cV_l2, cD_l2) = pywt.dwt2(cA_l1, 'haar')

    # Step 2: Extract
    height, width = marked_image.shape
    original_watermark = cv2.resize(original_watermark, (width >> 2, height >> 2))
    original_watermark = original_watermark.astype(np.float32)
    # original_watermark[original_watermark < 1] = -1
    alpha = 3
    extracted_watermark = cH_l2 * original_watermark + cV_l2 * original_watermark
    extracted_watermark = 255 * extracted_watermark / np.max(extracted_watermark)
    extracted_watermark[extracted_watermark < alpha] = 0
    extracted_watermark[extracted_watermark >= alpha] = 255

    return extracted_watermark.astype(np.uint8)



if __name__ == '__main__':
    img_gray = cv2.imread('images/cover2.png', cv2.IMREAD_GRAYSCALE)

    img_watermark = cv2.imread('images/watermark.png', cv2.IMREAD_GRAYSCALE)
    _, img_watermark = cv2.threshold(img_watermark, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_marked = dwt_embed(img_gray, img_watermark, 20240211)
    cv2.imwrite('images/cover_marked.png', img_marked)


    # print(img_marked.shape, type(img_marked), type(img_marked[0,0]))
    img_stego = cv2.imread('images/cover_marked.png', cv2.IMREAD_GRAYSCALE)
    img_watermark = cv2.imread('images/watermark.png', cv2.IMREAD_GRAYSCALE)
    cover_marked  = cv2.GaussianBlur(img_stego, (5, 5), 0)
    cover_marked_rotated = cv2.rotate(img_stego, cv2.ROTATE_180)  # Puoi specificare l'angolo di rotazione
    # Carica l'immagine PNG

# Specifica il percorso in cui salvare temporaneamente l'immagine come JPEG
    temp_jpeg_path = 'temp_compressed_image.jpg'

# Specifica il livello di compressione JPEG (0 = massima compressione, 100 = nessuna compressione)
    jpeg_compression_level = 95

# Salva temporaneamente l'immagine come JPEG
    cv2.imwrite(temp_jpeg_path, img_stego, [cv2.IMWRITE_JPEG_QUALITY, jpeg_compression_level])

# Carica nuovamente l'immagine come variabile
    compressed_img = cv2.imread(temp_jpeg_path, cv2.IMREAD_GRAYSCALE)


    _, img_watermark = cv2.threshold(img_watermark, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_watermark_extracted = dwt_extract(img_stego, img_watermark, 20240211)
    img_watermark_extracted2 = dwt_extract(cover_marked, img_watermark, 20240211)
    img_watermark_extracted3 = dwt_extract(cover_marked_rotated, img_watermark, 20240211)
    img_watermark_extracted4 = dwt_extract(compressed_img, img_watermark, 20240211)

    cv2.imwrite('images/watermark-extracted.png', img_watermark_extracted)



    plt.figure(figsize=(4, 3))    
    plt.subplot(221), plt.imshow(img_gray, cmap='gray'), plt.title('Cover'), plt.axis('off')
    plt.subplot(222), plt.imshow(img_marked, cmap='gray'), plt.title('Cover con Watermark'), plt.axis('off')
    plt.subplot(223), plt.imshow(img_watermark, cmap='gray'), plt.title('Watermark'), plt.axis('off')
    plt.subplot(224), plt.imshow(img_watermark_extracted, cmap='gray'), plt.title('Watermark Estratto'), plt.axis('off')
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(4, 3))
    cv2.imwrite('images/watermark-extracted2.png', img_watermark_extracted2)
    

    plt.subplot(221), plt.imshow(img_watermark_extracted2, cmap='gray'), plt.title('Watermark estr. BLURRING'), plt.axis('off')
    plt.subplot(222), plt.imshow(img_watermark_extracted4, cmap='gray'), plt.title('Watermark estr. COMPRESSIONE'), plt.axis('off')
    plt.subplot(223), plt.imshow(img_watermark_extracted3, cmap='gray'), plt.title('Watermark estr. Rotazione'), plt.axis('off')
    plt.subplot(224), plt.imshow(img_watermark_extracted, cmap='gray'), plt.title('Watermark estratto senza attacchi'), plt.axis('off')
    plt.tight_layout()
    plt.show()

img_watermark = cv2.resize(img_watermark, (img_watermark_extracted2.shape[1], img_watermark_extracted2.shape[0]))



print("Dimensioni img_watermark:", img_watermark.shape)
print("Dimensioni img_watermark_extracted2:", img_watermark_extracted2.shape)
print("Dimensioni img_watermark_extracted4:", img_watermark_extracted4.shape)

# Controlla le dimensioni delle immagini e ridimensiona se necessario
if img_watermark.shape != img_watermark_extracted2.shape:
    print("Le dimensioni delle immagini non corrispondono. Ridimensiono img_watermark_extracted2.")
    img_watermark_extracted2 = cv2.resize(img_watermark_extracted2, (img_watermark.shape[1], img_watermark.shape[0]))

# Controlla le dimensioni anche per img_watermark_extracted4
if img_watermark.shape != img_watermark_extracted4.shape:
    print("Le dimensioni delle immagini non corrispondono. Ridimensiono img_watermark_extracted4.")
    img_watermark_extracted4 = cv2.resize(img_watermark_extracted4, (img_watermark.shape[1], img_watermark.shape[0]))

# Calcola MSE e PSNR
mse2 = mean_squared_error(img_watermark.flatten(), img_watermark_extracted2.flatten())
psnr2 = peak_signal_noise_ratio(img_watermark, img_watermark_extracted2)
mse4 = mean_squared_error(img_watermark.flatten(), img_watermark_extracted4.flatten())
psnr4 = peak_signal_noise_ratio(img_watermark, img_watermark_extracted4)
mse = mean_squared_error(img_watermark.flatten(), img_watermark_extracted.flatten())
psnr = peak_signal_noise_ratio(img_watermark, img_watermark_extracted)
mse3 = mean_squared_error(img_watermark.flatten(), img_watermark_extracted3.flatten())
psnr3 = peak_signal_noise_ratio(img_watermark, img_watermark_extracted3)
# Crea una tabella
data = {'Metodo': ['Blur', 'Compressione','ruotato', 'Nessun Attacco'],
        'MSE': [mse2, mse4, mse3,mse],
        'PSNR': [psnr2, psnr4,psnr3, psnr]}

df = pd.DataFrame(data)
print(df)
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

plt.show()





print("Dimensioni img_watermark:", img_watermark.shape)
print("Dimensioni img_watermark_extracted2:", img_watermark_extracted2.shape)
print("Dimensioni img_watermark_extracted4:", img_watermark_extracted4.shape)

# Controlla le dimensioni delle immagini e ridimensiona se necessario
if img_watermark.shape != img_watermark_extracted2.shape:
    print("Le dimensioni delle immagini non corrispondono. Ridimensiono img_watermark_extracted2.")
    img_watermark_extracted2 = cv2.resize(img_watermark_extracted2, (img_watermark.shape[1], img_watermark.shape[0]))

# Controlla le dimensioni anche per img_watermark_extracted4
if img_watermark.shape != img_watermark_extracted4.shape:
    print("Le dimensioni delle immagini non corrispondono. Ridimensiono img_watermark_extracted4.")
    img_watermark_extracted4 = cv2.resize(img_watermark_extracted4, (img_watermark.shape[1], img_watermark.shape[0]))

# Calcola MSE e PSNR
mse2 = mean_squared_error(img_watermark_extracted.flatten(), img_watermark_extracted2.flatten())
psnr2 = peak_signal_noise_ratio(img_watermark_extracted, img_watermark_extracted2)
mse4 = mean_squared_error(img_watermark_extracted.flatten(), img_watermark_extracted4.flatten())
psnr4 = peak_signal_noise_ratio(img_watermark_extracted, img_watermark_extracted4)
mse = mean_squared_error(img_watermark_extracted.flatten(), img_watermark_extracted.flatten())
psnr = peak_signal_noise_ratio(img_watermark_extracted, img_watermark_extracted)

# Crea una tabella
data = {'Metodo': ['Blur', 'Compressione', 'Nessun Attacco'],
        'MSE': [mse2, mse4, mse],
        'PSNR': [psnr2, psnr4, psnr]}

df = pd.DataFrame(data)
print(df)
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

plt.show()


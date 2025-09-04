import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.interpolate import interp1d
from PIL import Image
from skimage.io import imread, imshow
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import os
import time

start = time.process_time()


# Define os parâmetros da janela deslizante
WINDOW_WIDTH = 140  # Largura da janela em pixels
TOLERANCE = 0.9   # Tolerância para diferença entre áreas
TOLERANCE_MEDIA = 150 # Tolerancia para a media retinex
LIMIAR_STD = 70
image_path = "./IMG_3260.JPG"

output_path = "./"



def save_image(image, title="Imagem", max_width=1200, max_height=800):
    global output_path

    # Garante que o diretório exista
    os.makedirs(output_path, exist_ok=True)

    # Normaliza para uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Converte RGB -> BGR se necessário
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # Verifica arquivos já existentes para numerar corretamente
    existing_files = [f for f in os.listdir(output_path) if f.startswith(title) and f.endswith(".png")]
    counter = len(existing_files) + 1
    filename = f"{title}_{counter}.png"
    filepath = os.path.join(output_path, filename)

    # Salva a imagem
    cv2.imwrite(filepath, image)
    print(f"Imagem salva em: {filepath}")

def exibir_limites_componentes(image, labels, stats, text = 'Bouding boxes'):

    # Se a imagem estiver em escala de cinza, converte para RGB para poder desenhar em cores.
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_color = image.copy()

    num_labels = stats.shape[0]



    # Iterar sobre cada componente, ignorando o fundo (índice 0)
    for i in range(1, num_labels):
        color = (255, 0, 0)        # Obter os limites a partir dos stats:
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # Desenhar o retângulo delimitador
        cv2.rectangle(image_color, (x, y), (x + w, y + h),color, 12)

def manter_somente_componentes(labels, stats, comp1, comp2):
    
    # Criar máscara que mantém apenas os componentes desejados
    mascara_valida = np.isin(labels, [comp1, comp2])

    # Criar nova matriz de labels zerada e reindexar os componentes
    labels_filtrados = np.zeros_like(labels, dtype=np.int32)

    # Reindexar para garantir que os dois componentes mantêm suas informações
    labels_filtrados[labels == comp1] = comp1  # Novo índice para comp1
    labels_filtrados[labels == comp2] = comp2  # Novo índice para comp2

    # Garantir que o array de stats tenha pelo menos o número de componentes que você vai manter
    stats_filtrados = np.zeros((max(comp1, comp2) + 1, stats.shape[1]), dtype=stats.dtype)  # Tamanho ajustado
    stats_filtrados[comp1] = stats[comp1]  # Manter stats de comp1
    stats_filtrados[comp2] = stats[comp2]  # Manter stats de comp2
    return labels_filtrados, stats_filtrados

def atualizar_estatisticas(stats, comp_id, deslocamento_x, deslocamento_y):
    stats[comp_id, cv2.CC_STAT_LEFT] += deslocamento_x
    stats[comp_id, cv2.CC_STAT_TOP] += deslocamento_y
    return stats

def alinhar_horizontalmente(comp_img, labels, stats, comp_id, mask):
    image = comp_img.copy()

    # Encontrar contornos da máscara do componente
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError(f"Não foi possível encontrar contornos para o componente {comp_id}")

    # Calcular a orientação principal usando momentos
    M = cv2.moments(contours[0])
    if M["mu20"] == 0:
        return comp_img, stats, labels

    angle = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
    angle = np.degrees(angle)

    h, w = comp_img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotacionar apenas a máscara do componente
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    img_rotated = cv2.warpAffine(comp_img, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    # Criar uma nova matriz de labels preservando os outros componentes
    new_labels = labels.copy()
    new_labels[mask > 0] = 0  # Remover componente original
    new_labels[rotated_mask > 0] = comp_id  # Adicionar o componente rotacionado

    # Recalcular bounding box do componente
    contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        stats[comp_id] = [x, y, w, h, stats[comp_id, cv2.CC_STAT_AREA]]  # Mantém a área original

    return img_rotated,  stats, new_labels

def exibe(image, title = "Imagem", max_width=1200, max_height=800):


    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Redimensiona automaticamente se for maior que o limite
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0) 
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size)
    
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def verificar_papelao_por_saturacao(imagem_rgb, mostrar_mascara=False):
    imagem_hsv = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2HSV)
    saturacao = imagem_hsv[:, :, 1]
    # Threshold baseado na saturação observada (~intensidade média do papelão)
    _, mascara_papelao = cv2.threshold(saturacao, 100, 255, cv2.THRESH_BINARY)

    altura, largura = saturacao.shape
    total_pixels = altura * largura
    # Proporção de pixels considerados "papelão"
    papelao_pixels = cv2.countNonZero(mascara_papelao)
    proporcao = papelao_pixels / total_pixels if total_pixels > 0 else 0

    #if mostrar_mascara:
    #    exibe(mascara_papelao, 'mascara_papelao')
    #print(f"Pixels com saturação alta ({proporcao*100:.2f}%)")
    return ((proporcao > 0.02), mascara_papelao)  # threshold ajustável


def carregar_imagem(path):

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Verificar dimensões
    altura, largura = image.shape[:2]
    # Se a altura for maior que a largura, girar 180 graus
    if altura > largura:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return image


def transformar_em_hsv(image):
    image_original_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    

    return image_original_hsv



def encontrar_Contornos(image, hsv):
    hue_channel = hsv[:, :, 0]

    mask = cv2.bitwise_not(cv2.inRange(hue_channel, 100, 150)) #Lembrando que a faixa o Hue não vai ate 255, mas sim até 179

    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criar imagem para exibição dos resultados
    segmented_image = image.copy()
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 10)

    # Identificar componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned)

    return num_labels, labels, stats, centroids, segmented_image


def segmenta(image, image_original_hsv):
    
    """#CAMERA CANON"""
    num_labels_original, labels_original, stats_original, centroids_original, segmented_image = encontrar_Contornos(image, image_original_hsv)


    num_labels = num_labels_original
    labels = labels_original.copy()
    stats = stats_original.copy()
    centroids = centroids_original.copy()

    image_BB = exibir_limites_componentes(image, labels, stats)

    # Ordenar componentes pelo tamanho (área), ignorando o fundo (índice 0)
    componentes_ordenados = sorted(range(1, num_labels_original), key=lambda i: stats_original[i, cv2.CC_STAT_AREA], reverse=True)

    # Selecionar os dois maiores componentes
    if len(componentes_ordenados) < 2:
        raise ValueError("Não foram encontrados dois componentes para alinhar.")

    comp1, comp2 = componentes_ordenados[:2]  # Índices dos dois maiores componentes
    # Certifique-se de que comp1 esteja acima de comp2 na imagem
    if stats_original[comp1, cv2.CC_STAT_TOP] > stats_original[comp2, cv2.CC_STAT_TOP]:
        # Troca os índices se comp1 estiver abaixo de comp2
        comp1, comp2 = comp2, comp1
    # Criar máscara dos componentes selecionados
    mask_comp1 = (labels_original == comp1).astype(np.uint8) * 255
    mask_comp2 = (labels_original == comp2).astype(np.uint8) * 255
    mask_comp_combined = cv2.bitwise_or(mask_comp1, mask_comp2)


    # Criar imagens temporárias apenas com os componentes
    comp1_img = np.zeros_like(image)
    comp2_img = np.zeros_like(image)
    comps_img = np.zeros_like(image)

    comp1_img[mask_comp1 == 255] = image[mask_comp1 == 255]
    comp2_img[mask_comp2 == 255] = image[mask_comp2 == 255]
    comps_img[mask_comp_combined == 255] = image[mask_comp_combined == 255]
    labels_original, stats_original = manter_somente_componentes(labels_original, stats_original, comp1, comp2)
    exibir_limites_componentes(comps_img, labels_original, stats_original, "original")
    labels_original, stats_original = manter_somente_componentes(labels_original, stats_original, comp1, comp2)
    return num_labels_original, labels_original, stats_original,num_labels,labels,stats, comp1, comp2, mask_comp1, mask_comp2, comp1_img, comp2_img



def alinhamento( labels_original, stats_original, comp1, comp2, mask_comp1, mask_comp2, comp1_img, comp2_img):
    # Aplicar operação para alinhar horizontalmente cada componente
    comp1_alinhado_img, stats_horizontal, labels_horizontal = alinhar_horizontalmente( comp1_img,labels_original, stats_original, comp1,mask_comp1)
    comp2_alinhado_img, stats_horizontal,labels_horizontal = alinhar_horizontalmente( comp2_img, labels_original, stats_original, comp2,mask_comp2)

    #mascara com uniao dos componentes alinhados
    mascara_unida = cv2.bitwise_or(comp1_alinhado_img, comp2_alinhado_img)

    #filtra os labels pra tirar os outros componentes
    labels_filtrados, stats_filtrados = manter_somente_componentes(labels_horizontal, stats_horizontal, comp1, comp2)

    # Obter as coordenadas x dos componentes
    x1 = stats_filtrados[comp1, cv2.CC_STAT_LEFT]
    x2 = stats_filtrados[comp2, cv2.CC_STAT_LEFT]

    # Garantir que sempre deslocamos o que está mais à direita em direção ao mais à esquerda
    if x1 > x2:
        comp_direita, comp_esquerda = comp1, comp2
    else:
        comp_direita, comp_esquerda = comp2, comp1

    # Calcular deslocamentos
    deslocamento_x = stats_filtrados[comp_direita, cv2.CC_STAT_LEFT] - stats_filtrados[comp_esquerda, cv2.CC_STAT_LEFT]
    deslocamento_y = (stats_filtrados[comp1, cv2.CC_STAT_TOP] + stats_filtrados[comp1, cv2.CC_STAT_HEIGHT]) - stats_filtrados[comp2, cv2.CC_STAT_TOP]

    mascara_comp2_alinhada = (labels_filtrados == comp2).astype(np.uint8)
    labels_alinhados = labels_filtrados.copy()
    labels_alinhados[labels_filtrados == comp2] = 0
    mascara_comp2_deslocada = np.zeros_like(labels_filtrados).astype(np.uint8)

    for y in range(labels_filtrados.shape[0]):
        for x in range(labels_filtrados.shape[1]):
            if mascara_comp2_alinhada[y, x] == 1:
                novo_y, novo_x = y + deslocamento_y, x + deslocamento_x
                if 0 <= novo_y < labels_filtrados.shape[0] and 0 <= novo_x < labels_filtrados.shape[1]:
                    mascara_comp2_deslocada[novo_y, novo_x] = 200


    labels_alinhados[mascara_comp2_deslocada == 200] = comp2


    # Criar uma cópia da imagem original para preservar comp1
    imagem_alinhada = comp1_alinhado_img.copy()

    # Extrair bounding box do componente comp2
    x2, y2, w2, h2, _ = stats_original[comp2]

    imagem_alinhada[labels_alinhados == comp2] = comp2_alinhado_img[labels_filtrados == comp2]
    stats_alinhados = atualizar_estatisticas(stats_filtrados, comp2, deslocamento_x, deslocamento_y)


    exibir_limites_componentes(imagem_alinhada,labels_alinhados,stats_alinhados)
    return stats_alinhados, comp_direita,comp_esquerda, imagem_alinhada, labels_alinhados, stats_filtrados, comp1_alinhado_img, comp2_alinhado_img, labels_filtrados

def substituir_fundo(imagem_alinhada):
    # Carregar o patch de fundo real
    patch_fundo = cv2.imread("Selagem/fundo_1.jpg")
    patch_fundo = cv2.cvtColor(patch_fundo, cv2.COLOR_BGR2RGB)

    # Redimensiona o patch de fundo para cobrir toda a imagem alinhada
    h_img, w_img = imagem_alinhada.shape[:2]
    h_patch, w_patch = patch_fundo.shape[:2]

    # Repetir o patch para cobrir o fundo da imagem alinhada
    fundo_texturizado = np.tile(patch_fundo, (int(np.ceil(h_img / h_patch)), int(np.ceil(w_img / w_patch),), 1))
    fundo_texturizado = fundo_texturizado[:h_img, :w_img]  # corta o excesso

    # Criar máscara onde o fundo é preto (ou quase preto, por segurança)
    mascara_preto = np.all(imagem_alinhada <= [1, 1, 1], axis=2)  # tolerância pequena

    # Substituir apenas os pixels pretos pelo fundo texturizado
    imagem_alinhada[mascara_preto] = fundo_texturizado[mascara_preto]


    #exibe(imagem_alinhada, 'imagem_alinhada')
    return imagem_alinhada

def linha_superior_inferior(image,image_original_hsv):
    ####---ALINHAMENTO---###
    num_labels_original, labels_original, stats_original,num_labels,labels,stats, comp1, comp2, mask_comp1, mask_comp2, comp1_img, comp2_img = segmenta(image, image_original_hsv)
    stats_alinhados, comp_direita,comp_esquerda, imagem_alinhada_sem_fundo, labels_alinhados, stats_filtrados, comp1_alinhado_img, comp2_alinhado_img, labels_filtrados = alinhamento(labels_original, stats_original, comp1, comp2, mask_comp1, mask_comp2, comp1_img, comp2_img)


    exibir_limites_componentes(image, labels, stats)
    exibir_limites_componentes(imagem_alinhada_sem_fundo, labels_alinhados, stats_filtrados)
    # Extrair bounding boxes
    x1_min = stats_alinhados[comp1, cv2.CC_STAT_LEFT]
    y1_min = stats_alinhados[comp1, cv2.CC_STAT_TOP]
    w1 = stats_alinhados[comp1, cv2.CC_STAT_WIDTH]
    h1 = stats_alinhados[comp1, cv2.CC_STAT_HEIGHT]
    x1_max = x1_min + w1
    y1_max = y1_min + h1

    x2_min = stats_alinhados[comp2, cv2.CC_STAT_LEFT]
    y2_min = stats_alinhados[comp2, cv2.CC_STAT_TOP]
    w2 = stats_alinhados[comp2, cv2.CC_STAT_WIDTH]
    h2 = stats_alinhados[comp2, cv2.CC_STAT_HEIGHT]
    x2_max = x2_min + w2
    y2_max = y2_min + h2

    # Descobrir quem está "em cima" (menor y) e quem está "embaixo" (maior y)
    if y1_min < y2_min:
        # comp1 está em cima, comp2 está embaixo
        y_top = y1_max   # parte inferior do de cima
        y_bottom = y2_min  # parte superior do de baixo
        x_min = max(x1_min, x2_min)
        x_max = min(x1_max, x2_max)
    else:
        # comp2 está em cima, comp1 está embaixo
        y_top = y2_max
        y_bottom = y1_min
        x_min = max(x1_min, x2_min)
        x_max = min(x1_max, x2_max)

    # Ponto médio em Y
    y_mid = int((y_top + y_bottom) / 2)

    # Criar pontos da linha média
    points_central_line = np.array([[x, y_mid] for x in range(x_min, x_max + 1)], dtype=np.float32)



    # Verifica se nenhum pixel branco foi encontrado
    if len(points_central_line) == 0:
        print("Nenhum pixel branco encontrado na imagem 'and_image'.")
    else:

        # Separa as coordenadas em X e y para realizar a regressão (ajustando y em função de x)
        X = points_central_line[:, 0].reshape(-1, 1)  # Extrai a coluna x e reformata para matriz 2D
        y = points_central_line[:, 1]                 # Extrai a coluna y

        # Define o grau do polinômio desejado (exemplo: 2 para uma curva quadrática) para o RANSAC
        grau = 2

        # Cria um pipeline que transforma as características para polinomiais e depois aplica regressão linear
        modelo_pipeline = make_pipeline(PolynomialFeatures(grau), LinearRegression())

        # Instancia o RANSAC com o pipeline definido, usando min_samples igual a grau+1 para garantir dados suficientes
        ransac = RANSACRegressor(estimator=modelo_pipeline, min_samples=grau+1, random_state=42)
        ransac.fit(X, y)  # Ajusta o modelo aos dados (X, y)

        # Gera uma sequência de valores de X para plotar a curva ajustada de forma suave
        x_range_central_line = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)  # 500 pontos entre o mínimo e o máximo de X
        y_range = ransac.predict(x_range_central_line)  # Prediz os valores de y para cada valor em x_range_central_line usando o modelo RANSAC

        # Cria uma cópia da imagem original para desenhar a curva sem alterá-la
        line_image = imagem_alinhada_sem_fundo.copy()

        # Percorre os pontos gerados para desenhar a curva conectando pontos consecutivos
        for i in range(len(x_range_central_line) - 1):
            # Define o primeiro ponto (pt1) com coordenadas convertidas para inteiros
            pt1 = (int(x_range_central_line[i, 0]), int(y_range[i]))
            # Define o segundo ponto (pt2) correspondente ao próximo par (x, y)
            pt2 = (int(x_range_central_line[i + 1, 0]), int(y_range[i + 1]))
            # Desenha uma linha entre pt1 e pt2 na imagem, com cor verde (0, 255, 0) e espessura 2
            cv2.line(line_image, pt1, pt2, (255, 0, 0), 10)

    _,img_masked = verificar_papelao_por_saturacao(imagem_alinhada_sem_fundo, True)

    # Criação de um kernel elíptico para operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Aplica a operação de abertura (morphological open) para remover ruídos pequenos  Limpar pequenas formas e manter grandes
    cleaned_cardboard = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel)

    # Aplica a operação de fechamento (morphological close) para preencher buracos na máscara
    closed_cardboard = cv2.morphologyEx(cleaned_cardboard, cv2.MORPH_CLOSE, kernel)

    # Aplica um desfoque gaussiano para suavizar a máscara tratada
    blurred_image = cv2.GaussianBlur(closed_cardboard, (11, 11), 0)

    # Detecta bordas na imagem usando o detector de Canny
    edges_cardboard = cv2.Canny(blurred_image, 80, 100)
    central_points = [(x_range_central_line[i, 0], y_range[i]) for i in range(len(x_range_central_line))]  #points fica ordenado porque x_range é ordenado la na função np.linspace(X.min(), X.max(), 500)

    # Inicializa as imagens onde os pontos serão marcados em branco (255)
    superior_line = np.zeros_like(edges_cardboard, dtype=np.uint8)
    inferior_line = np.zeros_like(edges_cardboard, dtype=np.uint8)

    for (x, y_central) in central_points:
        x = int(x)
        y_central = int(y_central)

        # Verifica se x está dentro dos limites da imagem
        if x < 0 or x >= edges_cardboard.shape[1]:
            continue

        # ---- Pontos brancos acima da linha central ----
        coluna = edges_cardboard[:y_central, x]
        brancos_acima = np.where(coluna == 255)[0]
        if brancos_acima.size > 0:
            y_top = brancos_acima[0]
            # Marca com um valor alto (branco) para visualização
            cv2.circle(superior_line, (x, y_top), 10, 255, -1)

        # ---- Pontos brancos abaixo da linha central ----
        coluna = edges_cardboard[y_central:, x]
        brancos_abaixo = np.where(coluna == 255)[0]
        if brancos_abaixo.size > 0:
            y_bottom = y_central + brancos_abaixo[-1]
            cv2.circle(inferior_line, (x, y_bottom), 10, 255, -1)

    # Extrai as coordenadas dos pixels brancos (valor 255) nas imagens 'superior_line' e 'inferior_line'
    points_superior_line = np.argwhere(superior_line == 255)  # Obtém as coordenadas dos pixels brancos na linha superior
    points_inferior_line = np.argwhere(inferior_line == 255)  # Obtém as coordenadas dos pixels brancos na linha inferior

    # Verifica se alguma das linhas não possui pixels brancos
    if not len(points_superior_line) or not len(points_inferior_line):
        print("Nenhum pixel branco encontrado na margem")  # Mensagem de erro caso não haja pontos brancos
    else:
        # Inverte a ordem das coordenadas de (y, x) para (x, y) e converte para float32
        points_superior_line = points_superior_line[:, [1, 0]].astype(np.float32)
        points_inferior_line = points_inferior_line[:, [1, 0]].astype(np.float32)

        # Separa as coordenadas X e Y para a regressão polinomial
        X_superior_line = points_superior_line[:, 0].reshape(-1, 1)  # Extrai X da linha superior e converte para formato 2D
        y_superior_line = points_superior_line[:, 1]  # Extrai Y da linha superior
        X_inferior_line = points_inferior_line[:, 0].reshape(-1, 1)  # Extrai X da linha inferior e converte para formato 2D
        y_inferior_line = points_inferior_line[:, 1]  # Extrai Y da linha inferior

        # Define o grau do polinômio desejado (exemplo: 2 para uma curva quadrática)
        grau = 2
        modelo_pipeline = make_pipeline(PolynomialFeatures(grau), LinearRegression())  # Cria um pipeline com regressão polinomial

        # Cria o modelo RANSAC para a linha superior
        ransac_superior = RANSACRegressor(estimator=modelo_pipeline, min_samples=grau+1, random_state=42)
        ransac_superior.fit(X_superior_line, y_superior_line)  # Ajusta o modelo aos dados da linha superior

        # Cria o modelo RANSAC para a linha inferior
        ransac_inferior = RANSACRegressor(estimator=modelo_pipeline, min_samples=grau+1, random_state=42)
        ransac_inferior.fit(X_inferior_line, y_inferior_line)  # Ajusta o modelo aos dados da linha inferior

        # Gera uma sequência de valores de X para plotar a curva ajustada da linha superior
        x_range_superior_line = np.linspace(X_superior_line.min(), X_superior_line.max(), 500).reshape(-1, 1)
        y_range_superior_line = ransac_superior.predict(x_range_superior_line)  # Prediz os valores de Y correspondentes

        # Gera uma sequência de valores de X para plotar a curva ajustada da linha inferior
        x_range_inferior_line = np.linspace(X_inferior_line.min(), X_inferior_line.max(), 500).reshape(-1, 1)
        y_range_inferior_line = ransac_inferior.predict(x_range_inferior_line)  # Prediz os valores de Y correspondentes


    # Criar listas de pontos (x, y) para as linhas superior e inferior
    superior_points = []
    for i in range(len(x_range_superior_line)):
        superior_points.append((x_range_superior_line[i, 0], y_range_superior_line[i]))

    inferior_points = []
    for i in range(len(x_range_inferior_line)):
        inferior_points.append((x_range_inferior_line[i, 0], y_range_inferior_line[i]))

    # Criar funções de interpolação linear (para retornar y dado x)
    x_superior = []
    y_superior_vals = []
    for p in superior_points:
        x_superior.append(p[0])
        y_superior_vals.append(p[1])
    interp_superior = interp1d(x_superior, y_superior_vals, kind='linear', fill_value="extrapolate")

    x_inferior = []
    y_inferior_vals = []
    for p in inferior_points:
        x_inferior.append(p[0])
        y_inferior_vals.append(p[1])
    interp_inferior = interp1d(x_inferior, y_inferior_vals, kind='linear', fill_value="extrapolate")

    x_center = []
    y_center_vals = []
    for p in central_points:
        x_center.append(p[0])
        y_center_vals.append(p[1])
    interp_center = interp1d(x_center, y_center_vals, kind='linear', fill_value="extrapolate")

    # Visualização das linhas ajustadas (superior, inferior e central)
    segments_image = imagem_alinhada_sem_fundo.copy()

    for points, color in [(superior_points, (0, 255, 0)), (inferior_points, (0, 255, 0)), (central_points, (255, 0, 0))]:
        for i in range(len(points) - 1):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
            cv2.line(segments_image, pt1, pt2, color, 10)
    #exibe(segments_image)
    return imagem_alinhada_sem_fundo, central_points, interp_superior, interp_inferior, interp_center

def single_scale_retinex(img, sigma):
    """ Aplica Retinex de escala única """
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log1p(img) - np.log1p(blur)
    return retinex

def multi_scale_retinex(img, sigmas=[15, 80, 250]):
    """ Aplica Multi-Scale Retinex """
    img = img.astype(np.float32) + 1e-6  # Evita log(0)
    retinex = np.zeros_like(img)

    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)

    retinex = retinex / len(sigmas)  # Média das escalas
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)

    return retinex.astype(np.uint8)


def media_sem_zeros(arr):


    arr_filtrado = arr[arr != 0]  # Filtra os valores diferentes de zero

    return np.mean(arr_filtrado) if arr_filtrado.size > 0 else 0  # Evita erro com array vazio

def calcaDivSoma(test_image, central_points, interp_superior, interp_inferior, interp_center ):

    x_coords = np.array([p[0] for p in central_points])
    y_coords = np.array([p[1] for p in central_points])


    # Interpolar os valores de y para cada x
    x_range = np.arange(x_coords[0], x_coords[-1])
    y_superior = interp_superior(x_range)
    y_inferior = interp_inferior(x_range)
    y_center = interp_center(x_range) #usada depois na parte de passar a janela deslizante


    # Percorre a região ao longo da linha central usando uma janela deslizante
    for x_start in range(int(x_coords[0]), int(x_coords[-1]) - WINDOW_WIDTH, WINDOW_WIDTH):
        x_end = x_start + WINDOW_WIDTH  # Define o limite final da janela
        sum_above = 0  # Soma dos valores da região acima da linha central
        sum_below = 0  # Soma dos valores da região abaixo da linha central

        # Itera sobre os valores de x dentro da janela atual
        for x in range(x_start, x_end):
            # Verifica se x está dentro do intervalo válido
            if x < x_range[0] or x >= x_range[-1]:
                continue  # Ignora valores fora do intervalo

            # Encontra o índice correspondente a x no array x_range
            idx = np.searchsorted(x_range, x)

            # Obtém os valores interpolados de y para os limites superior, inferior e a linha central
            y_top = int(y_superior[idx])  # Novo limite superior
            y_bottom = int(y_inferior[idx])  # Novo limite inferior
            y_mid = int(y_center[idx])  # Linha central interpolada

            # Garante que o limite superior esteja acima do inferior
            if y_top >= y_bottom:
                continue  # Evita regiões inválidas

            # Seleciona a região acima da linha central
            region_above = test_image[y_top:y_mid, x] if y_top < y_mid else None

            # Seleciona a região abaixo da linha central
            region_below = test_image[y_mid:y_bottom, x] if y_mid < y_bottom else None

            if region_above is not None and region_above.size > 0:
                denom_above = np.max(region_above[:, 1])
                if denom_above != 0:
                    sum_above += np.sum(region_above[:, 0] / denom_above)

            if region_below is not None and region_below.size > 0:
                denom_below = np.max(region_below[:, 1])
                if denom_below != 0:
                    sum_below += np.sum(region_below[:, 0] / denom_below)


        total = sum_above + sum_below
        if total == 0:
            continue

        difference = abs(sum_above - sum_below) / total
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        window = gray[y_top:y_bottom, x_start:x_end]

        retinex_upper = multi_scale_retinex(window[0:y_mid - y_top, 0:x_end - x_start])
        retinex_lower = multi_scale_retinex(window[y_mid - y_top:y_bottom - y_top, 0:x_end - x_start])

        mean_upper = media_sem_zeros(retinex_upper)
        std_upper  = np.std(retinex_upper)

        mean_lower = media_sem_zeros(retinex_lower)
        std_lower  = np.std(retinex_lower)

        if difference < TOLERANCE:
            if (mean_upper > TOLERANCE_MEDIA and std_upper > LIMIAR_STD) and (mean_lower > TOLERANCE_MEDIA and std_lower > LIMIAR_STD):
                cv2.rectangle(test_image, (x_start, y_top), (x_end, y_bottom), (0, 255, 0), 2)






def main():
    image = carregar_imagem(image_path)
    image_original_hsv = transformar_em_hsv(image)
    imagem_alinhada_sem_fundo, central_points, interp_superior, interp_inferior, interp_center = linha_superior_inferior(image, image_original_hsv)
    test_image = imagem_alinhada_sem_fundo.copy()
    calcaDivSoma(test_image, central_points, interp_superior, interp_inferior, interp_center)
    save_image(test_image)
    #exibe(test_image)

if __name__ == "__main__":
    main()

end = time.process_time()
print(f"Tempo de CPU: {end - start:.4f} segundos")


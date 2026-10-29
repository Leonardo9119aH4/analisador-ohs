import os
import sys
if sys.platform.startswith("linux"): 
    os.environ["QT_QPA_PLATFORM"] = "xcb" # Força o OpenCV a usar o XCB para evitar erros de interface no Linux
import cv2
import pandas as pd
import numpy as np

video_path = "video_pendulo.mp4" # Vídeo de entrada
output_csv = "dados_pendulo.csv" # Planilha de saída

# Intervalos de HSV da cor do pêndulo
lower_color1 = np.array([0, 100, 50])
upper_color1 = np.array([10, 255, 255])
lower_color2 = np.array([170, 100, 50])
upper_color2 = np.array([180, 255, 255])

cap = cv2.VideoCapture(video_path) 
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo '{video_path}'")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps} | Total de quadros: {total_frames}")
print(f"A processar o vídeo... Pressione 'q' para interromper")

# variáveis da janela, dos dados extraídos e contador de frames
window_name = "Rastreamento do pêndulo"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)
dados = []
frame_count = 0

# Processamento do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Fim do vídeo
    tempo = frame_count / fps 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Converte o quadro de BGR para HSV
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1) # Cria as 2 máscaras de cor para o pêndulo e as junta
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
    mask = cv2.bitwise_or(mask1, mask2) # Funde as 2 máscaras
    kernel = np.ones((5, 5), np.uint8) # Limpeza de ruídos na máscara
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Busca os contornos do pêndulo
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea) # Pega o maior contorno encontrado
        M = cv2.moments(c) # Calcula o centro geométrico (posição x e y)
        if M["m00"] > 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            dados.append({ # Salva os dados na lista
                "tempo_s": tempo,
                "pos_x": cX,
                "pos_y": cY
            })
            cv2.circle(frame, (round(cX), round(cY)), 8, (0, 255, 0), -1) # Desenha um círculo verde no ponto rastreado para visualização
    # Mostra o vídeo na tela com o rastreamento em tempo real
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processamento interrompido, gerando planilha...")
        break 
    frame_count += 1
# Salva a planilha do pêndulo rastreado
cap.release()
cv2.destroyAllWindows()
df = pd.DataFrame(dados)
df.to_csv(output_csv, index=False)
print(f"Processamento concluído com sucesso!")
print(f"Foram coletados {len(df)} pontos de dados.")
print(f"Os resultados foram salvos no ficheiro: '{output_csv}'")
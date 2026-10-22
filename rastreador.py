import cv2
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURAÇÕES
# ==========================================
video_path = "video_pendulo.mp4"
output_csv = "dados_pendulo.csv"

# Limites de cor em HSV para rastrear o marcador do pêndulo.
# O padrão abaixo busca um objeto VERMELHO brilhante.
# Se o seu pêndulo tiver outra cor, você precisará ajustar esses valores.
lower_color = np.array([0, 120, 70])
upper_color = np.array([10, 255, 255])

# ==========================================
# INICIALIZAÇÃO
# ==========================================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo '{video_path}'. Verifique se ele está na mesma pasta.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Vídeo carregado com sucesso!")
print(f"FPS: {fps} | Total de quadros: {total_frames}")
print("Processando... Pressione 'q' na janela do vídeo se quiser interromper antes do fim.\n")

dados = []
frame_count = 0

# ==========================================
# LOOP DE PROCESSAMENTO DO VÍDEO
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Fim do vídeo
    
    # Calcula o tempo atual em segundos com base no quadro e no FPS
    tempo = frame_count / fps
    
    # Converte a imagem de BGR para o padrão HSV (melhor para isolar cores)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Cria uma máscara binária destacando apenas a cor definida
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Remove ruídos visuais pequenos da máscara
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Busca os contornos dos objetos encontrados na máscara
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Pega o maior contorno (assume-se que seja a massa do pêndulo)
        c = max(contours, key=cv2.contourArea)
        
        # Calcula o centro geométrico (centróide) do contorno
        M = cv2.moments(c)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Salva as informações na lista
            dados.append({
                "tempo_s": round(tempo, 4),
                "pos_x": cX,
                "pos_y": cY
            })
            
            # Desenha um círculo verde no ponto rastreador para visualização
            cv2.circle(frame, (cX, cY), 8, (0, 255, 0), -1)
    
    # Mostra o vídeo na tela com o rastreamento em tempo real
    cv2.imshow("Rastreamento do Pendulo", frame)
    
    # Interrompe se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processamento interrompido pelo usuário.")
        break
        
    frame_count += 1

# ==========================================
# FINALIZAÇÃO E SALVAMENTO
# ==========================================
cap.release()
cv2.destroyAllWindows()

# Transforma a lista de dados em uma tabela e salva como CSV
df = pd.DataFrame(dados)
df.to_csv(output_csv, index=False)

print(f"Processamento concluído com sucesso!")
print(f"Foram coletados {len(df)} pontos de dados.")
print(f"Os resultados foram salvos no arquivo: '{output_csv}'")
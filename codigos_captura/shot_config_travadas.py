import gphoto2 as gp
import cv2
import os
import numpy as np
from datetime import datetime
import sys
import time

# --- 1. CONFIGURAÇÕES A SEREM TRAVADAS ---
CONFIGS_PARA_TRAVAR = {
    # --- Configs de Pós-Processamento ---
    'picturestyle': 'Neutral',  
    'alomode': 'Disable',       
    'colorspace': 'sRGB',       
    
    # --- Config de Foco ---
    'focusmode': 'Manual Focus',
    
    # --- Configs de Exposição ---
    'imageformat': 'JPEG grande fino',
    'whitebalance': 'Daylight',
    'iso': '100',
    'aperture': '8',
    'shutterspeed': '1/60',
    
    # --- Configs do Live View ---
    'liveviewsize': 'Large'
}

# --- 2. CONFIGURAÇÕES DO SCRIPT (DO SEU CÓDIGO) ---
CAPTURE_DIR = 'FOTOS_CAPTURADAS' 

camera = None
context = None

# Globais para controle de foco (do seu código)
current_near_step = 1
current_far_step = 1

# Dicionário para salvar o log
configs_log = {}

# -----------------------------------------------------------------
# FUNÇÕES AUXILIARES (DO SEU CÓDIGO)
# -----------------------------------------------------------------

def apply_config(camera, context, cfg, retries=3, wait=0.15):
    """(Sua função original)"""
    for _ in range(retries):
        try:
            camera.set_config(cfg, context)
            time.sleep(wait)
            return True
        except gp.GPhoto2Error:
            time.sleep(wait)
    return False

def desligar_liveview():
    """(Sua função original)"""
    global camera, context
    try:
        cfg = camera.get_config(context)
        vf = cfg.get_child_by_name('viewfinder')
        if vf and vf.get_value() == 1: 
            vf.set_value(0)
            return apply_config(camera, context, cfg)
    except gp.GPhoto2Error:
        pass
    return False

def ligar_liveview():
    """(Sua função original)"""
    global camera, context
    try:
        cfg = camera.get_config(context)
        vf = cfg.get_child_by_name('viewfinder')
        if vf and vf.get_value() == 0:
            vf.set_value(1)
            return apply_config(camera, context, cfg)
    except gp.GPhoto2Error:
        pass
    return False

def set_widget(camera, context, name, value, retries=3, wait=0.15):
    """(Sua função original, adaptada para logar)"""
    global configs_log 
    
    for _ in range(retries):
        try:
            cfg = camera.get_config(context)
            w = cfg.get_child_by_name(name)
            if w is None:
                print(f"Erro: Config '{name}' não encontrada.")
                return False
            
            w.set_value(value)
            
            if not apply_config(camera, context, cfg):
                time.sleep(wait)
                continue
            
            cfg2 = camera.get_config(context)
            w2 = cfg2.get_child_by_name(name)
            if w2 is None:
                return False
            
            current = w2.get_value()
            
            # Salva o valor lido no log
            # (Não salvamos o 'manualfocusdrive' no log para não poluir)
            if name != 'manualfocusdrive':
                configs_log[name] = current
            
            if str(value).lower() in str(current).lower() or str(current).lower() in str(value).lower():
                print(f"Config '{name}' definida/travada em: '{current}'")
                return True
        except gp.GPhoto2Error:
            time.sleep(wait)
    
    print(f"FALHA ao definir '{name}' para '{value}'.")
    return False

# -----------------------------------------------------------------
# FUNÇÃO PARA TRAVAR AS CONFIGURAÇÕES (MODIFICADA)
# -----------------------------------------------------------------
def travar_configuracoes_manuais():
    """
    Chama a sua função 'set_widget' para cada item no 
    dicionário CONFIGS_PARA_TRAVAR.
    """
    global camera, context, configs_log
    
    print("--- 1. Travando Configurações da Câmera (Modo M) ---")
    
    # 1. Trava as configurações do dicionário
    for config_nome, config_valor in CONFIGS_PARA_TRAVAR.items():
        set_widget(camera, context, config_nome, config_valor)
        
    # 2. Imprime o valor do Zoom do Live View
    try:
        cfg = camera.get_config(context)
        widget = cfg.get_child_by_name('eoszoomposition')
        valor = widget.get_value()
        configs_log['liveview_zoom_detectado'] = valor
        print(f"Zoom do Live View detectado: '{valor}' (Salvo no log)")
    except gp.GPhoto2Error:
        print("Aviso: Não foi possível ler 'eoszoomposition'.")
    
    print("--- Configurações da Câmera Prontas. ---")

def salvar_log_config():
    """
    Salva o dicionário 'configs_log' em um arquivo de texto.
    """
    global configs_log
    
    if configs_log:
        log_filepath = os.path.join(CAPTURE_DIR, "captura_config.txt")
        try:
            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write(f"--- Configurações de Captura em {datetime.now()} ---\n")
                for key, value in configs_log.items():
                    f.write(f"{key}: {value}\n")
            print(f"Log de configurações salvo em: {log_filepath}")
        except Exception as e:
            print(f"Erro ao salvar o log de configurações: {e}")

# -----------------------------------------------------------------
# FUNÇÕES DE CONTROLE E CAPTURA (DO SEU CÓDIGO)
# -----------------------------------------------------------------

def execute_command(key):
    """(Sua função original)"""
    global current_near_step, current_far_step, camera, context
    try:
        if key == ord('+'):
            value = f"Near {current_near_step}"
            current_near_step = (current_near_step % 3) + 1
            set_widget(camera, context, 'manualfocusdrive', value)
        elif key == ord('-'):
            value = f"Far {current_far_step}"
            current_far_step = (current_far_step % 3) + 1
            set_widget(camera, context, 'manualfocusdrive', value)
    except gp.GPhoto2Error:
        pass

# ### MODIFICADO (V10): Adicionado 'f' para flash ###
def print_controls():
    """(Sua função original, modificada)"""
    print("\n--- 2. Iniciando Live View ---")
    print("Controles: [ESPAÇO] captura  [Q|ESC] sair  [F] Levantar Flash")
    print("           [+] foco perto   [-] foco longe")

def capture_image(camera, context):
    """(Sua função original)"""
    print("--- 3. Capturando Foto em Alta Qualidade ---")
    
    try:
        file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
        
        camera_file_get = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
        camera_file_get.save(target_path)
        print("Foto salva em:", target_path)
        
    except gp.GPhoto2Error as ex:
        print("Falha na captura:", ex)
        print("Verifique se a lente está em 'MF' (Foco Manual) se o erro for de foco.")

# -----------------------------------------------------------------
# FUNÇÃO PRINCIPAL (MODIFICADA)
# -----------------------------------------------------------------

def main():
    global camera, context

    # --- AVISO DE ERRO [-53] ---
    print("--- Script de Captura Manual Controlada ---")
    print("\n!! AVISO DE ERRO [-53] 'Could not claim' !!")
    print("Se o script falhar com esse erro, rode este comando no terminal:")
    print("   killall gvfs-gphoto2-volume-monitor")
    print("E então tente rodar o script novamente.\n")
    
    print("AVISO: Para este script funcionar, a câmera DEVE estar:")
    print("  1. Seletor de modo em 'M' (Manual).")
    print("  2. Chave da LENTE em 'MF' (Foco Manual).")
    print("     (Isto é OBRIGATÓRIO para 'focusmode' e os controles [+] e [-] funcionarem.)")
    # --- FIM DO AVISO ---

    if not os.path.exists(CAPTURE_DIR):
        os.makedirs(CAPTURE_DIR)

    context = gp.Context()
    camera = gp.Camera()

    try:
        print("Inicializando câmera...")
        camera.init(context)
        
        travar_configuracoes_manuais()
        ligar_liveview()
        print_controls()

        # Loop principal
        while True:
            try:
                camera_file = camera.capture_preview()
                file_data = camera_file.get_data_and_size()
                image_data = np.frombuffer(file_data, dtype=np.uint8)
                frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    cv2.imshow('Live View', frame)

                key = cv2.waitKey(30) & 0xFF
                if key == 255:
                    continue

                if key == ord('q') or key == 27:
                    print("Saindo...")
                    break

                elif key == ord(' '):
                    desligar_liveview()
                    capture_image(camera, context)
                    ligar_liveview()
                
                elif key == ord('+') or key == ord('-'):
                    desligar_liveview()
                    execute_command(key)
                    ligar_liveview()
                
                # ### NOVO (V10): Adiciona a tecla 'f' para o flash ###
                elif key == ord('f'):
                    print("--- ATIVANDO FLASH POP-UP ---")
                    desligar_liveview()
                    # O valor '1' (inteiro) diz para a câmera levantar o flash
                    set_widget(camera, context, 'popupflash', 1) 
                    ligar_liveview()

            except gp.GPhoto2Error as ex:
                print("Erro no loop do Live View:", ex)
                break

    except gp.GPhoto2Error as ex:
        print("Erro inicializando câmera:", ex)
        sys.exit(1)
        
    finally:
        print("Encerrando...")
        desligar_liveview()
        camera.exit(context)
        cv2.destroyAllWindows()
        
        salvar_log_config()
        
        print("Finalizado.")

if __name__ == "__main__":
    main()

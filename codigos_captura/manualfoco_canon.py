import gphoto2 as gp
import cv2
import os
import numpy as np
from datetime import datetime
import sys
import time

CAPTURE_DIR = 'Fotos-embalagem-28-10-2025/TestesAna'

camera = None
context = None
config = None

current_near_step = 1
current_far_step = 1

def apply_config(camera, context, cfg, retries=3, wait=0.15):
    for _ in range(retries):
        try:
            camera.set_config(cfg, context)
            time.sleep(wait)
            return True
        except gp.GPhoto2Error:
            time.sleep(wait)
    return False

def desligar_liveview():
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
    """Seta um widget pelo nome, retornando True se a leitura posterior confirmar aplicação."""
    for _ in range(retries):
        try:
            cfg = camera.get_config(context)
            w = cfg.get_child_by_name(name)
            if w is None:
                return False
            w.set_value(value)
            if not apply_config(camera, context, cfg):
                time.sleep(wait)
                continue
            # confirmar leitura
            cfg2 = camera.get_config(context)
            w2 = cfg2.get_child_by_name(name)
            if w2 is None:
                return False
            current = w2.get_value()
            if str(value).lower() in str(current).lower() or str(current).lower() in str(value).lower():
                return True
        except gp.GPhoto2Error:
            time.sleep(wait)
    return False

def execute_command(key):
    global current_near_step, current_far_step, camera, context
    try:
        if key == ord('+'):
            value = f"Near {current_near_step}"
            current_near_step = (current_near_step % 3) + 1
            # tenta aplicar manualfocusdrive (refresh interno)
            set_widget(camera, context, 'manualfocusdrive', value)
        elif key == ord('-'):
            value = f"Far {current_far_step}"
            current_far_step = (current_far_step % 3) + 1
            set_widget(camera, context, 'manualfocusdrive', value)
    except gp.GPhoto2Error:
        pass

def print_controls():
    print("Controles: [ESPAÇO] captura  [Q|ESC] sair  [+] foco perto  [-] foco longe")

def capture_image(camera, context):
    # tenta cancelar autofocus e aplicar manualfocusdrive se houver
    set_widget(camera, context, 'cancelautofocus', 1)
    set_widget(camera, context, 'autofocusdrive', 0)
    time.sleep(0.25)

    try:
        file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
        camera_file_get = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
        camera_file_get.save(target_path)
        print("Foto salva em:", target_path)
    except gp.GPhoto2Error as ex:
        print("Falha na captura:", ex)
    finally:
        # tentar reverter cancelautofocus
        try:
            set_widget(camera, context, 'cancelautofocus', 0)
        except Exception:
            pass

def main():
    global camera, context, config

    if not os.path.exists(CAPTURE_DIR):
        os.makedirs(CAPTURE_DIR)

    context = gp.Context()
    camera = gp.Camera()

    try:
        camera.init(context)
        config = camera.get_config(context)
    except gp.GPhoto2Error as ex:
        print("Erro inicializando câmera:", ex)
        sys.exit(1)

    ligar_liveview()
    print_controls()

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
                # captura com tentativa de cancelar AF antes
                capture_image(camera, context)
                ligar_liveview()
            else:
                desligar_liveview()
                execute_command(key)
                ligar_liveview()

        except gp.GPhoto2Error as ex:
            print("Erro no loop do Live View:", ex)
            break

    desligar_liveview()
    camera.exit(context)
    cv2.destroyAllWindows()
    print("Finalizado.")

if __name__ == "__main__":
    main()

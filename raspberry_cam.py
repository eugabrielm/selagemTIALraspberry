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

# Ajustes para Raspberry
CONFIG_RETRIES = 5
CONFIG_WAIT = 0.25
PREVIEW_RETRIES = 5
PREVIEW_WAIT = 0.20
POST_LIVEVIEW_WAIT = 0.40
POST_COMMAND_WAIT = 0.20
POST_CAPTURE_WAIT = 0.50
LOOP_WAIT_MS = 80


def apply_config(camera, context, cfg, retries=CONFIG_RETRIES, wait=CONFIG_WAIT):
    for _ in range(retries):
        try:
            camera.set_config(cfg, context)
            time.sleep(wait)
            return True
        except gp.GPhoto2Error:
            time.sleep(wait)
    return False


def safe_get_config(camera, context, retries=CONFIG_RETRIES, wait=CONFIG_WAIT):
    for _ in range(retries):
        try:
            return camera.get_config(context)
        except gp.GPhoto2Error:
            time.sleep(wait)
    return None


def desligar_liveview():
    global camera, context
    try:
        cfg = safe_get_config(camera, context)
        if cfg is None:
            return False

        vf = cfg.get_child_by_name('viewfinder')
        if vf and vf.get_value() == 1:
            vf.set_value(0)
            ok = apply_config(camera, context, cfg)
            time.sleep(POST_COMMAND_WAIT)
            return ok
    except gp.GPhoto2Error:
        pass
    return False


def ligar_liveview():
    global camera, context
    try:
        cfg = safe_get_config(camera, context)
        if cfg is None:
            return False

        vf = cfg.get_child_by_name('viewfinder')
        if vf and vf.get_value() == 0:
            vf.set_value(1)
            ok = apply_config(camera, context, cfg)
            time.sleep(POST_LIVEVIEW_WAIT)
            return ok
    except gp.GPhoto2Error:
        pass
    return False


def set_widget(camera, context, name, value, retries=CONFIG_RETRIES, wait=CONFIG_WAIT):
    """Seta um widget pelo nome, com mais tolerância a falhas."""
    for _ in range(retries):
        try:
            cfg = safe_get_config(camera, context)
            if cfg is None:
                time.sleep(wait)
                continue

            w = cfg.get_child_by_name(name)
            if w is None:
                return False

            w.set_value(value)

            if not apply_config(camera, context, cfg, retries=1, wait=wait):
                time.sleep(wait)
                continue

            time.sleep(wait)

            cfg2 = safe_get_config(camera, context)
            if cfg2 is None:
                continue

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
            set_widget(camera, context, 'manualfocusdrive', value)
            time.sleep(POST_COMMAND_WAIT)

        elif key == ord('-'):
            value = f"Far {current_far_step}"
            current_far_step = (current_far_step % 3) + 1
            set_widget(camera, context, 'manualfocusdrive', value)
            time.sleep(POST_COMMAND_WAIT)

    except gp.GPhoto2Error:
        pass


def print_controls():
    print("Controles: [ESPAÇO] captura  [Q|ESC] sair  [+] foco perto  [-] foco longe")


def safe_capture_preview():
    """Tenta capturar o preview algumas vezes antes de desistir."""
    for _ in range(PREVIEW_RETRIES):
        try:
            return camera.capture_preview()
        except gp.GPhoto2Error as ex:
            print("Erro no preview:", ex)
            time.sleep(PREVIEW_WAIT)
    return None


def capture_image(camera, context):
    # tenta cancelar autofocus e aplicar manualfocusdrive se houver
    set_widget(camera, context, 'cancelautofocus', 1)
    set_widget(camera, context, 'autofocusdrive', 0)
    time.sleep(0.30)

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
        try:
            set_widget(camera, context, 'cancelautofocus', 0)
        except Exception:
            pass
        time.sleep(POST_CAPTURE_WAIT)


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
    time.sleep(POST_LIVEVIEW_WAIT)

    while True:
        try:
            camera_file = safe_capture_preview()
            if camera_file is None:
                time.sleep(0.10)
                continue

            try:
                file_data = camera_file.get_data_and_size()
                image_data = np.frombuffer(file_data, dtype=np.uint8)
                frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            except Exception as ex:
                print("Erro ao decodificar preview:", ex)
                frame = None

            if frame is not None:
                cv2.imshow('Live View', frame)

            key = cv2.waitKey(LOOP_WAIT_MS) & 0xFF
            if key == 255:
                continue

            if key == ord('q') or key == 27:
                print("Saindo...")
                break

            elif key == ord(' '):
                desligar_liveview()
                time.sleep(0.30)
                capture_image(camera, context)
                ligar_liveview()

            else:
                desligar_liveview()
                time.sleep(0.20)
                execute_command(key)
                ligar_liveview()

        except gp.GPhoto2Error as ex:
            print("Erro no loop do Live View:", ex)
            time.sleep(0.30)
            continue

        except Exception as ex:
            print("Erro inesperado no loop:", ex)
            time.sleep(0.30)
            continue

    desligar_liveview()
    try:
        camera.exit(context)
    except Exception:
        pass

    cv2.destroyAllWindows()
    print("Finalizado.")


if __name__ == "__main__":
    main()

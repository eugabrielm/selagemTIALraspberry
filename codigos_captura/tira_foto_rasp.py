import gphoto2 as gp
import os
import time
from datetime import datetime

CAPTURE_DIR = "Fotos-Teste"

def main():
    if not os.path.exists(CAPTURE_DIR):
        os.makedirs(CAPTURE_DIR)

    context = gp.Context()
    camera = gp.Camera()

    try:
        print("Inicializando câmera...")
        camera.init(context)
        print("Câmera inicializada com sucesso.")

        # pequena pausa para estabilizar
        time.sleep(1)

        print("Capturando foto...")
        file_path = camera.capture(gp.GP_CAPTURE_IMAGE)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = os.path.join(CAPTURE_DIR, f"foto_{timestamp}.jpg")

        print("Baixando foto...")
        camera_file = camera.file_get(
            file_path.folder,
            file_path.name,
            gp.GP_FILE_TYPE_NORMAL
        )

        camera_file.save(target)

        print(f"Foto salva em: {target}")

    except gp.GPhoto2Error as ex:
        print("Erro gphoto2:", ex)

    finally:
        try:
            camera.exit(context)
            print("Conexão encerrada.")
        except:
            pass

if __name__ == "__main__":
    main()

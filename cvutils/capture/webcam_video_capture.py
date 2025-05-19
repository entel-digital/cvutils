import cv2
import time
import logging
import subprocess as sp
import numpy as np
from threading import Thread
from queue import Queue
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class WebcamVideoCapture:
    def __init__(self, src=0, fourcc=None, width=None, height=None, fps=None,
                 transform=None, queue_size=20, name="WebcamVideoCapture", use_hw_accel=False, target_fps=None):
        self.src = src
        self.use_hw_accel = use_hw_accel
        self.target_fps = target_fps
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.transform = transform
        self.name = name
        self.reconnect_attempts = 0
        self.total_reconnect_attempts = 0
        self.last_frame_time = time.time()
        self.logger = logging.getLogger(name)
        self.width = width
        self.height = height
        self.fps = fps
        
        # Popen object para FFmpeg si se usa
        self.ffmpeg_pipe = None
        # Hilo para leer stderr de FFmpeg
        self.ffmpeg_stderr_thread = None

        # Determinar si usar FFmpeg o OpenCV
        if isinstance(src, str) and src.startswith('rtsp://') and use_hw_accel:
            self.use_ffmpeg = True
            self.initialize_ffmpeg_capture()
        else:
            self.use_ffmpeg = False
            self.initialize_capture(fourcc, width, height, fps)

        self.thread = Thread(target=self.update, args=(), name=name)
        self.thread.daemon = True

        self.watchdog_thread = Thread(target=self.watchdog, name=f"{name}_watchdog")
        self.watchdog_thread.daemon = True

    def initialize_ffmpeg_capture(self):
        command = [
            'ffmpeg',
            '-hwaccel', 'cuda',
            '-hwaccel_device', '0',
            '-c:v', 'hevc_cuvid',
            '-rtsp_transport', 'tcp',
            '-i', self.src,
            '-fps_mode', 'passthrough',
            '-fflags', 'nobuffer',
            '-f', 'rawvideo',
            '-pix_fmt', 'nv12'
        ]
        
        if self.width and self.height:
            command.extend(['-s', f'{self.width}x{self.height}'])
            
        command.append('pipe:1')
        
        self.logger.info(f"Iniciando FFmpeg con comando: {' '.join(command)}")
        
        try:
            self.ffmpeg_pipe = sp.Popen(
                command,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                bufsize=10**4
            )
            
            # Hilo para leer el stderr de FFmpeg
            self.ffmpeg_stderr_thread = Thread(
                target=self._capture_ffmpeg_stderr,
                name=f"{self.name}_ffmpeg_stderr",
                daemon=True
            )
            self.ffmpeg_stderr_thread.start()

            # Si no se especificaron dimensiones, intentar obtenerlas del primer frame
            if not self.width or not self.height:
                raw_image = self.ffmpeg_pipe.stdout.read(1920*1080*3)
                if not raw_image:
                    raise RuntimeError("No se pudo leer el primer frame desde FFmpeg (dimensiones desconocidas).")
                frame = np.frombuffer(raw_image, dtype='uint8')
                frame = frame.reshape((1080, 1920, 3))
                self.height, self.width = frame.shape[:2]
                self.queue.put(frame)  # Guardar el primer frame
            
            self.frame_size = int(self.width * self.height * 1.5)
            self.logger.info(f"FFmpeg iniciado correctamente. Dimensiones: {self.width}x{self.height}")
            
        except Exception as e:
            self.logger.error(f"Error iniciando FFmpeg: {str(e)}")
            raise

    def _capture_ffmpeg_stderr(self):
        """Lee continuamente el stderr de FFmpeg y lo registra en el logger."""
        self.logger.debug(f"[{self.name}] Iniciando hilo de lectura de stderr de FFmpeg.")
        while not self.stopped and self.ffmpeg_pipe and self.ffmpeg_pipe.stderr:
            line = self.ffmpeg_pipe.stderr.readline()
            if not line:
                break
            #Ignorar "warning: 'non monotonically increasing dts'"
            if b"non monotonically increasing dts" in line:
                continue
            # Decodificar y loguear
            msg = line.decode('utf-8', errors='replace').strip()
            self.logger.info(f"[FFmpeg {self.name} stderr] {msg}")
        self.logger.debug(f"[{self.name}] Hilo de stderr finalizado.")

    def nv12_to_bgr(self, nv12_frame):
        height = self.height
        width = self.width
        
        # Separar componentes Y y UV
        y = nv12_frame[:height*width].reshape(height, width)
        uv = nv12_frame[height*width:].reshape(height//2, width)
        
        # Convertir a BGR usando cv2
        yuv = np.zeros((height + height//2, width), dtype=np.uint8)
        yuv[:height, :] = y
        yuv[height:, :] = uv
        
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

    def initialize_capture(self, fourcc, width, height, fps):
        self.logger.debug(f"[{self.name}] Inicializando captura con OpenCV VideoCapture.")
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG if self.use_hw_accel else cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {self.src}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)

        if self.use_hw_accel:
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            self.cap.set(cv2.CAP_PROP_HW_DEVICE, 0)

        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def start(self):
        self.logger.debug(f"[{self.name}] Iniciando hilo principal de captura (update).")
        self.thread.start()
        # Activar el watchdog
        self.watchdog_thread.start()
        return self

    def update(self):
        consecutive_successful_reads = 0
        last_buffer_cleanup = time.time()
        
        while not self.stopped:
            try:
                current_time = time.time()
                
                # Limpieza periódica del buffer
                if current_time - last_buffer_cleanup > 5:  # Cada 5 segundos
                    while self.queue.qsize() > self.queue.maxsize * 0.5:  # Mantener solo 50% del buffer
                        self.logger.debug(f"[{self.name}] Limpieza de buffer, descartando frame.")
                        self.queue.get()
                    last_buffer_cleanup = current_time

                if self.use_ffmpeg:
                    if not self.ffmpeg_pipe:
                        self.logger.debug(f"[{self.name}] ffmpeg_pipe is None, intentando reconectar.")
                        self.reconnect()
                        continue
                                
                    raw_image = self.ffmpeg_pipe.stdout.read(self.frame_size)
                    if not raw_image:
                        self.logger.warning(f"No data received from FFmpeg for {self.name}")
                        self.reconnect()
                        continue
                            
                    frame = np.frombuffer(raw_image, dtype='uint8')
                    # Convertir NV12 a BGR
                    frame = self.nv12_to_bgr(frame)
                            
                else:
                    if not self.cap.isOpened():
                        self.logger.warning(f"[{self.name}] cap no está abierto, intentando reconectar.")
                        self.reconnect()
                        continue

                    ret, frame = self.cap.read()
                    if not ret:
                        self.logger.warning(f"Error reading frame from {self.name}")
                        self.reconnect()
                        continue

                self.last_frame_time = time.time()

                if self.transform:
                    frame = self.transform(frame)

                # Control de buffer
                while self.queue.qsize() > self.queue.maxsize * 0.75:
                    self.logger.debug(f"[{self.name}] Buffer near maxsize, descartando un frame de la cola.")
                    self.queue.get()
                self.queue.put(frame)

                consecutive_successful_reads += 1
                if consecutive_successful_reads >= 100:
                    self.logger.debug(f"[{self.name}] Reset de reconnect_attempts tras 100 lecturas exitosas.")
                    self.reconnect_attempts = 0
                    consecutive_successful_reads = 0

            except Exception as e:
                self.logger.error(f"Error in update loop ({self.name}): {str(e)}")
                self.reconnect()
                time.sleep(1)

    def reconnect(self):
        self.total_reconnect_attempts += 1
        wait_time = min(30, 2 ** (self.total_reconnect_attempts % 10))
        self.logger.info(f"Attempting to reconnect {self.name}... (Total attempts: {self.total_reconnect_attempts}, waiting {wait_time}s)")
        
        if self.use_ffmpeg:
            if self.ffmpeg_pipe:
                self.logger.debug(f"[{self.name}] Terminando ffmpeg_pipe antes de reintentar.")
                self.ffmpeg_pipe.terminate()
                self.ffmpeg_pipe = None
            time.sleep(wait_time)
            try:
                self.initialize_ffmpeg_capture()
            except Exception as e:
                self.logger.warning(f"FFmpeg reconnection failed for {self.name}: {str(e)}")
        else:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            time.sleep(wait_time)
            try:
                self.initialize_capture(None, None, None, None)
            except Exception as e:
                self.logger.warning(f"OpenCV reconnection failed for {self.name}: {str(e)}")

    def watchdog(self):
        """Si pasan 30s sin frames, intentamos reconectar."""
        self.logger.debug(f"[{self.name}] Watchdog iniciado.")
        while not self.stopped:
            if time.time() - self.last_frame_time > 30:
                self.logger.warning(f"Watchdog detected no frames for {self.name}, restarting capture")
                self.reconnect()
            time.sleep(5)
        self.logger.debug(f"[{self.name}] Watchdog finalizado (stop).")

    def read(self, timeout=None):
        """Lee un frame de la cola con timeout (si se especifica)."""
        try:
            if timeout is not None:
                frame = self.queue.get(timeout=timeout) 
            else:
                frame = self.queue.get()

            return frame
        except queue.Empty:
            self.logger.warning(f"No frame received in {timeout} seconds.")
            return None

    def running(self):
        return self.more() or not self.stopped

    def more(self):
        tries = 0
        while self.queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.queue.qsize() > 0

    def stop(self):
        self.logger.info(f"[{self.name}] Stopping capture.")
        self.stopped = True
        self.thread.join()
        self.watchdog_thread.join()
        
        if self.use_ffmpeg and self.ffmpeg_pipe:
            self.logger.debug(f"[{self.name}] Terminando ffmpeg_pipe en stop().")
            self.ffmpeg_pipe.terminate()
            self.ffmpeg_pipe = None
        
        elif not self.use_ffmpeg and hasattr(self, 'cap'):
            self.cap.release()
            self.cap = None
        
        self.logger.debug(f"[{self.name}] Capture stopped.")

    def get_health(self):
        return {
            "is_running": not self.stopped,
            "is_capture_open": (self.ffmpeg_pipe is not None) if self.use_ffmpeg else (
                self.cap.isOpened() if self.cap else False
            ),
            "last_frame_time": self.last_frame_time,
            "reconnect_attempts": self.reconnect_attempts,
            "queue_size": self.queue.qsize(),
            "capture_method": "FFmpeg+CUDA" if self.use_ffmpeg else "OpenCV"
        }
    
    def get(self, propId):
        if self.use_ffmpeg:
            if propId == cv2.CAP_PROP_FPS:
                return self.fps if self.fps else 30  # valor por defecto
            elif propId == cv2.CAP_PROP_FRAME_WIDTH:
                return self.width
            elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
                return self.height
            return None
        else:
            if hasattr(self, 'cap') and self.cap:
                return self.cap.get(propId)
            return None
    
    def set(self, propId, value):
        if self.use_ffmpeg:
            return False
        if hasattr(self, 'cap') and self.cap:
            return self.cap.set(propId, value)
        return False
    
    def is_alive(self):
        return not self.stopped and (self.ffmpeg_pipe is not None if self.use_ffmpeg else (
            self.cap.isOpened() if self.cap else False
        ))

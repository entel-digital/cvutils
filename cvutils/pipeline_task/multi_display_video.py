import cv2
from cvutils.pipeline_task.pipeline_task import PipelineTask
import math

class MultiDisplayVideo(PipelineTask):
    """Pipeline task to display multiple video streams in a single window."""

    def __init__(self, src, window_name="Multi-Stream Display", window_size=(1920, 1080), border_color=(72, 80, 96), border_thickness=2):
        self.src = src
        self.window_name = window_name
        self.window_width, self.window_height = window_size
        self.border_color = border_color
        self.border_thickness = border_thickness
        #cv2.startWindowThread()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        super().__init__()

    def map(self, data_list):
        num_streams = len(data_list)
        grid_cols = math.ceil(math.sqrt(num_streams))
        grid_rows = math.ceil(num_streams / grid_cols)
        
        # Ajustar correctamente filas y columnas si hay menos streams que celdas
        if grid_cols * (grid_rows - 1) >= num_streams:
            grid_rows -= 1

        # Tamaño del frame original
        frame_height, frame_width = data_list[0][self.src].shape[:2]
        
        # Tamaño de cada frame redimensionado dentro de la ventana
        resized_frame_width = self.window_width // grid_cols
        resized_frame_height = self.window_height // grid_rows

        # Crea un frame en negro con el tamaño deseado
        display_frame = cv2.resize(cv2.hconcat([data_list[0][self.src] for _ in range(grid_cols)]), 
                                   (self.window_width, self.window_height))
        display_frame = cv2.resize(cv2.vconcat([display_frame for _ in range(grid_rows)]), 
                                   (self.window_width, self.window_height))

        display_frame[:] = (0, 0, 0)  # Asigna un color negro de fondo

        for i, data in enumerate(data_list):
            row = i // grid_cols
            col = i % grid_cols
            x_offset = resized_frame_width * col
            y_offset = resized_frame_height * row

            resized_frame = cv2.resize(data[self.src], (resized_frame_width, resized_frame_height))
            display_frame[y_offset:y_offset + resized_frame.shape[0], x_offset:x_offset + resized_frame.shape[1]] = resized_frame

            # Dibujar el contorno alrededor de cada flujo de video
            cv2.rectangle(display_frame, (x_offset, y_offset), 
                        (x_offset + resized_frame.shape[1], y_offset + resized_frame.shape[0]), 
                        self.border_color, self.border_thickness)

        cv2.imshow(self.window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q") or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE) < 0:
            raise StopIteration

        return data_list

    def close(self):
        cv2.destroyWindow(self.window_name)
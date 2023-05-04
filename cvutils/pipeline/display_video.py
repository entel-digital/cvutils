import cv2

from cvutils.pipeline import Pipeline


class DisplayVideo(Pipeline):
    """Pipeline task to display images as a video."""

    def __init__(self, src, window_name=None, org=None, full_screen=False):
        self.src = src
        self.window_name = window_name if window_name else src

        cv2.startWindowThread()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        if org:
            # Set the window position
            x, y = org
            cv2.moveWindow(self.window_name, x, y)
        if full_screen:
            cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        super().__init__()

    def map(self, data):
        image = data[self.src]

        cv2.imshow(self.window_name, image)

        #print('mapping display')
        # Exit?
        key = cv2.waitKey(1) & 0xFF
        # Esc key pressed or window closed?
        if key == 27 or key == ord("q"): # or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            raise StopIteration
        #print(cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE))
        return data

    def close(self):
        cv2.destroyWindow(self.window_name)

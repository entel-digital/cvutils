from cvutils.pipeline import Pipeline
#from mediapipe.solutions import pose
#from .libs.pose_tracker import PoseTracker
import mediapipe as mp
import cv2


class AnnotateVideo(Pipeline):
    def __init__(self, dst, annotate_pose=False, annotate_fps=False, annotate_aruco=False):
        self.dst = dst
        self.annotate_pose = annotate_pose
        self.annotate_fps = annotate_fps
        self.annotate_aruco = annotate_aruco

        super().__init__()
        # self.metadata = MetadataCatalog.get(self.metadata_name)
        # self.instance_mode = instance_mode
        # self.frame_num = frame_num
        # self.predictions = predictions
        # self.pose_flows = pose_flows

        # self.cpu_device = torch.device("cpu")
        #self.video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

    def map(self, data):
        dst_image = data["image"].copy()
        data[self.dst] = dst_image

        # if self.frame_num:
        #     self.annotate_frame_num(data)
        # if self.predictions:
        #     self.annotate_predictions(data)
        # if self.pose_flows:
        #     self.annotate_pose_flows(data)
        if self.annotate_aruco:
            self.aruco_annotator(data)
        if self.annotate_pose:
            self.pose_annotator(data)
        if self.annotate_fps:
            self.fps_annotator(data)

        #self.mode_title_annotator(data)
        #self.mode_counters_annotator(data)

        # if 'debug_line' in data['mode'].keys():
        #     self.debug_line_annotator(data)

        return data

    def fps_annotator(self, data):
        dst_image = data[self.dst]
        if 'fps' in data:
            frame_rate = data['fps']
            pos = (int(dst_image.shape[1] - 250), 50)
            cv2.putText(dst_image, f"{frame_rate:.2f} fps", pos,
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1.4,
                    color=(255, 255, 255))

        return data

    def pose_annotator(self, data):
        dst_image = data[self.dst]
        mp.solutions.drawing_utils.draw_landmarks(
                dst_image,
                data['results_mp_pose'].pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    def aruco_annotator(self, data):
        dst_image = data[self.dst]
        if 'aruco' in data:
            for key, corners in data['aruco'].items():
                topLeft, topRight, bottomRight, bottomLeft = corners
                # draw the bounding box of the ArUCo detection
                cv2.line(dst_image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(dst_image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(dst_image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(dst_image, bottomLeft, topLeft, (0, 255, 0), 2)

                # draw the ArUco marker ID on the image
                cv2.putText(dst_image, str(key),
                            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

    def mode_title_annotator(self, data):
        dst_image = data[self.dst]
        pos = (int((dst_image.shape[1] / 2) - 30), 50)
        cv2.putText(dst_image, data['mode']['display_name'], pos,
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=2,
                    color=(170, 0, 255))

    def mode_counters_annotator(self, data):
        dst_image = data[self.dst]
        # TODO: refactor to make it multy counter compatible (when refactored to pygame)
        counter, value = next(iter(data['mode']['counters'].items()))
        text = f"{counter}: {value}"
        pos = (10, 50)
        cv2.putText(dst_image, text, pos,
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=2,
                    color=(85, 255, 51))

    def debug_line_annotator(self, data):
        dst_image = data[self.dst]
        pos = (10, dst_image.shape[0] - 50)
        cv2.putText(dst_image, data['mode']['debug_line'], pos,
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=2,
                    color=(0, 170, 255))

# cvutils/pipeline_task/aruco_finder.py
from cvutils.pipeline_task.pipeline_task import PipelineTask
import cv2

class ArucoFinder(PipelineTask):
    def __init__(self, aruco_map: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters() 
        self.aruco_map = aruco_map if aruco_map else {i: f'aruco {i}' for i in range(50)}

    def map(self, data_list):
        for data in data_list:
            self.find_aruco(data)

        return data_list

    def find_aruco(self, data):
        image = data["image"]
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, self.arucoDict, parameters=self.arucoParams)

        aruco_found = {}
        if len(corners) > 0:
            ids = ids.flatten()
            for (markerCorner, markerID) in zip(corners, ids):
                if markerID in self.aruco_map.keys():
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    aruco_found[self.aruco_map[markerID]] = (topLeft, topRight, bottomRight, bottomLeft)

            data['aruco'] = aruco_found
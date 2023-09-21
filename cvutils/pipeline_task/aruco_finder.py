from cvutils.pipeline_task.pipeline_task import PipelineTask

import cv2

class ArucoFinder(PipelineTask):
    def __init__(self, aruco_map: dict = None, **kwargs):
        super().__init__(**kwargs)
        # API changed for 4.7.x
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        #self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        #self.arucoParams = cv2.aruco.DetectorParameters_create()
        # / API changed for 4.7.x
        if aruco_map is not None:
            self.aruco_map = aruco_map
        else:
            self.aruco_map = {}
            for item in range(50):
                self.aruco_map[item] = 'aruco ' + str(item)

    def map(self, data):
        data = self.find_aruco(data)

        return data

    #TODO: exponer ejes (x,y,z)
    def find_aruco(self, data):
        image = data["image"]
        # API changed for 4.7.x
        #(corners, ids, rejected) = cv2.aruco.detectMarkers(image, self.arucoDict,
        #                                                   parameters=self.arucoParams)
        (corners, ids, rejected) = self.detector.detectMarkers(image)
        # / API changed for 4.7.x

        aruco_found = {}
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()

            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                if markerID in self.aruco_map.keys():
                    # extract the marker corners (which are always returned in
                    # top-left, top-right, bottom-right, and bottom-left order)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    aruco_found[self.aruco_map[markerID]] = (topLeft, topRight, bottomRight, bottomLeft)

            data['aruco'] = aruco_found

        return data

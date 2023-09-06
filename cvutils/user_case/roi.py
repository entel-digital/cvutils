from shapely import Polygon



class Roi:
    def __init__(self, points):
        self.polygon = Polygon(points)

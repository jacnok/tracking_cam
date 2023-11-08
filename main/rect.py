class Rect:
    """ A class to manage the rectangle attributes of a bounding box. """
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.ex = None  # End x (x + width)
        self.ey = None  # End y (y + height)
        self.cx = None  # Center x
        self.cy = None  # Center y

    def overlap(self, box: 'Rect') -> tuple[bool, float]:
        """ Calculate if there's an overlap with another box based on distance. """
        deltax = self.cx - box.cx
        deltay = self.cy - box.cy
        dist = (deltax ** 2) + (deltay ** 2)
        return (dist < (self.w * box.w)), dist

    def set(self, bbox_tuple: tuple[int, int, int, int]):
        """ Update rectangle coordinates and derived metrics. """
        (self.x, self.y, self.w, self.h) = bbox_tuple
        self.ex = self.x + self.w
        self.ey = self.y + self.h
        self.cx = self.x + (self.w/2)
        self.cy = self.y + (self.h/2)
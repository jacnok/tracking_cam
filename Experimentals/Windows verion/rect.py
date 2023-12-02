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

    def setC(self, cx,cy):
        """ Update rectangle coordinates and derived metrics. """
        self.cx=cx
        self.cy=cy
        if(self.w is not None):
            self.ex = self.cx + (self.w/2)
            self.ey = self.cy + (self.h/2)
            self.x = self.cx - (self.w/2)
            self.y = self.cy - (self.h/2)
        else:
            print("error in rect class, creating boxes from center")
            self.w=100
            self.h=100
            self.ex = self.cx + (self.w/2)
            self.ey = self.cy + (self.h/2)
            self.x = self.cx - (self.w/2)
            self.y = self.cy - (self.h/2)
from PyQt5.QtGui import QPainter
from PyQt5.QtOpenGL import QGLWidget

class GLImageWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None


    def setImage(self, image):
        self.image = image
        self.update()

    def paintGL(self):
        if self.image:
            painter = QPainter(self)
            painter.drawImage(self.rect(), self.image, self.image.rect())
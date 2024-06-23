import numpy as np

from manim import *
from manim import config

class CoordinateSystemScene(MovingCameraScene):
    def construct(self):
        rotate_center = ORIGIN
        theta_tracker = ValueTracker(0)
        theta_tex = Tex(
            r"$\theta = $",
            str(int(theta_tracker.get_value())),
            r"$^\circ$",
            font_size=50,
            color=RED,
        ).shift(UP * 2)
        vector1 = Vector([2, 3], color=BLUE)
        vector2 = Vector([2, 1], color=RED)
        # 创建一个坐标系
        number_plane = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "stroke_opacity": 0.6,
            }
        )
        multiply_res = Text(vector1 @ vector2)
        self.play(Create(number_plane))
        self.play(Create(vector1), Create(vector2))
        # self.play(Rotate(vector1, PI, about_point=number_plane.c2p(0, 0)),
        #           run_time=2)

# 运行代码
if __name__ == "__main__":
    scene = CoordinateSystemScene()
    scene.render()

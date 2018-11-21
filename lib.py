import numpy as np
from PIL import Image, ImageDraw


class Point:
    def __init__(self, *args):
        if len(args) == 1 and len(args[0]) == 3:
            self.coors = np.array(args[0])
        elif len(args) == 3:
            self.coors = np.array(args)

    def x(self):
        return self.coors[0]

    def y(self):
        return self.coors[1]

    def z(self):
        return self.coors[2]

    def __getitem__(self, key):
        return self.coors[key]

    def __iter__(self):
        return self.coors.__iter__()

    def __len__(self):
        return 3

    def apply_transform(self, transform):
        pt = np.ones((4, 1))
        pt[:3, :] = self.coors.reshape((3, 1))
        m = transform.matrix.dot(pt).reshape((4))
        m /= m[3]
        return Point(m[:3])

    def __str__(self):
        return self.coors.__str__()

    def __repr__(self):
        return self.__str__()


class Line:
    def __init__(self, *args):
        if len(args) == 2:
            self.data = np.hstack((
                np.array(args[0]).reshape((3, 1)),
                np.array(args[1]).reshape((3, 1))
            ))

    def __getitem__(self, key):
        if key == 0:
            return Point(self.data[:, 0])
        elif key == 1:
            return Point(self.data[:, 1])


class Polygon:
    def __init__(self, *args):
        if len(args) == 1:
            points = args[0]
        else:
            points = args
        self.data = np.hstack(
            map(lambda x: np.array(x).reshape((3, 1)), points)
        )

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, key):
        return self.data[:, key]


class Polyhedron:
    def __init__(self, points, sides):  # side is an iterable of indexes of points

        self.points = np.hstack(
            map(lambda x: np.array(x).reshape((3, 1)), points)
        ).astype(np.float64)
        self.sides = sides
        self.center = self.find_center()
        self.normals = self.find_normals()

    def find_center(self):
        s = self.points.sum(axis=1)
        s /= self.points.shape[1]
        return Point(s.reshape(3))

    def find_normals(self):
        normals = np.ndarray((len(self.sides), 3))
        for i in range(len(self.sides)):
            s = self.sides[i]
            p = self.points[:, s[0]]
            p_next = self.points[:, s[1]]
            p_prev = self.points[:, s[-1]]

            v_out = p_next-p
            v_in = p-p_prev

            n = np.cross(v_in, v_out)
            normals[i, :] = n / (n**2).sum()**.5
        return normals

    def assign_vertex_normals(self):
        ns = np.zeros((self.points.shape[1], 3))
        cs = np.zeros(self.points.shape[1])
        for n_side in range(len(self.sides)):
            for v in self.sides[n_side]:
                cs[v] += 1
                ns[v] += self.normals[n_side]
        ns /= cs[...,np.newaxis].repeat(3, axis=1)
        ns /= ((ns**2).sum(axis=1)**.5)[..., np.newaxis].repeat(3, axis=1)
        self.vertex_normals = ns

    def apply_relative_transform(self, transform):
        tr = Transform.translate(
            self.center[0],
            self.center[1],
            self.center[2]
        ).compose(transform).compose(Transform.translate(
            -self.center[0],
            -self.center[1],
            -self.center[2]
        ))
        return self.apply_transform(tr)

    def apply_transform(self, transform):
        l = self.points.shape[1]
        p = np.ones((4, l))
        p[:3, :] = self.points
        r = transform.matrix.dot(p)
        r /= r[3, :]
        return Polyhedron(r[:3, :].T, self.sides)

    def save_obj(self, filename):
        with open(filename, 'wt') as f:
            for p in self.points.T:
                f.write("v {} {} {}\n".format(*p))
            for s in self.sides:
                f.write("f {} {} {}\n".format(s[0] + 1, s[1] + 1, s[2] + 1))

    @staticmethod
    def load_obj(filename):
        points = []
        sides = []
        with open(filename, 'rt') as f:
            for line in f.readlines():
                ls = line.split()
                if len(ls) != 4:
                    continue
                t = ls[0]
                if t == 'f':
                    d = [int(x.split('/')[0]) - 1 for x in ls[1:]]
                    sides.append(d)
                elif t == 'v':
                    d = [float(x.split('/')[0]) for x in ls[1:]]
                    points.append(Point(*d))
        return Polyhedron(points, sides)

    @staticmethod
    def Tetrahedron(center, radius):
        tetrahedral_angle = np.arccos(-1 / 3)
        tr_rot = Transform.rotate('z', 120 / 180 * np.pi)
        p1 = Point(0, 0, radius)
        p2 = p1.apply_transform(Transform.rotate('x', tetrahedral_angle))
        p3 = p2.apply_transform(tr_rot)
        p4 = p3.apply_transform(tr_rot)
        p = Polyhedron([p1, p2, p3, p4], [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ])
        p = p.apply_transform(Transform.translate(center.x(), center.y(), center.z()))
        return p

    @staticmethod
    def Cube(center, side):
        points = [
            Point(0, 0, 0),
            Point(0, side, 0),
            Point(side, side, 0),
            Point(side, 0, 0),
            Point(0, 0, side),
            Point(0, side, side),
            Point(side, side, side),
            Point(side, 0, side)
        ]
        sides = [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [3, 4, 0], [3, 7, 4],
            [1, 6, 2], [1, 5, 6],
            [2, 7, 3], [2, 6, 7]
        ]
        p = Polyhedron(points, sides)
        p = p.apply_transform(Transform.translate(
            center.x() - side / 2,
            center.y() - side / 2,
            center.z() - side / 2
        ))
        return p

    @staticmethod
    def Octahedron(center, radius):
        points = [
            Point(0, 0, radius),
            Point(radius, 0, 0),
            Point(0, radius, 0),
            Point(-radius, 0, 0),
            Point(0, -radius, 0),
            Point(0, 0, -radius)
        ]
        sides = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 5, 2],
            [2, 5, 3],
            [3, 5, 4],
            [4, 5, 1]
        ]
        p = Polyhedron(points, sides)
        p = p.apply_transform(Transform.translate(
            center.x(),
            center.y(),
            center.z()
        ))
        return p


class Transform:
    def __init__(self, matrix):
        self.matrix = np.ndarray((4, 4))
        self.matrix[...] = matrix

    def compose(self, transform):
        return Transform(self.matrix.dot(transform.matrix))

    @staticmethod
    def identity():
        return Transform(np.identity(4))

    @staticmethod
    def translate(dx, dy, dz):
        return Transform([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def scale(sx, sy, sz):
        return Transform([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def rotate(axis, angle):
        sin = np.sin(angle)
        cos = np.cos(angle)
        if axis == 'x':
            return Transform([
                [1, 0, 0, 0],
                [0, cos, -sin, 0],
                [0, sin, cos, 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'y':
            return Transform([
                [cos, 0, sin, 0],
                [0, 1, 0, 0],
                [-sin, 0, cos, 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'z':
            return Transform([
                [cos, -sin, 0, 0],
                [sin, cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

    @staticmethod
    def rotate_around_line(line, angle):
        dx = line[1].x() - line[0].x()
        dy = line[1].y() - line[0].y()
        dz = line[1].z() - line[0].z()
        angle_to_yz = np.arctan2(dx, dz)
        angle_to_xz = np.arctan2(dy, dz)
        tr = Transform.rotate('y', angle_to_xz).compose(
            Transform.rotate('x', angle_to_yz)
        )
        untr = Transform.rotate('x', -angle_to_yz).compose(
            Transform.rotate('y', -angle_to_xz)
        )
        return untr.compose(
            Transform.rotate('z', angle)
        ).compose(tr)

    @staticmethod
    def reflect(plane):
        if plane == 'xy':
            return Transform([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
        elif plane == 'xz':
            return Transform([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif plane == 'yz':
            return Transform([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])


class Camera:
    def __init__(self, matrix, vec_view, pre_transform=Transform.identity(), pos=[0]*3, angles=[0]*3):
        self.matrix = np.ndarray((3, 4))
        self.matrix[...] = matrix
        self.vec_view = vec_view
        self.pre_transform = pre_transform.compose(
            Transform.rotate('x', -angles[0]).compose(
                Transform.rotate('y', -angles[1]).compose(
                    Transform.rotate('z', -angles[2]).compose(
                        Transform.translate(-pos[0], -pos[1], -pos[2])
                    )
                )
            )
        )

    def draw(self, size, polyhedron):
        transformed = polyhedron.apply_transform(self.pre_transform)
        points = transformed.points
        lines  = transformed.sides

        image = Image.new('RGB', size)
        size = np.array(size)
        l = points.shape[1]
        p = np.ones((4, l))
        p[:3, :] = points
        r = self.matrix.dot(p)
        pts = (r / r[2, :])[:2, :]
        draw = ImageDraw.Draw(image)
        red = np.array((255, 0, 0))
        blue = np.array((0, 0, 255))
        for line in lines:
            for i in range(len(line)):
                b = pts[:, line[i]].T + size / 2
                e = pts[:, line[(i + 1) % len(line)]].T + size / 2
                # draw.line((b[0], size[1]-b[1], e[0], size[1]-e[1]), fill=(255,0,0))
                for j in range(10):
                    pb = b + (e - b) / 10 * j
                    pe = b + (e - b) / 10 * (j + 1)
                    pz = points[:, line[i]][1] + (
                                points[:, line[(i + 1) % len(line)]][1] - points[:, line[i]][1]) * j / 10
                    k = pz / 50
                    k = 1 if k > 1 else (0 if k < 0 else k)
                    col = (red + k * (blue - red)).astype('int')
                    draw.line((pb[0], size[1] - pb[1], pe[0], size[1] - pe[1]), fill=tuple(col))

        return image

    def draw_with_culling(self, size, polyhedron):
        transformed = polyhedron.apply_transform(self.pre_transform)
        points = transformed.points
        lines  = transformed.sides
        normals = transformed.normals

        image = Image.new('RGB', size)
        size = np.array(size)
        l = points.shape[1]
        p = np.ones((4, l))
        p[:3, :] = points
        r = self.matrix.dot(p)
        pts = (r / r[2, :])[:2, :]
        draw = ImageDraw.Draw(image)
        red = np.array((255, 0, 0))
        blue = np.array((0, 0, 255))
        for k in range(len(lines)):
            line = lines[k]
            n = normals[k]
            vec = self.vec_view(points[:, line[0]].reshape(3))

            if n.dot(vec) >= 0:
                continue

            for i in range(len(line)):
                b = pts[:, line[i]].T + size / 2
                e = pts[:, line[(i + 1) % len(line)]].T + size / 2
                for j in range(10):
                    pb = b + (e - b) / 10 * j
                    pe = b + (e - b) / 10 * (j + 1)
                    pz = points[:, line[i]][1] + (
                                points[:, line[(i + 1) % len(line)]][1] - points[:, line[i]][1]) * j / 10
                    k = pz / 50
                    k = 1 if k > 1 else (0 if k < 0 else k)
                    col = (red + k * (blue - red)).astype('int')
                    draw.line((pb[0], size[1] - pb[1], pe[0], size[1] - pe[1]), fill=tuple(col))

        return image

    def draw_with_both_culling_and_zbuf(self, size, polyhedron):
        transformed = polyhedron.apply_transform(self.pre_transform)
        points = transformed.points
        lines = transformed.sides
        normals = transformed.normals

        image = np.zeros(size + (3,))
        zbuf = np.full(size, np.finfo('d').max)
        size = np.array(size)
        l = points.shape[1]
        p = np.ones((4, l))
        p[:3, :] = points
        r = self.matrix.dot(p)
        pts = (r / r[2, :])[:2, :]

        # front_col = np.array((255, 0, 127))
        # back_col = np.array((0, 127, 255))
        front_col = np.array((255, 0, 0))
        back_col = np.array((255, 255, 255))

        for j in range(len(lines)):
            line = lines[j]
            n = normals[j]
            vec = self.vec_view(points[:, line[0]].reshape(3))

            if n.dot(vec) >= 0:
                continue

            sorted_p = sorted(line, key=lambda x: pts[1, x])
            p_up = pts[:, sorted_p[2]].reshape(2).astype('int')
            r_up = np.sum(self.vec_view(points[:, sorted_p[2]].reshape(3))**2)
            p_mid = pts[:, sorted_p[1]].reshape(2).astype('int')
            r_mid = np.sum(self.vec_view(points[:, sorted_p[1]].reshape(3))**2)
            p_down = pts[:, sorted_p[0]].reshape(2).astype('int')
            r_down = np.sum(self.vec_view(points[:, sorted_p[0]].reshape(3))**2)

            for y in range((p_down[1]), (p_up[1] + 1)):
                if size[1] // 2 - y < 0:
                    break
                if size[1] // 2 - y >= size[1]:
                    continue
                if y >= int(p_mid[1]):
                    if (p_mid[1]) == (p_up[1]):
                        continue
                    k = (y - p_mid[1]) / (p_up[1] - p_mid[1])
                    x_mid = (p_up[0] - p_mid[0]) * k + p_mid[0]
                    r_mid2 = (r_up - r_mid) * k + r_mid
                else:
                    if (p_down[1]) == (p_mid[1]):
                        continue
                    k = (y - p_down[1]) / (p_mid[1] - p_down[1])
                    x_mid = (p_mid[0] - p_down[0]) * k + p_down[0]
                    r_mid2 = (r_mid - r_down) * k + r_down
                k2 = (y - p_down[1]) / (p_up[1] - p_down[1])
                x_up = (p_up[0] - p_down[0]) * k2 + p_down[0]
                r_up2 = (r_up - r_down) * k2 + r_down
                if x_mid < x_up:
                    x_left, x_right = int(x_mid), int(x_up)
                    r_left, r_right = (r_mid2), (r_up2)
                else:
                    x_left, x_right = int(x_up), int(x_mid)
                    r_left, r_right = (r_up2), (r_mid2)
                x_left = min(max(-size[0] // 2, x_left), size[0] // 2)
                x_right = max(min(size[0] // 2, x_right), -size[0] // 2)
                interp_k = np.linspace(0, 1, x_right - x_left)

                interp = interp_k * (r_right - r_left) + r_left

                # color = np.clip(interp / 2500, 0, 1) * (back_col - front_col).reshape(3, 1) + front_col.reshape(3, 1)
                color = np.ones_like(interp) * (j / len(lines)) * (back_col - front_col).reshape(3, 1) + front_col.reshape(3, 1)

                x_left += size[0] // 2
                x_right += size[0] // 2
                y = size[1] // 2 - y
                if y >= size[0] or y <= 0:
                    continue
                ln = np.where((zbuf[y, x_left:x_right] > interp)[:, np.newaxis].repeat(3, axis=1), color.T,
                              image[y, x_left:x_right])
                image[y, x_left:x_right] = ln
                zbuf[y, x_left:x_right] = np.where(zbuf[y, x_left:x_right] > interp, interp,
                                                       zbuf[y, x_left:x_right])

        return Image.fromarray(image.astype("uint8"))

    def draw_with_culling_and_zbuf_and_shading(self, size, polyhedron, light_pos):
        transformed = polyhedron.apply_transform(self.pre_transform)
        points = transformed.points
        lines = transformed.sides
        normals = transformed.normals

        transformed.assign_vertex_normals()
        v_normals = transformed.vertex_normals

        image = np.zeros(size + (3,))
        zbuf = np.full(size, np.finfo('d').max)
        size = np.array(size)
        l = points.shape[1]
        p = np.ones((4, l))
        p[:3, :] = points
        r = self.matrix.dot(p)
        pts = (r / r[2, :])[:2, :]

        # front_col = np.array((255, 0, 127))
        # back_col = np.array((0, 127, 255))
        front_col = np.array((255, 0, 0))
        back_col = np.array((255, 255, 255))

        for j in range(len(lines)):
            line = lines[j]
            n = normals[j]
            vec = self.vec_view(points[:, line[0]].reshape(3))

            if n.dot(vec) >= 0:
                continue

            sorted_p = sorted(line, key=lambda x: pts[1, x])
            p_up = pts[:, sorted_p[2]].reshape(2).astype('int')
            r_up = np.sum(self.vec_view(points[:, sorted_p[2]].reshape(3))**2)
            v_up = v_normals[sorted_p[2]]
            p3_up = points[:, sorted_p[2]]
            p_mid = pts[:, sorted_p[1]].reshape(2).astype('int')
            r_mid = np.sum(self.vec_view(points[:, sorted_p[1]].reshape(3))**2)
            v_mid = v_normals[sorted_p[1]]
            p3_mid = points[:, sorted_p[1]]
            p_down = pts[:, sorted_p[0]].reshape(2).astype('int')
            r_down = np.sum(self.vec_view(points[:, sorted_p[0]].reshape(3))**2)
            v_down = v_normals[sorted_p[0]]
            p3_down = points[:, sorted_p[0]]

            for y in range((p_down[1]), (p_up[1] + 1)):
                if size[1] // 2 - y < 0:
                    break
                if size[1] // 2 - y >= size[1]:
                    continue
                if y >= int(p_mid[1]):
                    if (p_mid[1]) == (p_up[1]):
                        continue
                    k = (y - p_mid[1]) / (p_up[1] - p_mid[1])
                    x_mid = (p_up[0] - p_mid[0]) * k + p_mid[0]
                    r_mid2 = (r_up - r_mid) * k + r_mid
                    v_mid2 = (v_up - v_mid) * k + v_mid
                    p3_mid2 = (p3_up - p3_mid) * k + p3_mid
                else:
                    if (p_down[1]) == (p_mid[1]):
                        continue
                    k = (y - p_down[1]) / (p_mid[1] - p_down[1])
                    x_mid = (p_mid[0] - p_down[0]) * k + p_down[0]
                    r_mid2 = (r_mid - r_down) * k + r_down
                    v_mid2 = (v_mid - v_down) * k + v_down
                    p3_mid2 = (p3_mid - p3_down) * k + p3_down
                k2 = (y - p_down[1]) / (p_up[1] - p_down[1])
                x_up = (p_up[0] - p_down[0]) * k2 + p_down[0]
                r_up2 = (r_up - r_down) * k2 + r_down
                v_up2 = (v_up - v_down) * k2 + v_down
                p3_up2 = (p3_up - p3_down) * k2 + p3_down
                if x_mid < x_up:
                    x_left, x_right = int(x_mid), int(x_up)
                    r_left, r_right = r_mid2, r_up2
                    v_left, v_right = v_mid2, v_up2
                    p3_left, p3_right = p3_mid2, p3_up2
                else:
                    x_left, x_right = int(x_up), int(x_mid)
                    r_left, r_right = r_up2, r_mid2
                    v_left, v_right = v_up2, v_mid2
                    p3_left, p3_right = p3_up2, p3_mid2
                x_left = min(max(-size[0] // 2, x_left), size[0] // 2)
                x_right = max(min(size[0] // 2, x_right), -size[0] // 2)

                if x_right - x_left <= 0:
                    continue

                # print('x_right - x_left:', x_right - x_left)

                interp_k = np.linspace(0, 1, x_right - x_left)

                # print('interp_k:', interp_k)

                interp = interp_k * (r_right - r_left) + r_left
                # print('interp:', interp)

                v_interp = interp_k * (v_right.reshape(3, 1) - v_left.reshape(3, 1)) + v_left.reshape(3, 1)
                # print('v_interp:', v_interp)
                p3_interp = interp_k * (p3_left.reshape(3, 1) - p3_right.reshape(3, 1)) + p3_right.reshape(3, 1)
                # print('p3_interp:', p3_interp)
                light_norm_interp = np.array(light_pos).reshape(3, 1) - p3_interp
                light_norm_interp /= (light_norm_interp**2).sum(axis=0)**.5
                # print('light_norm_interp:', light_norm_interp)
                light_interp = (v_interp * light_norm_interp).sum(axis=0)
                # print('light_interp:', light_interp)

                # color = np.clip(interp / 2500, 0, 1) * (back_col - front_col).reshape(3, 1) + front_col.reshape(3, 1)
                # color = np.ones_like(interp) * (j / len(lines)) * (back_col - front_col).reshape(3, 1) + front_col.reshape(3, 1)
                light_interp = light_interp[np.newaxis,...].repeat(3, axis=0)

                color = light_interp * 127 + 127
                # print('color:', color)

                x_left += size[0] // 2
                x_right += size[0] // 2
                y = size[1] // 2 - y
                if y >= size[0] or y <= 0:
                    continue
                ln = np.where((zbuf[y, x_left:x_right] > interp)[:, np.newaxis].repeat(3, axis=1), color.T,
                              image[y, x_left:x_right])
                image[y, x_left:x_right] = ln
                zbuf[y, x_left:x_right] = np.where(zbuf[y, x_left:x_right] > interp, interp,
                                                       zbuf[y, x_left:x_right])

        return Image.fromarray(image.astype("uint8"))

    @staticmethod
    def ortho(pos=[0]*3, angles=[0]*3):
        return Camera([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], lambda p: np.array([
            0, p[1], 0
        ]), pos=pos, angles=angles)

    @staticmethod
    def persp(k, pos=[0]*3, angles=[0]*3):
        return Camera([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, k, 0, 1]
        ], lambda p: np.array([p[0], p[1]+1/k, p[2]]),
            pos=pos, angles=angles)

    @staticmethod
    def iso(a=np.arcsin(np.tan(30 / 180 * np.pi)), b=np.pi / 4, pos=[0,0,0], angles=[0,0,0]):
        tr = Transform.rotate('x', a).compose(
            Transform.rotate('z', b)
        )
        # vec = tr.matrix.dot(
        #     np.array([
        #         0, 1, 0, 1
        #     ]).reshape((4,1))
        # )
        # vec = (vec / vec[3,:])[:3, :]
        m = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return Camera(m, #lambda p: vec)
            lambda p: np.array([
                0, p[1], 0
            ]),
            tr,
            pos=pos, angles=angles
        )

if __name__ == "__main__":
    import imageio as iio
    t = Polyhedron.Octahedron(Point(0,70,0),75)
    # print(t.points)
    c = Camera.persp(0.012, pos=[0,-20,0], angles=[0,0,0])
    with iio.get_writer("test.gif", fps=30) as w:
        for i in range(200):
            tr = Transform.scale(1,1,1).compose(
                Transform.rotate('x', 0.2).compose(
                    Transform.rotate('z', 2*np.pi*i/200)
                    # Transform.identity()
                    # Transform.rotate_around_line(
                    #     Line(Point(0,0,0), Point(1-i/100,0,i/100)),
                    #     2*np.pi*i/10
                    # )
                )
            )
            # tr = Transform.reflect('yz').compose(tr)
            p = t.apply_relative_transform(tr)
            light_pos = [100*np.cos(2*np.pi*i/100), 0, 100*np.cos(2*np.pi*i/100)]
            im=np.array(c.draw_with_culling_and_zbuf_and_shading((224, 224), p, light_pos=light_pos))
            w.append_data(im)
    # s = Polyhedron.Cube(Point(0, 200, 0), 100)
    # s.save_obj('tmp.obj')
    # t = Polyhedron.load_obj('tmp.obj').apply_relative_transform(Transform.rotate('x', 1))
    # t = Polyhedron.load_obj('deer.obj') \
    #     .apply_relative_transform(
    #     Transform.rotate('x', np.pi / 2).compose(
    #         Transform.scale(*[0.1] * 3)
    #     )
    # )
    # t = t.apply_transform(
    #     Transform.translate(-150, -850, 0)
    # )
    # c = Camera.persp(0.01)
    # # c = Camera.ortho()
    # c.draw((200, 200), t.points, t.sides).show()

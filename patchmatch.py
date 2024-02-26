import numpy as np


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class PatchMatch(object):
    """
    quickly searching the corresponded patches between two images
    """

    def __init__(self, image_A, image_B, patch_size=5, search_radius=6, iters=2):
        """
        image_A: source image numpy.ndarray H*W*C
        image_B: target image numpy.ndarray H*W*C
        search_radius: if search_radius >= 1, do random search within search_radius to avoid local minimum
        """
        self.A = image_A
        self.B = image_B
        self.A_norm = normalization(self.A)
        self.B_norm = normalization(self.B)
        self.patch_size = patch_size
        self.search_radius = search_radius
        self.iters = iters

    def nnf_Init(self):
        """
        Before searching, the nearest neighbour field needs random initialization.
        A[i][j] is corresponding to B[nnf[i, j, 0]][nnf[i, j, 1]]
        """
        A_H = self.A.shape[0]
        A_W = self.A.shape[1]
        nnf = np.zeros([A_H, A_W, 2], dtype=np.int32)
        nnf[:, :, 0] = np.random.randint(0, self.B.shape[0], size=(A_H, A_W))
        nnf[:, :, 1] = np.random.randint(0, self.B.shape[1], size=(A_H, A_W))
        return nnf

    def cal_Distance(self, a_x, a_y, b_x, b_y):
        """
        calculate the distance between two patches
        """
        A_H = self.A.shape[0]
        A_W = self.A.shape[1]
        B_H = self.B.shape[0]
        B_W = self.B.shape[1]
        dx0 = dy0 = self.patch_size // 2
        dx1 = dy1 = self.patch_size // 2 + 1
        dx0 = min(a_x, b_x, dx0)
        dx1 = min(A_H - a_x, B_H - b_x, dx1)
        dy0 = min(a_y, b_y, dy0)
        dy1 = min(A_W - a_y, B_W - b_y, dy1)
        patch_A = self.A_norm[a_x - dx0: a_x + dx1, a_y - dy0: a_y + dy1]
        patch_B = self.B_norm[b_x - dx0: b_x + dx1, b_y - dy0: b_y - dy1]
        dist = np.sum((patch_A - patch_B) ** 2) / (dx0 + dx1) * (dy0 + dy1)
        return dist

    def nnd_Init(self, nnf):
        """
        nnd records the offset between each corresponding patch.
        """
        A_H = self.A.shape[0]
        A_W = self.A.shape[1]
        nnd = np.zeros([A_H, A_W])
        for i in range(A_H):
            for j in range(A_W):
                nnd[i, j] = self.cal_Distance(i, j, nnf[i, j, 0], nnf[i, j, 1])
        return nnd

    def nnf_Search(self):
        self.nnf_Init()
        A_H = self.A.shape[0]
        A_W = self.A.shape[1]
        nnf = self.nnf_Init()
        nnd = self.nnd_Init(nnf)
        for iter in range(1, self.iters + 1):
            if iter % 2 == 0:  # even iterations: use a reverse scan
                for i in range(A_H - 1, -1, -1):
                    for j in range(A_W, -1, -1):
                        nnf, nnd = self.propagation(i, j, nnf, nnd, False)
                        nnf, nnd = self.random_search(i, j, nnf, nnd, self.search_radius)

            else:  # odd iterations: use a forward scan
                for i in range(A_H):
                    for j in range(A_W):
                        nnf, nnd = self.propagation(i, j, nnf, nnd, True)
                        nnf, nnd = self.random_search(i, j, nnf, nnd, self.search_radius)

    def propagation(self, a_x, a_y, nnf, nnd, is_odd):
        """
        nearest neighbour field (NNF) propagation
        """
        A_H = self.A.shape[0]
        A_W = self.A.shape[1]
        B_H = self.B.shape[0]
        B_W = self.B.shape[1]
        dist_best = nnd[a_x, a_y]
        if is_odd:  # odd iterations
            if a_y - 1 >= 0:
                b_x, b_y = nnf[a_x, a_y - 1, 0], nnf[a_x, a_y - 1, 1] + 1
                if b_y < B_W:
                    dist = self.cal_Distance(a_x, a_y, b_x, b_y)
                    if dist < dist_best:
                        nnf[a_x, a_y] = [b_x, b_y]
                        dist_best = dist
                        nnd[a_x, a_y] = dist_best

            if a_x - 1 >= 0:
                b_x, b_y = nnf[a_x - 1, a_y, 0] + 1, nnf[a_x - 1, a_y, 1]
                if b_x < B_H:
                    dist = self.cal_Distance(a_x, a_y, b_x, b_y)
                    if dist < dist_best:
                        nnf[a_x, a_y] = [b_x, b_y]
                        dist_best = dist
                        nnd[a_x, a_y] = dist_best
        else:  # even iterations
            if a_y + 1 < A_W:
                b_x, b_y = nnf[a_x, a_y + 1, 0], nnf[a_x, a_y + 1, 1] - 1
                if b_y >= 0:
                    dist = self.cal_Distance(a_x, a_y, b_x, b_y)
                    if dist < dist_best:
                        nnf[a_x, a_y] = [b_x, b_y]
                        dist_best = dist
                        nnd[a_x, a_y] = dist_best
            if a_x + 1 < A_H:
                b_x, b_y = nnf[a_x + 1, a_y + 1, 0] - 1, nnf[a_x + 1, a_y + 1, 1]
                if b_x >= 0:
                    dist = self.cal_Distance(a_x, a_y, b_x, b_y)
                    if dist < dist_best:
                        nnf[a_x, a_y] = [b_x, b_y]
                        dist_best = dist
                        nnd[a_x, a_y] = dist_best
        return nnf, nnd

    def random_search(self, a_x, a_y, nnf, nnd, radius):
        """
        to avoid local minimum
        """
        B_H = self.B.shape[0]
        B_W = self.B.shape[1]
        b_x_best, b_y_best = nnf[a_x, a_y, 0], nnf[a_x, a_y, 1]
        dist_best = nnd[a_x, a_y]
        while radius >= 1:
            start_x = max(b_x_best - self.search_radius, 0)
            end_x = min(b_x_best + self.search_radius + 1, B_H)
            start_y = max(b_y_best - self.search_radius, 0)
            end_y = min(b_y_best + self.search_radius + 1, B_W)
            b_x = np.random.randint(start_x, end_x)
            b_y = np.random.randint(start_y, end_y)
            dist = self.cal_Distance(a_x, a_y, b_x, b_y)
            if dist < dist_best:
                b_x_best = b_x
                b_y_best = b_y
                dist_best = dist
            radius //= 2
        nnf[a_x, a_y, 0], nnf[a_x, a_y, 1] = b_x_best, b_y_best
        nnd[a_x, a_y] = dist_best
        return nnf, nnd

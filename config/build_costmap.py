import imageio
from PIL import Image
import numpy as np

im = imageio.imread('301_1f.png')
w,h = im.shape
print(im.shape)

clusters = []
visited = [[False for i in range(h)] for j in range(w)]

for i in range(w):
    for j in range(h):
        if im[i][j] < 210:
            visited[i][j] = True

A = np.zeros((w,h), dtype=int)

for i in range(w):
    for j in range(h):
        if visited[i][j]:
            A[i][j] = 255

im = Image.fromarray(np.uint8(A))
im.save("your.png")

dx=[1,-1,0,0]
dy=[0,0,1,-1]

for i in range(w):
    for j in range(h):
        if visited[i][j]:
            continue
        cluster = []
        q = []
        q.append([i,j])
        while len(q) > 0:
            cur = q[0]
            q.pop(0)
            if visited[cur[0]][cur[1]]:
                continue
            visited[cur[0]][cur[1]] = True
            cluster.append(cur)
            for k in range(4):
                nx = cur[0] + dx[k]
                ny = cur[1] + dy[k]
                if visited[nx][ny]:
                    continue
                q.append([nx, ny])
        clusters.append(cluster)

idx = 0
n = len(clusters[0])
for i in range(len(clusters)):
    if n < len(clusters[i]):
        n = len(clusters[i])
        idx = i

free_space = clusters[idx]

A = np.zeros((w,h), dtype=int)
B = np.full((w,h), 255, dtype=int)

for p in clusters[idx]:
    A[p[0]][p[1]] = 255
    B[p[0]][p[1]] = 0

min_d = 7
step_size = 0.04
half_kernel = 33
kernel_size = 2*half_kernel + 1
kernel = np.zeros((kernel_size, kernel_size))
for i in range(kernel_size):
    for j in range(kernel_size):
        d = abs(i-half_kernel) + abs(j-half_kernel)
        if d <= min_d:
            kernel[i][j] = 255
        else:
            value = 1 - (d - min_d) * step_size
            if value < 0:
                continue
            kernel[i][j] = int(255 * value)
print(kernel)

# B = np.full((w,h), 255, dtype=int)
for i in range(w):
    for j in  range(h):
        if A[i][j] == 0:
            continue
        for k in range(4):
            nx = i + dx[k]
            ny = j + dy[k]
            if A[nx][ny] == 255:
                continue
            for ki in range(-half_kernel, half_kernel):
                for kj in range(-half_kernel, half_kernel):
                    ni = nx + ki
                    nj = ny + kj
                    if ni < 0 or ni >= w or nj < 0 or nj >= h:
                        continue
                    B[ni][nj] = max(B[ni][nj], kernel[ki + half_kernel][kj + half_kernel])

# f = open("collision_301_1f.txt", 'w')

# space = []
# for i in range(w):
#     for j in range(h):
#         if B[i][j] == 255:
#             space.append([i,j])
# f.write("%d %d %d\n" %(w, h, len(space)))
# for p in space:
#     f.write("%d %d\n" %(p[0], p[1]))
# f.close()
# space = np.array(space)
# print(np.min(space,axis=0))
# print(np.max(space,axis=0))

im = Image.fromarray(np.uint8(A))
im_collision = Image.fromarray(np.uint8(B))
# im.save("free_space_301_1f.png")
im_collision.save("costmap_301_1f.png")
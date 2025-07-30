import matplotlib.pyplot as plt
from collections import deque
import math

# 地图参数
ROWS = 7
COLS = 9
CELL_SIZE = 0.5
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 生成格子编号
cols_label = [f"A{i}" for i in range(1, COLS+1)]
rows_label = [f"B{i}" for i in range(1, ROWS+1)]

def cell_to_coord(cell):
    if cell.startswith("A") and "B" in cell:
        a = int(cell[1:cell.index("B")])
        b = int(cell[cell.index("B")+1:])
        # 地图左下角是 A1B1 → (B1 是第1行)，行数向上增长（index0 是 B1）
        return b - 1, a - 1  # row, col
    raise ValueError("非法格子编号")

def coord_to_cell(r, c):
    return f"A{c+1}B{r+1}"

# 生成地图
def init_grid():
    return [[1]*COLS for _ in range(ROWS)]

def is_valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def get_neighbors(r, c, grid):
    neighbors = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if is_valid(nr, nc) and grid[nr][nc] == 1:
            neighbors.append((nr, nc))
    return neighbors

def bfs(start, end, grid):
    visited = [[False]*COLS for _ in range(ROWS)]
    parent = [[None]*COLS for _ in range(ROWS)]
    q = deque([start])
    visited[start[0]][start[1]] = True
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break
        for nr, nc in get_neighbors(r, c, grid):
            if not visited[nr][nc]:
                visited[nr][nc] = True
                parent[nr][nc] = (r, c)
                q.append((nr, nc))
    path = []
    curr = end
    while curr and curr != start:
        path.append(curr)
        curr = parent[curr[0]][curr[1]]
    if curr == start:
        path.append(start)
        path.reverse()
        return path
    return []

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def nearest_neighbor_path(start, points, grid):
    path = [start]
    visited = set()
    visited.add(start)
    current = start
    while len(visited) < len(points):
        unvisited = [p for p in points if p not in visited]
        if not unvisited:
            break
        next_pt = min(unvisited, key=lambda p: manhattan(current, p))
        subpath = bfs(current, next_pt, grid)
        if not subpath:
            break
        for pt in subpath[1:]:
            path.append(pt)
        visited.add(next_pt)
        current = next_pt
    return_path = bfs(current, start, grid)
    if return_path:
        path.extend(return_path[1:])
    return path

def draw(grid, path, obstacles):
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c * CELL_SIZE, (ROWS - 1 - r) * CELL_SIZE
            color = "white"
            if (r, c) in obstacles:
                color = "gray"
            rect = plt.Rectangle((x, y), CELL_SIZE, CELL_SIZE,
                                 edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(x + 0.25, y + 0.25, coord_to_cell(r, c),
                    ha='center', va='center', fontsize=6)

    if path:
        xs = [c * CELL_SIZE + 0.25 for _, c in path]
        ys = [(ROWS - 1 - r) * CELL_SIZE + 0.25 for r, _ in path]
        ax.plot(xs, ys, 'r-', linewidth=2)
        ax.plot(xs, ys, 'bo', markersize=4)
        sr, sc = path[0]
        ax.plot(sc * CELL_SIZE + 0.25, (ROWS - 1 - sr) * CELL_SIZE + 0.25,
                'go', markersize=10, label='Start A9B1')
    ax.set_xlim(0, COLS * CELL_SIZE)
    ax.set_ylim(0, ROWS * CELL_SIZE)
    ax.set_aspect('equal')
    ax.set_title("Path Planning")
    ax.legend()
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    print("请输入三个连续禁飞格子（如：A3B4 A4B4 A5B4）：")
    obstacle_cells = input("输入禁飞格子，用空格隔开: ").split()
    grid = init_grid()
    obstacles = []

    try:
        for oc in obstacle_cells:
            r, c = cell_to_coord(oc)
            if is_valid(r, c):
                grid[r][c] = 0
                obstacles.append((r, c))
    except Exception as e:
        print("输入错误:", e)
        exit(1)

    start = cell_to_coord("A9B1")  # 固定起点：右下角 A9B1
    points = [(r, c) for r in range(ROWS) for c in range(COLS) if grid[r][c] == 1]

    path = nearest_neighbor_path(start, points, grid)
    if len(path) < len(points):
        print("⚠️ 有部分点无法到达（孤岛存在）")
    print(f"路径共 {len(path)} 步，遍历率：{100*len(set(path))/len(points):.1f}%")

    # 输出路径到文件
    path_labels = [coord_to_cell(r, c) for r, c in path]
    with open("path_plan.txt", "w") as f:
        f.write(" -> ".join(path_labels))
    print("路径已保存至 path_plan.txt")

    draw(grid, path, obstacles)

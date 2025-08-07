#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import matplotlib.pyplot as plt
from collections import deque
import math
import os
import struct
import time
from PIL import Image

# 地图参数
ROWS = 7
COLS = 9
CELL_SIZE = 0.5
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

cols_label = [f"A{i}" for i in range(1, COLS+1)]
rows_label = [f"B{i}" for i in range(1, ROWS+1)]

def cell_to_coord(cell):
    if cell.startswith("A") and "B" in cell:
        a = int(cell[1:cell.index("B")])
        b = int(cell[cell.index("B")+1:])
        return b - 1, a - 1
    raise ValueError("非法格子编号")

def coord_to_cell(r, c):
    return f"A{c+1}B{r+1}"

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

def draw(grid, path, obstacles, filename_jpg):
    temp_png = "/tmp/temp_path.png"
    fig, ax = plt.subplots(figsize=(5, 3))  # 保持原始画布尺寸
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
        # 不再显示每个航点的蓝色点
        # ax.plot(xs, ys, 'bo', markersize=4)
        # 不再显示起飞点的框（绿色圆点仍保留）
        sr, sc = path[0]
        ax.plot(sc * CELL_SIZE + 0.25, (ROWS - 1 - sr) * CELL_SIZE + 0.25,
                'go', markersize=10, label='Start A9B1')
        # 添加方向箭头，两两航点之间
        for i in range(len(path)-1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            x1, y1 = c1 * CELL_SIZE + 0.25, (ROWS - 1 - r1) * CELL_SIZE + 0.25
            x2, y2 = c2 * CELL_SIZE + 0.25, (ROWS - 1 - r2) * CELL_SIZE + 0.25
            dx, dy = x2 - x1, y2 - y1
            norm = math.hypot(dx, dy)
            if norm > 0.01:
                xm = x1 + dx * 0.4
                ym = y1 + dy * 0.4
                ax.arrow(xm, ym, dx * 0.2, dy * 0.2, width=0.03, head_width=0.12, head_length=0.12,
                         length_includes_head=True, fc='orange', ec='orange', zorder=10)
    ax.set_xlim(0, COLS * CELL_SIZE)
    ax.set_ylim(0, ROWS * CELL_SIZE)
    ax.set_aspect('equal')
    ax.set_title("Path Planning")
    ax.legend()
    plt.gca().invert_yaxis()
    plt.savefig(temp_png, dpi=120)  # 提高dpi以提升清晰度
    plt.close()

    # 用 Pillow 再次压缩成 jpg
    img = Image.open(temp_png)
    img = img.convert("RGB")
    img.save(filename_jpg, "JPEG", quality=50, optimize=True)

def send_image_to_screen(publisher, filepath, savepath=r"ram/a.jpg"):
    filesize = os.path.getsize(filepath)
    cmd = f'twfile "{savepath}",{filesize}\r\n'
    publisher.publish(String(data=cmd))
    time.sleep(0.3)

    with open(filepath, 'rb') as f:
        data = f.read()

    chunk_size = 256  # 调小以提高可靠性
    packet_id = 0
    offset = 0

    while offset < len(data):
        chunk = data[offset:offset + chunk_size]
        header = b"\x3a\xa1\xbb\x44\x7f\xff\xfe"
        header += b"\x00"  # 无校验
        header += struct.pack('<H', packet_id)
        header += struct.pack('<H', len(chunk))
        packet = header + chunk
        hex_str = packet.hex()
        publisher.publish(String(data=hex_str))
        offset += len(chunk)
        packet_id += 1
        time.sleep(0.02)

    # 结束透传（强制结束包）
    publisher.publish(String(data="3aa1bb447ffffe00ffff0000"))

class GroundStationNode(Node):
    def __init__(self):
        super().__init__('ground_station_node')
        self.subscription = self.create_subscription(
            String,
            '/read_screen_command',
            self.serial_callback,
            10)
        self.publisher = self.create_publisher(String, '/grid_waypoint', 10)
        self.serial_cmd_publisher = self.create_publisher(String, '/serial_screen_command', 10)
        self.command_buffer = []
        self.last_command = None

    def serial_callback(self, msg):
        command = msg.data.strip()
        if command != self.last_command:
            self.command_buffer.append(command)
            self.last_command = command
            self.get_logger().info(f"收到禁飞格: {command}")
        if len(self.command_buffer) == 3:
            self.get_logger().info(f"已接收3个禁飞格: {self.command_buffer}")
            self.process_commands()
            self.command_buffer = []
            self.last_command = None

    def process_commands(self):
        grid = init_grid()
        obstacles = []
        try:
            for oc in self.command_buffer:
                r, c = cell_to_coord(oc)
                if is_valid(r, c):
                    grid[r][c] = 0
                    obstacles.append((r, c))
        except Exception as e:
            self.get_logger().error(f"解析禁飞格出错: {e}")
            return

        start = cell_to_coord("A9B1")
        points = [(r, c) for r in range(ROWS) for c in range(COLS) if grid[r][c] == 1]

        path = nearest_neighbor_path(start, points, grid)
        path_labels = [coord_to_cell(r, c) for r, c in path]

        with open("path_plan.txt", "w") as f:
            f.write(" -> ".join(path_labels))

        self.publisher.publish(String(data=",".join(path_labels)))
        self.get_logger().info(f"路径已发布，共 {len(path_labels)} 点")

        image_path = r"/home/delicers/Desktop/way_point_map.jpg"
        draw(grid, path, obstacles, image_path)
        self.get_logger().info(f"路径图已保存至 {image_path}")
        send_image_to_screen(self.serial_cmd_publisher, image_path)

def main(args=None):
    rclpy.init(args=args)
    node = GroundStationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

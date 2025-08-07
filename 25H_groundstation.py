#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from collections import deque
import math
import time

# 地图参数 (9列 x 7行)
COLS = 9  # A1-A9 (横向)
ROWS = 7  # B1-B7 (纵向)
CELL_SIZE = 70
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def cell_to_coord(cell):
    """将单元格标识(A1B1)转换为行列坐标"""
    if cell.startswith("A") and "B" in cell:
        a_part = cell[1:cell.index("B")]  # A后的数字部分
        b_part = cell[cell.index("B")+1:]  # B后的数字部分
        col = int(a_part) - 1  # A1对应列0，A9对应列8
        row = int(b_part) - 1  # B1对应行0，B7对应行6
        return (row, col)  # (行, 列)
    raise ValueError("非法格子编号")

def coord_to_cell(row, col):
    """将行列坐标转换为单元格标识(A1B1)"""
    a_label = f"A{col + 1}"  # 列0对应A1，列8对应A9
    b_label = f"B{row + 1}"  # 行0对应B1，行6对应B7
    return f"{a_label}{b_label}"

def init_grid():
    """初始化网格地图，1表示可通行，0表示障碍物"""
    return [[1 for _ in range(COLS)] for _ in range(ROWS)]

def is_valid(row, col):
    """检查坐标是否在网格范围内"""
    return 0 <= row < ROWS and 0 <= col < COLS

def get_neighbors(row, col, grid):
    """获取指定单元格的有效邻居"""
    neighbors = []
    for dr, dc in DIRECTIONS:
        nr, nc = row + dr, col + dc
        if is_valid(nr, nc) and grid[nr][nc] == 1:
            neighbors.append((nr, nc))
    return neighbors

def bfs(start, end, grid):
    """广度优先搜索最短路径"""
    visited = [[False for _ in range(COLS)] for _ in range(ROWS)]
    parent = [[None for _ in range(COLS)] for _ in range(ROWS)]
    q = deque([start])
    visited[start[0]][start[1]] = True
    
    while q:
        row, col = q.popleft()
        if (row, col) == end:
            break
        for nr, nc in get_neighbors(row, col, grid):
            if not visited[nr][nc]:
                visited[nr][nc] = True
                parent[nr][nc] = (row, col)
                q.append((nr, nc))
    
    # 回溯路径
    path = []
    curr = end
    while curr and curr != start:
        path.append(curr)
        curr = parent[curr[0]][curr[1]]
    
    if curr == start:
        path.append(start)
        path.reverse()
        return path
    return []  # 无路径

def manhattan(a, b):
    """曼哈顿距离计算"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def nearest_neighbor_path(start, points, grid):
    """最近邻路径规划算法"""
    path = [start]
    visited = set([start])
    current = start
    
    while len(visited) < len(points):
        unvisited = [p for p in points if p not in visited]
        if not unvisited:
            break
        # 选择最近的未访问点
        next_point = min(unvisited, key=lambda p: manhattan(current, p))
        # 计算当前点到下一点的路径
        subpath = bfs(current, next_point, grid)
        if not subpath:
            break
        # 添加子路径（排除起点，避免重复）
        path.extend(subpath[1:])
        visited.add(next_point)
        current = next_point
    
    # 返回起点
    return_path = bfs(current, start, grid)
    if return_path:
        path.extend(return_path[1:])
    
    return path

def draw_arrow(commands, x, y, dx, dy, color):
    """
    在(x, y)为箭头中心点，dx,dy为方向向量（会归一化），长度固定，绘制V型箭头
    用两条line指令
    """
    arrow_length = 16  # 箭头长度
    arrow_angle = math.radians(25)  # 箭头两侧夹角(度数转弧度)
    main_length = arrow_length / 2  # 箭头一半长度（从中点到顶端）
    
    # 归一化方向
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    
    # 箭头顶点（指向方向）
    tip_x = x + ux * main_length
    tip_y = y + uy * main_length

    # 反方向
    bx = -ux
    by = -uy

    # 左右两侧方向
    sin_a = math.sin(arrow_angle)
    cos_a = math.cos(arrow_angle)
    # 左侧
    left_dx = bx * cos_a - by * sin_a
    left_dy = bx * sin_a + by * cos_a
    # 右侧
    right_dx = bx * cos_a + by * sin_a
    right_dy = -bx * sin_a + by * cos_a

    side_length = arrow_length * 0.7  # 两侧线段长度（比主线略短）

    left_x = x + left_dx * side_length
    left_y = y + left_dy * side_length
    right_x = x + right_dx * side_length
    right_y = y + right_dy * side_length

    # 两条V型线
    commands.append(
        f"line {tip_x:.1f},{tip_y:.1f},{left_x:.1f},{left_y:.1f},{color}\r\n"
    )
    commands.append(
        f"line {tip_x:.1f},{tip_y:.1f},{right_x:.1f},{right_y:.1f},{color}\r\n"
    )

def generate_draw_commands(grid, path, obstacles):
    """生成绘图指令，并在路径每段中点画方向箭头"""
    commands = []
    # 颜色定义
    WHITE = "WHITE"
    BLACK = "BLACK"
    RED = "RED"
    GREEN = "GREEN"
    GRAY = "GRAY"
    ARROW = "BLACK"

    # 清屏
    commands.append("cls\r\n")

    # 绘图参数设置
    cell_pixels = 50  # 每个格子的像素大小
    map_width = COLS * cell_pixels
    map_height = ROWS * cell_pixels

    # 将地图定位到右下角，A9B1在(700,400)
    start_x = 700 - map_width  # X坐标 = 目标X - 地图宽度
    start_y = 400 - map_height # Y坐标 = 目标Y - 地图高度

    # 绘制格子内容（障碍物和可通行区域）
    for row in range(ROWS):
        for col in range(COLS):
            # 计算格子的左上角坐标（Y轴反转）
            x1 = start_x + col * cell_pixels
            y1 = start_y + (ROWS - 1 - row) * cell_pixels
            x2 = x1 + cell_pixels
            y2 = y1 + cell_pixels
            
            # 关键修改：为禁飞区格子添加灰色填充
            if (row, col) in obstacles:
                # 先画灰色背景
                commands.append(f"cirs {x1+25},{y1+25},25,{GRAY}\r\n")
            else:
                # 可通行格子
                commands.append(f"draw {x1},{y1},{x2},{y2},{WHITE}\r\n")
            
            # 绘制格子标签
            label = coord_to_cell(row, col)
            text_x = x1 + cell_pixels // 2
            text_y = y1 + cell_pixels // 2
            commands.append(f"text \"{label}\",{text_x},{text_y},{BLACK},1\r\n")

    # 绘制网格线
    # 横线
    for i in range(ROWS + 1):
        y = start_y + i * cell_pixels
        commands.append(f"line {start_x},{y},{start_x + map_width},{y},{BLACK}\r\n")
    # 竖线
    for j in range(COLS + 1):
        x = start_x + j * cell_pixels
        commands.append(f"line {x},{start_y},{x},{start_y + map_height},{BLACK}\r\n")

    # 绘制路径线（红色）并在每段中点画箭头
    if path and len(path) >= 2:
        for i in range(len(path) - 1):
            row1, col1 = path[i]
            row2, col2 = path[i + 1]
            x1 = start_x + col1 * cell_pixels + cell_pixels // 2
            y1 = start_y + (ROWS - 1 - row1) * cell_pixels + cell_pixels // 2
            x2 = start_x + col2 * cell_pixels + cell_pixels // 2
            y2 = start_y + (ROWS - 1 - row2) * cell_pixels + cell_pixels // 2
            commands.append(f"line {x1},{y1},{x2},{y2},{RED}\r\n")

            # 画方向箭头：取中点，方向为(x2-x1, y2-y1)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            dx = x2 - x1
            dy = y2 - y1
            draw_arrow(commands, mid_x, mid_y, dx, dy, ARROW)

    # 绘制起点标记（绿色实心圆）
    if path:
        start_row, start_col = path[0]
        start_x_pixel = start_x + start_col * cell_pixels + cell_pixels // 2
        start_y_pixel = start_y + (ROWS - 1 - start_row) * cell_pixels + cell_pixels // 2
        commands.append(f"cirs {start_x_pixel},{start_y_pixel},5,{GREEN}\r\n")

    return commands

class GroundStationNode(Node):
    """地面站节点"""
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
        """处理接收到的禁飞格命令"""
        command = msg.data.strip()
        # 避免重复接收相同命令
        if command != self.last_command:
            self.command_buffer.append(command)
            self.last_command = command
            self.get_logger().info(f"收到禁飞格: {command}")
        
        # 收集到3个禁飞格后处理
        if len(self.command_buffer) == 3:
            self.get_logger().info(f"已接收3个禁飞格: {self.command_buffer}")
            self.process_commands()
            self.command_buffer = []
            self.last_command = None

    def process_commands(self):
        """处理禁飞格命令并生成路径"""
        grid = init_grid()
        obstacles = []
        
        try:
            # 解析禁飞格
            for oc in self.command_buffer:
                row, col = cell_to_coord(oc)
                if is_valid(row, col):
                    grid[row][col] = 0  # 标记为障碍物
                    obstacles.append((row, col))
                    self.get_logger().info(f"设置禁飞格: {oc} ({row},{col})")
        except Exception as e:
            self.get_logger().error(f"解析禁飞格出错: {e}")
            return

        # 设置起点为A9B1（右下角）
        start = cell_to_coord("A9B1")  # 对应(row=0, col=8)
        
        # 获取所有可通行点
        points = [(row, col) for row in range(ROWS) for col in range(COLS) if grid[row][col] == 1]
        
        # 如果起点不可通行，直接返回
        if start not in points:
            self.get_logger().error("起点A9B1被标记为禁飞区，请重新设置禁飞格")
            return

        # 生成路径
        path = nearest_neighbor_path(start, points, grid)
        if not path:
            self.get_logger().error("无法生成路径")
            return

        # 转换路径为格子标签
        path_labels = [coord_to_cell(row, col) for row, col in path]
        
        # 保存路径到文件
        with open("path_plan.txt", "w") as f:
            f.write(" -> ".join(path_labels))
        
        # 发布路径
        self.publisher.publish(String(data=",".join(path_labels)))
        self.get_logger().info(f"路径已发布，共 {len(path_labels)} 点")

        # 生成并发送绘图指令
        commands = generate_draw_commands(grid, path, obstacles)
        for cmd in commands:
            msg = String()
            msg.data = cmd
            self.serial_cmd_publisher.publish(msg)
            time.sleep(0.02)  # 适当延迟，确保指令正确接收

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = GroundStationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

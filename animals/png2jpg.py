import os
from PIL import Image

def convert_png_to_jpg(input_dir, output_dir=None, quality=95, delete_png=False):
    """
    批量将PNG图片转换为JPG格式
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径，如果为None则在原目录生成
        quality (int): JPG质量，范围1-100
        delete_png (bool): 转换成功后是否删除原PNG文件
    """
    # 如果没有指定输出目录，使用输入目录
    if output_dir is None:
        output_dir = input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计转换数量
    converted_count = 0
    deleted_count = 0
    failed_files = []
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            try:
                # 构建完整的文件路径
                input_path = os.path.join(input_dir, filename)
                output_filename = os.path.splitext(filename)[0] + '.jpg'
                output_path = os.path.join(output_dir, output_filename)
                
                # 打开PNG图片
                with Image.open(input_path) as img:
                    # 如果图片有透明通道，转换为RGB模式
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # 创建白色背景
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 保存为JPG格式
                    img.save(output_path, 'JPEG', quality=quality, optimize=True)
                    print(f"✓ 转换成功: {filename} -> {output_filename}")
                    converted_count += 1
                    
                    # 如果设置了删除PNG文件，则删除原文件
                    if delete_png:
                        try:
                            os.remove(input_path)
                            print(f"  └─ 已删除原文件: {filename}")
                            deleted_count += 1
                        except Exception as del_e:
                            print(f"  └─ 删除失败: {filename} - {del_e}")
                    
            except Exception as e:
                failed_files.append((filename, str(e)))
                print(f"✗ 转换失败: {filename} - {e}")
    
    # 输出转换结果统计
    print(f"\n转换完成!")
    print(f"成功转换: {converted_count} 个文件")
    if delete_png:
        print(f"成功删除: {deleted_count} 个PNG文件")
    if failed_files:
        print(f"转换失败: {len(failed_files)} 个文件")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")

if __name__ == "__main__":
    # 在这里修改您的目录路径
    input_directory = r"C:\Users\Administrator\Desktop\project\electronic_design\yolov5\animals\images"
    output_directory = None  # None表示在原目录生成，也可以指定其他目录
    jpg_quality = 95  # JPG质量 1-100
    delete_original_png = True  # 设置为True将删除原PNG文件
    
    print(f"开始转换PNG到JPG...")
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory or input_directory}")
    print(f"JPG质量: {jpg_quality}")
    print(f"删除原PNG: {'是' if delete_original_png else '否'}")
    print("-" * 50)
    
    convert_png_to_jpg(input_directory, output_directory, jpg_quality, delete_original_png)
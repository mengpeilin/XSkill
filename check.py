import zarr
import numpy as np

def inspect_zarr(path):
    try:
        # 打开 zarr 文件 (自动检测格式)
        data = zarr.open(path, mode='r')
        
        print(f"--- 正在查看 Zarr 文件: {path} ---")
        
        # 定义递归遍历函数
        def print_structure(obj, indent=0):
            prefix = "  " * indent
            if isinstance(obj, zarr.hierarchy.Group):
                print(f"{prefix}📂 Group: {obj.name or '/'}")
                # 遍历组内的所有成员
                for name, child in obj.items():
                    print_structure(child, indent + 1)
            elif isinstance(obj, zarr.core.Array):
                print(f"{prefix}🔢 Array: {obj.name} | Shape: {obj.shape} | Dtype: {obj.dtype} | Chunks: {obj.chunks}")
                # 如果数组不大，可以打印前几个数值（可选）
                # print(f"{prefix}   Data Sample: {obj[...] if obj.size < 100 else obj[0:1]}")

        print_structure(data)
        print("-" * (len(path) + 20))

    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    # 替换为你实际的 zarr 文件路径
    zarr_path = "./dataset/processed/pick_red_mug/robot.zarr" 
    inspect_zarr(zarr_path)

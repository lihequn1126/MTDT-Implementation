import os
import subprocess
import sys
import time
regions = [
    "baoxing",
    "hanyuan",
    "mingshan",
    "tianquan",
    "yingjing",
    "yucheng"
]

models = [
    "autoformer",
    "dlinear",
    "gru",
    "itransformer",
    "lstm",
    "mtdt"
    "transformer",
    "micn",
    "patchtst",
    "timesnet"
]

base_nwp_dir = "../nwpData"
base_load_dir = "../LoadData"
base_result_dir = "../result"

def main():
    total_tasks = len(regions) * len(models)
    current_task = 0
    
    print(f"=== 开始批量训练任务 ===")
    print(f"共 {len(regions)} 个地区, {len(models)} 个模型")
    print(f"总任务数: {total_tasks}")
    print("="*30)

    for region in regions:
        print(f"\n>>>>>> 正在处理地区: {region} <<<<<<")
        
        for model in models:
            current_task += 1
            print(f"[{current_task}/{total_tasks}] 正在运行模型: {model} ...")
            
            # 构造具体的文件路径
            # 例如: ../nwpData/baoxing.csv
            nwp_path = f"{base_nwp_dir}/{region}.csv"
            load_path = f"{base_load_dir}/{region}.csv"
            
            # 构造输出目录
            # 例如: ../result/baoxing_autoformer
            output_dir = f"{base_result_dir}/{region}_{model}"
            
            # 构造 Python 命令
            # 相当于在命令行执行: python autoformer.py --nwp_path ... --load_path ... --output_dir ...
            cmd = [
                sys.executable, f"{model}.py",  # 使用当前环境的 python 解释器
                "--nwp_path", nwp_path,
                "--load_path", load_path,
                "--output_dir", output_dir
            ]
            
            try:
                # 执行命令并等待完成
                # capture_output=False 表示直接把模型的输出打印到控制台，方便你看到训练进度
                subprocess.run(cmd, check=True)
                print(f"{region} - {model} 运行完成!")
                
            except subprocess.CalledProcessError as e:
                print(f"{region} - {model} 运行出错! 错误代码: {e.returncode}")
                # 如果你想出错后继续跑下一个，就pass；如果想出错停止，就raise
                # pass 
            except Exception as e:
                print(f"发生未知错误: {e}")

    print("\n" + "="*30)
    print("所有任务全部结束！")

if __name__ == "__main__":
    if not os.path.exists("autoformer.py"):
        print("错误: 请将此脚本放在 model 文件夹下运行 ")
    else:
        main()
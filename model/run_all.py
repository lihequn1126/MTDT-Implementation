import os
import subprocess
import sys
import time
regions = [
    "hanyuan",
    "lushan",
    "mingshan",
    "shimian",
    "tianquan",
    "yucheng"
]

models = [
    "autoformer",
    "dlinear",
    "gru",
    "itransformer",
    "lstm",
    "mtdt",
    "transformer"
]

base_nwp_dir = "../nwpData"
base_load_dir = "../LoadData"
base_result_dir = "../result"

def main():
    total_tasks = len(regions) * len(models)
    current_task = 0
    
    print(f"=== 开始训练 ===")
    print(f"共 {len(regions)} 个地区, {len(models)} 个模型")
    print(f"总任务数: {total_tasks}")

    for region in regions:
        print(f"\n>>>>>> 正在处理地区: {region} <<<<<<")
        
        for model in models:
            current_task += 1
            print(f"[{current_task}/{total_tasks}] 正在运行模型: {model} ...")
        
            nwp_path = f"{base_nwp_dir}/{region}.csv"
            load_path = f"{base_load_dir}/{region}.csv"
            
            output_dir = f"{base_result_dir}/{region}_{model}"
            
     
            cmd = [
                sys.executable, f"{model}.py", 
                "--nwp_path", nwp_path,
                "--load_path", load_path,
                "--output_dir", output_dir
            ]
        
            try:
                subprocess.run(cmd, check=True)
                print(f"{region} - {model} 运行完成!")
                
            except subprocess.CalledProcessError as e:
                print(f"{region} - {model} 运行出错! 错误代码: {e.returncode}")
            
            except Exception as e:
                print(f"发生未知错误: {e}")

    print("\n" + "="*30)
    print("所有任务全部结束！")

if __name__ == "__main__":
    if not os.path.exists("autoformer.py"):
        print("错误: 请将此脚本放在 model 文件夹下运行 ")
    else:
        main()
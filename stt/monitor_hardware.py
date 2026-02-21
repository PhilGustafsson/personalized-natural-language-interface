import time
import subprocess
import psutil

def get_gpu_usage():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"]
        )
        used, total = result.decode("utf-8").strip().split(", ")
        return int(used), int(total)
    except Exception as e:
        return None, None

def get_ram_usage():
    mem = psutil.virtual_memory()
    return mem.used / (1024 ** 3), mem.total / (1024 ** 3)

print("Monitoring GPU and RAM usage... (Ctrl+C to stop)")
while True:
    gpu_used, gpu_total = get_gpu_usage()
    ram_used, ram_total = get_ram_usage()

    if gpu_used is not None:
        print(f"GPU: {gpu_used} / {gpu_total} MB | RAM: {ram_used:.2f} / {ram_total:.2f} GB", end="\r")
    else:
        print(f"RAM: {ram_used:.2f} / {ram_total:.2f} GB (GPU info unavailable)", end="\r")

    time.sleep(2)

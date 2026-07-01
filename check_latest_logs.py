import os

# 训练结果目录
runs_dir = os.path.join("runs", "detect")

if os.path.exists(runs_dir):
    # 获取所有训练目录并按修改时间排序
    train_dirs = []
    for item in os.listdir(runs_dir):
        item_path = os.path.join(runs_dir, item)
        if os.path.isdir(item_path):
            # 获取目录的修改时间
            mtime = os.path.getmtime(item_path)
            train_dirs.append((mtime, item))

    # 按修改时间排序，最新的在前面
    train_dirs.sort(reverse=True)

    if train_dirs:
        latest_dir = train_dirs[0][1]
        log_path = os.path.join(runs_dir, latest_dir, "train.log")

        if os.path.exists(log_path):
            print(f"读取最新训练日志: {log_path}")
            print("=" * 80)
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                # 显示最后300行
                for line in lines[-300:]:
                    print(line.rstrip())
        else:
            print(f"最新训练目录 {latest_dir} 中没有train.log文件")
            # 查看目录内容
            latest_dir_path = os.path.join(runs_dir, latest_dir)
            print("目录内容:")
            for item in os.listdir(latest_dir_path):
                print(f"- {item}")
    else:
        print("没有找到训练结果目录")
else:
    print("runs/detect目录不存在")

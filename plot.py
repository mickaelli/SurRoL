import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_progress(log_dir):
    csv_path = os.path.join(log_dir, 'progress.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 提取关键指标
    steps = df['total/steps']
    actor_loss = df['train/loss_actor']
    critic_loss = df['train/loss_critic']
    returns = df['rollout/return']

    # 创建画布
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 绘制 Actor Loss
    axes[0].plot(steps, actor_loss, alpha=0.3, color='blue')
    axes[0].plot(steps, actor_loss.rolling(window=5).mean(), color='blue', label='Actor Loss (Smooth)')
    axes[0].set_ylabel('Actor Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制 Critic Loss
    axes[1].plot(steps, critic_loss, alpha=0.3, color='red')
    axes[1].plot(steps, critic_loss.rolling(window=5).mean(), color='red', label='Critic Loss (Smooth)')
    axes[1].set_ylabel('Critic Loss')
    axes[1].legend()
    axes[1].grid(True)

    # 绘制 Returns
    axes[2].plot(steps, returns, alpha=0.3, color='green')
    axes[2].plot(steps, returns.rolling(window=5).mean(), color='green', label='Return (Smooth)')
    axes[2].set_ylabel('Reward / Return')
    axes[2].set_xlabel('Total Steps')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.suptitle(f'Training Progress: {os.path.basename(log_dir)}', y=1.02)
    
    # 保存图片
    # output_plot = os.path.join(log_dir, 'loss_plot.png')
    # plt.savefig(output_plot)
    # print(f"Plot saved to: {output_plot}")
    plt.show()

if __name__ == "__main__":
    # 指定日志目录
    log_directory = "./logs/ddpg/ActiveTrack-1e5_0"
    plot_progress(log_directory)
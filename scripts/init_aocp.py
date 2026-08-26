"""
AOCP-Net 预训练权重初始化脚本

生成随机初始化的权重用于测试和演示
实际训练需要使用标注数据集
"""

import torch
import yaml
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    # 加载配置
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 创建AOCP-Net
    from src.alignment.articulatory_analyzer import AOCPNet

    model = AOCPNet(config["alignment"])

    # 保存随机初始化的权重
    save_dir = PROJECT_ROOT / "models" / "aocp_net"
    os.makedirs(save_dir, exist_ok=True)

    save_path = save_dir / "aocp_net_init.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config["alignment"]["aocp"],
        "model_info": {
            "name": "AOCP-Net",
            "version": "1.0.0",
            "description": "Articulatory Open-Close Perception Network",
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    }, save_path)

    print(f"✅ AOCP-Net 初始权重已保存: {save_path}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 测试前向传播
    batch_size = 2
    n_mels = config["alignment"]["aocp"]["n_mels"]
    seq_len = 200
    mel = torch.randn(batch_size, n_mels, seq_len)

    model.eval()
    with torch.no_grad():
        openness, states, boundaries = model(mel)

    print(f"\n📊 前向传播测试:")
    print(f"   输入: {mel.shape}")
    print(f"   开合度输出: {openness.shape} (范围: {openness.min():.3f} ~ {openness.max():.3f})")
    print(f"   状态输出: {states.shape}")
    print(f"   边界输出: {boundaries.shape}")


if __name__ == "__main__":
    main()

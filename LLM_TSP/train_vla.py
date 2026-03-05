"""
train_vla.py — 训练 VLA Agent 的 Action Module (冻结 LLM 全部参数)
====================================================================

训练目标:
  仅训练 Action Query Embeddings + BridgeActionHead
  LLM (Qwen3-8B) 和 ViT 全部冻结

数据来源:
  1. TSPLIB .tsp 文件 → 节点坐标
  2. LKH 初始解 .json → 路线
  3. llm_selections.csv → 历史选区坐标 + gain (>0 为正样本)

用法:
  python train_vla.py --epochs 50

  # 自定义路径
  python train_vla.py \
      --model_path /path/to/InternVL3_5-8B-Flash \
      --tsplib_dir /path/to/tsplib_repo \
      --lkh_dir /path/to/LKH_solutions \
      --runs_dir /path/to/runs \
      --epochs 100 --lr 2e-4 --batch_size 4
"""

import argparse
import ast
import csv
import json
import os
import sys
import glob
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "LLM_TSP"))

from helper.parse_instances import FileParser
from helper.plot_solution import SolutionPlot
from LLM_TSP.vla_agent import InternVL_VLA_Agent, plotly_fig_to_pil


# ============================================================
# Dataset: 从历史运行记录中收集 (image, gt_coords) 样本
# ============================================================
class TSPVLADataset(Dataset):
    """
    从 TSPLIB 实例 + LKH 解 + LLM 选区记录中构建训练样本。

    每个样本 = (PIL Image of TSP route, 归一化区域坐标 [4])

    样本来源:
      - llm_selections.csv 中 gain > 0 的记录 → 正样本 (有效的选区)
      - 可选: gain == 0 但 num_removed_nodes > threshold → 难负样本

    归一化方式:
      coords (x_min, x_max, y_min, y_max) → 除以图像尺寸
      注意: CSV 中坐标顺序是 (x_min, x_max, y_min, y_max)
            VLA 输出顺序是 (x_min, y_min, x_max, y_max)
    """

    def __init__(
        self,
        tsplib_dir: str,
        lkh_dir: str,
        runs_dir: str,
        min_gain: float = 0,
        image_size: int = 1000,
        cache_images: bool = True,
    ):
        super().__init__()
        self.tsplib_dir = tsplib_dir
        self.lkh_dir = lkh_dir
        self.runs_dir = runs_dir
        self.min_gain = min_gain
        self.image_size = image_size
        self.cache_images = cache_images

        self.file_parser = FileParser()
        self.plotter = SolutionPlot()

        # 收集所有样本: list of (instance_name, route, coordinates, boundary, norm_coords)
        self.samples = []
        self._image_cache = {}
        self._collect_samples()
        print(f"[Dataset] 共收集 {len(self.samples)} 个训练样本")

    def _collect_samples(self):
        """扫描 runs 目录，收集所有 gain > min_gain 的选区样本"""
        run_dirs = sorted(glob.glob(os.path.join(self.runs_dir, "*")))

        for run_dir in run_dirs:
            if not os.path.isdir(run_dir):
                continue

            csv_path = os.path.join(run_dir, "llm_selections.csv")
            if not os.path.exists(csv_path):
                continue

            # 从目录名解析实例名: e.g. "pr1002_20260303_052330" → "pr1002"
            dir_name = os.path.basename(run_dir)
            instance_name = dir_name.rsplit("_", 2)[0]  # 去掉 _日期_时间

            # 加载 TSP 实例
            tsp_path = os.path.join(self.tsplib_dir, f"{instance_name}.tsp")
            if not os.path.exists(tsp_path):
                print(f"  [跳过] 找不到 TSP 文件: {tsp_path}")
                continue

            # 加载 LKH 初始解
            lkh_path = os.path.join(self.lkh_dir, f"{instance_name}_solution.json")
            if not os.path.exists(lkh_path):
                print(f"  [跳过] 找不到 LKH 解: {lkh_path}")
                continue

            try:
                instance_info = self.file_parser.parse_instance_from_file(tsp_path)
            except Exception as e:
                print(f"  [跳过] 解析 {tsp_path} 出错: {e}")
                continue

            coordinates = instance_info["COORDINATES"]
            num_nodes = len(coordinates)

            with open(lkh_path, "r") as f:
                lkh_data = json.load(f)
            route = lkh_data["current_route"]

            # 计算边界
            x_min = min(c[0] for c in coordinates)
            x_max = max(c[0] for c in coordinates)
            y_min = min(c[1] for c in coordinates)
            y_max = max(c[1] for c in coordinates)
            boundary = (x_min, x_max, y_min, y_max)

            # 读取 CSV 收集正样本
            sample_count = 0
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gain = float(row["gain"])
                    if gain <= self.min_gain:
                        continue

                    # 解析坐标列表
                    try:
                        coord_list = ast.literal_eval(row["coordinates"])
                    except (ValueError, SyntaxError):
                        continue

                    # 每个选区都是一个独立样本
                    for coords_tuple in coord_list:
                        if len(coords_tuple) != 4:
                            continue

                        # CSV 中: (x_min, x_max, y_min, y_max)
                        cx_min, cx_max, cy_min, cy_max = coords_tuple

                        # 归一化到 [0,1] — VLA 输出顺序: (x_min, y_min, x_max, y_max)
                        # 使用实例边界范围做归一化
                        x_range = x_max - x_min if x_max > x_min else 1
                        y_range = y_max - y_min if y_max > y_min else 1

                        norm = [
                            (cx_min - x_min) / x_range,
                            (cy_min - y_min) / y_range,
                            (cx_max - x_min) / x_range,
                            (cy_max - y_min) / y_range,
                        ]
                        # Clamp to [0, 1]
                        norm = [max(0.0, min(1.0, v)) for v in norm]

                        self.samples.append({
                            "instance_name": instance_name,
                            "route": route,
                            "coordinates": coordinates,
                            "boundary": boundary,
                            "num_nodes": num_nodes,
                            "norm_coords": norm,
                            "gain": gain,
                        })
                        sample_count += 1

            if sample_count > 0:
                print(f"  [√] {instance_name}: {sample_count} 个正样本 (gain > {self.min_gain})")

    def _render_image(self, instance_name, route, coordinates, boundary, num_nodes):
        """渲染 TSP 路线图并返回 PIL Image"""
        cache_key = f"{instance_name}_{id(route)}"
        if self.cache_images and cache_key in self._image_cache:
            return self._image_cache[cache_key]

        x_min, x_max, y_min, y_max = boundary

        # 使用 Plotly 渲染
        # plot_tsp_solution_plotly 需要一个 args 参数但实际不使用
        fig = self.plotter.plot_tsp_solution_plotly(
            args=None,
            routes=route,
            coordinates=coordinates,
            x_max=x_max, y_max=y_max,
            x_min=x_min, y_min=y_min,
            num_nodes=num_nodes,
        )

        image = plotly_fig_to_pil(fig)

        if self.cache_images:
            self._image_cache[cache_key] = image

        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = self._render_image(
            sample["instance_name"],
            sample["route"],
            sample["coordinates"],
            sample["boundary"],
            sample["num_nodes"],
        )

        norm_coords = torch.tensor(sample["norm_coords"], dtype=torch.float32)

        return image, norm_coords


def collate_fn(batch):
    """自定义 collate: 返回 PIL Image 列表 + coords tensor"""
    images, coords = zip(*batch)
    return list(images), torch.stack(coords, dim=0)


# ============================================================
# 数据增强: 随机裁剪 / 翻转 区域坐标 (可选)
# ============================================================
def augment_coords(norm_coords: torch.Tensor, p_flip: float = 0.3):
    """
    对归一化坐标做随机水平/垂直翻转增强。
    norm_coords: [batch, 4] — (x_min, y_min, x_max, y_max) in [0,1]
    """
    if random.random() < p_flip:
        # 水平翻转: x → 1-x, swap x_min/x_max
        x_min = 1.0 - norm_coords[:, 2]
        x_max = 1.0 - norm_coords[:, 0]
        norm_coords = torch.stack([x_min, norm_coords[:, 1], x_max, norm_coords[:, 3]], dim=1)

    if random.random() < p_flip:
        # 垂直翻转: y → 1-y, swap y_min/y_max
        y_min = 1.0 - norm_coords[:, 3]
        y_max = 1.0 - norm_coords[:, 1]
        norm_coords = torch.stack([norm_coords[:, 0], y_min, norm_coords[:, 2], y_max], dim=1)

    return norm_coords


# ============================================================
# 训练器
# ============================================================
class VLATrainer:
    """
    VLA Action Module 训练器。

    训练范围 (冻结 LLM):
      - Action Query Embeddings   (~16K params)
      - BridgeActionHead           (~600M params, 12 blocks × 4096 dim)
    """

    def __init__(self, args):
        self.args = args

        # 创建 Agent (freeze_backbone=True, 不使用 LoRA)
        self.agent = InternVL_VLA_Agent(
            model_path=args.model_path,
            action_dim=4,
            num_action_tokens=args.num_action_tokens,
            num_bridge_blocks=args.num_bridge_blocks,
            num_attn_heads=args.num_attn_heads,
            freeze_backbone=True,
            use_lora=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # 触发模型加载
        self.agent._lazy_load()

        # 构建 optimizer — 只优化可训练参数
        self.optimizer = torch.optim.AdamW(
            self.agent.trainable_parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )

        # 训练 prompt（固定的，用于所有样本）
        self.train_prompt = (
            "You are viewing a Traveling Salesman Problem route visualization. "
            "Select a sub-region where the route can be significantly optimized. "
            "Output the bounding box coordinates."
        )

        # 预编码 prompt tokens (因为同一 prompt 用于所有样本)
        self._prompt_ids = None

    def _get_prompt_ids(self):
        """缓存 prompt 的 token ids"""
        if self._prompt_ids is None:
            self._prompt_ids = self.agent._tokenizer(
                self.train_prompt, return_tensors="pt"
            ).input_ids
            if torch.cuda.is_available():
                self._prompt_ids = self._prompt_ids.cuda()
        return self._prompt_ids

    def train_one_epoch(self, dataloader, epoch: int):
        """训练一个 epoch"""
        self.agent.train()
        # 冻结部分始终 eval
        self.agent._model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, gt_coords) in enumerate(dataloader):
            # gt_coords: [batch, 4]
            if torch.cuda.is_available():
                gt_coords = gt_coords.cuda()

            # 可选数据增强
            if self.args.augment:
                gt_coords = augment_coords(gt_coords)

            batch_loss = 0.0

            # 逐样本 forward (因为图像尺寸各异，需要各自 preprocess)
            for i, image in enumerate(images):
                pv = self.agent.preprocess_image(image)
                prompt_ids = self._get_prompt_ids()

                pred = self.agent.forward(pv, prompt_ids)  # [1, 4]
                gt = gt_coords[i:i+1].to(pred.dtype)       # [1, 4]

                loss = F.l1_loss(pred, gt)
                batch_loss += loss

            batch_loss = batch_loss / len(images)

            self.optimizer.zero_grad()
            batch_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.agent.trainable_parameters(), max_norm=1.0,
            )

            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

            if (batch_idx + 1) % self.args.log_interval == 0:
                avg = total_loss / num_batches
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {batch_loss.item():.6f} | Avg: {avg:.6f}")

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader):
        """验证集评估"""
        self.agent.eval()
        total_loss = 0.0
        num_samples = 0

        for images, gt_coords in dataloader:
            if torch.cuda.is_available():
                gt_coords = gt_coords.cuda()

            for i, image in enumerate(images):
                pv = self.agent.preprocess_image(image)
                prompt_ids = self._get_prompt_ids()
                pred = self.agent.forward(pv, prompt_ids)
                gt = gt_coords[i:i+1].to(pred.dtype)
                loss = F.l1_loss(pred, gt)
                total_loss += loss.item()
                num_samples += 1

        return total_loss / max(num_samples, 1)

    def train(self, train_loader, val_loader=None):
        """完整训练循环"""
        best_val_loss = float("inf")
        save_dir = self.args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        print("=" * 60)
        print("开始训练 VLA Action Module")
        print(f"  Epochs:           {self.args.epochs}")
        print(f"  学习率:            {self.args.lr}")
        print(f"  Bridge Blocks:    {self.args.num_bridge_blocks}")
        print(f"  Action Tokens:    {self.args.num_action_tokens}")
        print(f"  Attention Heads:  {self.args.num_attn_heads}")
        print(f"  训练样本数:        {len(train_loader.dataset)}")
        if val_loader:
            print(f"  验证样本数:        {len(val_loader.dataset)}")
        print(f"  保存目录:          {save_dir}")
        print("=" * 60)

        for epoch in range(self.args.epochs):
            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch)
            elapsed = time.time() - t0

            log = f"Epoch {epoch+1}/{self.args.epochs} | Train L1: {train_loss:.6f} | Time: {elapsed:.1f}s"

            if val_loader:
                val_loss = self.validate(val_loader)
                log += f" | Val L1: {val_loss:.6f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.agent.save_trainable(os.path.join(save_dir, "best.pt"))
                    log += " ★ best"
            else:
                # 没有验证集时，每个 epoch 都保存
                self.agent.save_trainable(os.path.join(save_dir, f"epoch_{epoch+1}.pt"))

            print(log)
            self.scheduler.step()

        # 最终保存
        self.agent.save_trainable(os.path.join(save_dir, "final.pt"))
        print(f"\n训练完成。最终模型保存到 {save_dir}/final.pt")
        if val_loader:
            print(f"最佳验证 L1 Loss: {best_val_loss:.6f}")


# ============================================================
# 主入口
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train VLA Action Module (frozen LLM)")

    # 路径
    parser.add_argument("--model_path", type=str,
                        default="/sciclone/home/yli95/codes/vitsp/InternVL3_5-8B-Flash",
                        help="InternVL3.5-8B-Flash 模型路径")
    parser.add_argument("--tsplib_dir", type=str,
                        default="/sciclone/home/yli95/codes/vitsp/instances/tsplib/tsplib_repo",
                        help="TSPLIB .tsp 文件目录")
    parser.add_argument("--lkh_dir", type=str,
                        default="/sciclone/home/yli95/codes/vitsp/experiments/LKH_solutions",
                        help="LKH 初始解 .json 目录")
    parser.add_argument("--runs_dir", type=str,
                        default="/sciclone/home/yli95/codes/vitsp/experiments/runs",
                        help="实验运行记录目录 (含 llm_selections.csv)")

    # 模型结构
    parser.add_argument("--num_action_tokens", type=int, default=4)
    parser.add_argument("--num_bridge_blocks", type=int, default=12)
    parser.add_argument("--num_attn_heads", type=int, default=8)

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="每批样本数 (受显存限制，建议 1-4)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 工作进程数 (0=主进程)")
    parser.add_argument("--min_gain", type=float, default=0,
                        help="最小 gain 阈值，>0 只用改善了路线的选区")

    # 其他
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="验证集比例 (0 = 不使用验证集)")
    parser.add_argument("--augment", action="store_true",
                        help="启用坐标增强 (随机翻转)")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="每 N 个 batch 打印一次")
    parser.add_argument("--save_dir", type=str, default="checkpoints/vla_action",
                        help="模型保存目录")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 构建数据集
    print("[1/4] 构建数据集 ...")
    dataset = TSPVLADataset(
        tsplib_dir=args.tsplib_dir,
        lkh_dir=args.lkh_dir,
        runs_dir=args.runs_dir,
        min_gain=args.min_gain,
    )

    if len(dataset) == 0:
        print("错误: 没有找到任何训练样本！请检查:")
        print(f"  - runs_dir: {args.runs_dir}")
        print(f"  - tsplib_dir: {args.tsplib_dir}")
        print(f"  - lkh_dir: {args.lkh_dir}")
        print(f"  - min_gain: {args.min_gain}")
        sys.exit(1)

    # 训练/验证拆分
    if args.val_split > 0 and len(dataset) > 5:
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"[2/4] 数据拆分: {train_size} 训练 / {val_size} 验证")
    else:
        train_dataset = dataset
        val_dataset = None
        print(f"[2/4] 全部 {len(dataset)} 样本用于训练 (无验证集)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )

    # 创建训练器
    print("[3/4] 初始化 VLA Agent (加载模型) ...")
    trainer = VLATrainer(args)

    # 开始训练
    print("[4/4] 开始训练 ...")
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

# LLM大海捞针实验的批量可视化和文件存储

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import glob
from typing import List, Dict, Optional
from tqdm import tqdm  

class NeedleVisualizer:
    def __init__(self, base_result_dir: str, exclude_dirs: List[str] = None, force_reprocess: bool = False):
        """
        初始化可视化器
        @param base_result_dir: 结果的根目录
        @param exclude_dirs: 需要排除的文件夹列表
        @param force_reprocess: 是否强制重新处理所有实验，即使已经存在可视化结果
        """
        self.base_result_dir = base_result_dir
        self.visualization_dir = os.path.join(base_result_dir, "visualizations")
        self.exclude_dirs = exclude_dirs or []
        self.force_reprocess = force_reprocess
        self.processed_experiments = set()  # 用于记录已处理的实验
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # 加载已有的处理记录
        self.record_file = os.path.join(self.visualization_dir, "processed_experiments.json")
        self._load_processed_records()

    def _load_processed_records(self):
        """加载已处理实验的记录"""
        try:
            if os.path.exists(self.record_file):
                with open(self.record_file, 'r') as f:
                    self.processed_experiments = set(json.load(f))
        except Exception as e:
            print(f"加载处理记录时出错: {str(e)}")
            self.processed_experiments = set()

    def _save_processed_records(self):
        """保存已处理实验的记录"""
        try:
            with open(self.record_file, 'w') as f:
                json.dump(list(self.processed_experiments), f)
        except Exception as e:
            print(f"保存处理记录时出错: {str(e)}")

    def _should_process_experiment(self, experiment_path: str) -> bool:
        """判断是否需要处理该实验"""
        dataset = self._extract_dataset_name(experiment_path)
        model_name = self._extract_model_name(experiment_path)
        
        if self.force_reprocess:
            return True
        
        if experiment_path in self.processed_experiments:
            return False
        
        # 检查热力图文件是否存在
        filename = f'{dataset}_{model_name}_heatmap.png'
        vis_path = os.path.join(self.visualization_dir, filename)
        original_path = os.path.join(experiment_path, filename)
        
        return not (os.path.exists(vis_path) and os.path.exists(original_path))

    def load_experiment_data(self, experiment_path: str) -> pd.DataFrame:
        """加载单个实验的数据"""
        data = []
        
        json_files = glob.glob(f"{experiment_path}/*.json") + glob.glob(f"{experiment_path}/*.jsonl")
        
        if not json_files:
            tqdm.write(f"警告: 在 {os.path.basename(experiment_path)} 下没有找到JSON文件")
            return pd.DataFrame()
        
        # 处理找到的JSON文件
        for file in tqdm(json_files, desc="处理JSON文件", unit="file", leave=False):
            try:
                with open(file, 'r') as f:
                    json_data = json.load(f)
                    # 验证必需的字段是否存在
                    required_fields = ["depth_percent", "context_length", "score"]
                    if not all(field in json_data for field in required_fields):
                        tqdm.write(f"警告: {os.path.basename(file)} 缺少必需的字段")
                        continue
                    
                    data.append({
                        "Document Depth": json_data["depth_percent"],
                        "Context Length": json_data["context_length"],
                        "Score": json_data["score"],
                        "Dataset": self._extract_dataset_name(experiment_path),
                        "Model": self._extract_model_name(experiment_path),  # 直接使用文件夹名作为模型名
                    })
            except json.JSONDecodeError:
                tqdm.write(f"错误: 无法解析JSON文件 {os.path.basename(file)}")
            except Exception as e:
                tqdm.write(f"处理文件 {os.path.basename(file)} 时发生错误: {str(e)}")
        
        df = pd.DataFrame(data)
        if df.empty:
            tqdm.write(f"警告: 从 {os.path.basename(experiment_path)} 加载的数据为空")
        return df

    def create_heatmap(self, df: pd.DataFrame, save_path: str, title: str):
        """生成热力图"""
        if df.empty:
            tqdm.write(f"警告: 跳过空数据集的热力图生成 {os.path.basename(save_path)}")
            return
        
        try:
            pivot_table = pd.pivot_table(
                df, 
                values='Score', 
                index=['Document Depth', 'Context Length'], 
                aggfunc='mean'
            ).reset_index()
            
            # 将数据重新组织成热力图所需的格式
            pivot_table = pivot_table.pivot(
                index="Document Depth",
                columns="Context Length",
                values="Score"
            )
            
            plt.figure(figsize=(17.5, 8))
            cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
            
            sns.heatmap(
                pivot_table,
                fmt="",  # 移除数值标注
                cmap=cmap,
                cbar_kws={'label': 'Score'},
                vmin=0,
                vmax=10,
                annot=False  # 不显示具体数值
            )

            plt.title(title)
            plt.xlabel('Token Limit')
            plt.ylabel('Depth Percent')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # 确保保存路径的目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            tqdm.write(f"成功生成热力图: {os.path.basename(save_path)}")
            
        except Exception as e:
            tqdm.write(f"生成热力图时发生错误 {os.path.basename(save_path)}: {str(e)}")
            plt.close()  # 确保关闭图形，即使发生错误

    def process_all_experiments(self):
        """处理所有实验数据"""
        all_data = pd.DataFrame()
        experiment_paths = self._get_experiment_paths()
        heatmap_count = 0  # 添加热力图计数器
        
        if not experiment_paths:
            print("警告: 没有找到任何实验路径")
            return all_data
        
        # 清理已处理记录中的无效记录
        self._clean_processed_records()
        
        # 过滤出需要处理的实验
        experiments_to_process = [
            path for path in experiment_paths 
            if self._should_process_experiment(path)
        ]
        
        if not experiments_to_process:
            print("所有实验都已处理完成，无需重新处理")
            return self._load_existing_data()
            
        print(f"找到 {len(experiments_to_process)} 个需要处理的实验路径")
        
        for experiment_path in tqdm(experiments_to_process, desc="处理实验", unit="exp"):
            try:
                df = self.load_experiment_data(experiment_path)
                if not df.empty:
                    all_data = pd.concat([all_data, df], ignore_index=True)
                    
                    dataset = self._extract_dataset_name(experiment_path)
                    model_name = self._extract_model_name(experiment_path)
                    
                    title = f'{dataset} - {model_name}'
                    filename = f'{dataset}_{model_name}_heatmap.png'
                    
                    original_save_path = os.path.join(experiment_path, filename)
                    vis_save_path = os.path.join(self.visualization_dir, filename)
                    
                    self.create_heatmap(df, original_save_path, title)
                    self.create_heatmap(df, vis_save_path, title)
                    heatmap_count += 2
                    
                    self.processed_experiments.add(experiment_path)
                    
            except Exception as e:
                tqdm.write(f"处理实验 {experiment_path} 时发生错误: {str(e)}")
        
        # 保存处理记录
        self._save_processed_records()
        
        # 打印热力图生成统计信息
        print(f"\n已生成 {heatmap_count} 张热力图")
        
        # 合并已有数据和新数据
        if not all_data.empty:
            existing_data = self._load_existing_data()
            if not existing_data.empty:
                all_data = pd.concat([existing_data, all_data], ignore_index=True)
            
            summary_path = os.path.join(self.visualization_dir, 'llm_all_experiments_data.csv')
            try:
                all_data.to_csv(summary_path, index=False)
                print(f"汇总数据已保存到: {summary_path}")
            except Exception as e:
                print(f"保存汇总数据时发生错误: {str(e)}")
        
        return all_data

    def _load_existing_data(self) -> pd.DataFrame:
        """加载已存在的汇总数据"""
        summary_path = os.path.join(self.visualization_dir, 'llm_all_experiments_data.csv')
        if os.path.exists(summary_path):
            try:
                return pd.read_csv(summary_path)
            except Exception as e:
                print(f"加载已存在的汇总数据时出错: {str(e)}")
        return pd.DataFrame()

    def _get_experiment_paths(self) -> List[str]:
        """获取所有实验路径"""
        # 首先获取所有 case_dirs（数据集目录）
        case_dirs = [d for d in glob.glob(os.path.join(self.base_result_dir, "*"))
                    if os.path.isdir(d) and not any(exclude_dir in d for exclude_dir in self.exclude_dirs)]
        print(f"找到 {len(case_dirs)} 个数据集目录")
        
        # 对于每个数据集目录，获取其下的模型目录
        experiment_paths = []
        for case_dir in tqdm(case_dirs, desc="扫描数据集", unit="dir"):
            model_dirs = [d for d in glob.glob(os.path.join(case_dir, "*"))
                         if os.path.isdir(d)]
            experiment_paths.extend(model_dirs)
        
        print(f"总共找到 {len(experiment_paths)} 个实验路径")
        return experiment_paths

    @staticmethod
    def _extract_dataset_name(path: str) -> str:
        """从路径中提取数据集名称（case_name）"""
        # 返回倒数第二级目录名
        parts = path.split(os.sep)
        return parts[-2]

    @staticmethod
    def _extract_model_name(path: str) -> str:
        """从路径中提取模型名称"""
        return os.path.basename(path)

    def _clean_processed_records(self):
        """清理已处理记录中的无效记录"""
        invalid_records = set()
        
        for experiment_path in tqdm(self.processed_experiments, desc="验证实验记录", unit="exp"):
            dataset = self._extract_dataset_name(experiment_path)
            model_name = self._extract_model_name(experiment_path)
            
            # 检查热力图文件是否存在
            filename = f'{dataset}_{model_name}_heatmap.png'
            vis_path = os.path.join(self.visualization_dir, filename)
            original_path = os.path.join(experiment_path, filename)
            
            # 如果热力图不完整，标记为无效记录
            if not (os.path.exists(vis_path) and os.path.exists(original_path)):
                tqdm.write(f"发现无效记录: {dataset}/{model_name}")
                invalid_records.add(experiment_path)
        
        # 从处理记录中移除无效记录
        self.processed_experiments -= invalid_records
        if invalid_records:
            print(f"已清理 {len(invalid_records)} 条无效记录")
            # 保存更新后的记录
            self._save_processed_records()

# 使用示例
if __name__ == "__main__":
    base_dir = ""
    # 如果需要排除某些文件夹，可以添加到 exclude_dirs 列表中
    exclude_dirs = ["visualizations"]

    # #如果需要强制重新处理所有实验，可以设置 force_reprocess=True
    visualizer = NeedleVisualizer(base_dir, exclude_dirs, force_reprocess=True)
    all_experiments_data = visualizer.process_all_experiments()

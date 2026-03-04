"""
阶段 2: MCP 数据爬虫

使用 MCP 服务器从网络获取高质量训练数据，自动分类到不同领域。
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils.logger import Colors, SimpleLogger

# Rich 支持
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# 初始化
if RICH_AVAILABLE:
    console = Console()
else:
    console = None
logger = SimpleLogger()


@dataclass
class CrawlConfig:
    """爬虫配置"""

    # 数据输出目录
    output_dir: str = "data/stage2"

    # 每个领域的数据量
    samples_per_domain: int = 1000

    # 并发请求数
    max_concurrent: int = 5

    # 搜索配置
    search_recency: str = "oneYear"  # oneYear, noLimit
    content_size: str = "medium"  # medium, high

    # 数据质量标准
    min_length: int = 100
    max_length: int = 10000
    min_unique_ratio: float = 0.3  # 最少唯一字符比例

    # 域域配置
    domains: Dict[str, Dict[str, List[str]]] = None

    def __post_init__(self):
        if self.domains is None:
            self.domains = {
                "code": {
                    "name": "代码编程",
                    "experts": [0, 1],
                    "keywords": [
                        "Python 算法 实现",
                        "LeetCode 题解 Python",
                        "machine learning tutorial code",
                        "GitHub popular Python projects",
                        "数据结构与算法 代码",
                    ],
                    "quality_filters": ["def ", "class ", "import ", "function "],
                },
                "writing": {
                    "name": "文学写作",
                    "experts": [2, 3],
                    "keywords": [
                        "小说 写作技巧",
                        "散文 创作 方法",
                        "创意写作 教程",
                        "文章写作 结构",
                        "写作范例 经典",
                    ],
                    "quality_filters": ["描写", "叙述", "情节", "意境"],
                },
                "math": {
                    "name": "数学推理",
                    "experts": [4, 5],
                    "keywords": [
                        "数学证明 方法",
                        "线性代数 教程",
                        "calculus explained",
                        "数学竞赛 解题",
                        "概率统计 基础",
                    ],
                    "quality_filters": ["证明", "定理", "公式", "计算"],
                },
                "knowledge": {
                    "name": "知识问答",
                    "experts": [6, 7],
                    "keywords": [
                        "Wikipedia 百科",
                        "量子力学 解释",
                        "人工智能 历史",
                        "宇宙 奥秘",
                        "生物科学 发现",
                    ],
                    "quality_filters": ["定义", "原理", "分类", "特点"],
                },
            }


class DataQualityChecker:
    """数据质量检查器"""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.seen_hashes = set()  # 用于去重

    def is_valid_length(self, text: str) -> bool:
        """检查长度"""
        return self.config.min_length <= len(text) <= self.config.max_length

    def is_valid_unique_ratio(self, text: str) -> bool:
        """检查唯一字符比例"""
        if len(text) == 0:
            return False
        unique_chars = len(set(text))
        ratio = unique_chars / len(text)
        return ratio >= self.config.min_unique_ratio

    def has_quality_indicators(self, text: str, domain_keywords: List[str]) -> bool:
        """检查质量指标"""
        # 简单启发式：是否包含关键词相关的内容
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in domain_keywords)

    def is_duplicate(self, text: str) -> bool:
        """检查重复"""
        text_hash = hash(text)
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def score_difficulty(self, text: str) -> str:
        """评估文本难度"""
        # 简单启发式规则
        difficulty_score = 0

        # 长度
        if len(text) > 500:
            difficulty_score += 1
        if len(text) > 1000:
            difficulty_score += 1

        # 复杂词汇 (简单列表)
        complex_words = [
            "原理", "机制", "算法", "证明", "推导",
            "application", "implementation", "optimization",
        ]
        text_lower = text.lower()
        for word in complex_words:
            if word in text_lower:
                difficulty_score += 1

        # 代码/公式相关
        if any(c in text for c in ["{", "}", "$$", "```"]):
            difficulty_score += 2

        if difficulty_score <= 2:
            return "simple"
        elif difficulty_score <= 4:
            return "medium"
        else:
            return "hard"


class DataCrawler:
    """MCP 数据爬虫"""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.checker = DataQualityChecker(config)

        # 确保输出目录存在
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    async def search_web(self, query: str, max_results: int = 10) -> List[str]:
        """使用 web-search 搜索"""
        try:
            # 使用 web-search-prime MCP
            from mcp_web_search_prime__webSearchPrime import webSearchPrime

            results = await webSearchPrime(
                search_query=query,
                search_recency_filter=self.config.search_recency,
                content_size=self.config.content_size,
            )

            # 解析结果获取 URL
            urls = []
            if results and "data" in results:
                for item in results["data"][:max_results]:
                    if "url" in item:
                        urls.append(item["url"])

            return urls[:max_results]

        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[dim]搜索失败 ({query}): {e}[/dim]")
            else:
                print(f"搜索失败 ({query}): {e}")
            return []

    async def read_url(self, url: str) -> Optional[str]:
        """使用 web-reader 读取网页"""
        try:
            # 使用 web-reader MCP
            from mcp_web_reader__webReader import webReader

            result = await webReader(
                url=url,
                return_format="markdown",
                retain_images=False,
            )

            if result and "content" in result:
                return result["content"]

            return None

        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[dim]读取失败 ({url[:50]}...): {e}[/dim]")
            return None

    async def crawl_domain(self, domain_key: str, domain_config: Dict) -> Dict[str, List[str]]:
        """爬取单个领域的数据"""
        domain_name = domain_config["name"]
        keywords = domain_config["keywords"]
        quality_filters = domain_config["quality_filters"]

        if RICH_AVAILABLE:
            console.print(f"\n[bold yellow]爬取 {domain_name} 数据...[/bold yellow]")
        else:
            print(f"\n爬取 {domain_name} 数据...")

        all_data = {
            "simple": [],
            "medium": [],
            "hard": [],
        }

        collected = 0
        target = self.config.samples_per_domain

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console if RICH_AVAILABLE else None,
            disable=not RICH_AVAILABLE,
        ) as progress:
            task = progress.add_task(f"收集 {domain_name}", total=target)

            for keyword in keywords:
                if collected >= target:
                    break

                # 搜索
                urls = await self.search_web(keyword, max_results=10)

                # 读取内容
                for url in urls:
                    if collected >= target:
                        break

                    content = await self.read_url(url)

                    if content is None:
                        continue

                    # 质量检查
                    if not self.checker.is_valid_length(content):
                        continue

                    if not self.checker.is_valid_unique_ratio(content):
                        continue

                    if self.checker.is_duplicate(content):
                        continue

                    # 评估难度
                    difficulty = self.checker.score_difficulty(content)

                    # 保存
                    all_data[difficulty].append(content)
                    collected += 1

                    progress.update(task, advance=1)

        return all_data

    def save_domain_data(self, domain_key: str, domain_data: Dict[str, List[str]]):
        """保存领域数据"""
        base_path = Path(self.config.output_dir)

        total_saved = 0
        for difficulty, data in domain_data.items():
            file_path = base_path / f"{domain_key}_{difficulty}.txt"
            file_path = file_path.with_suffix(".txt")

            with open(file_path, 'a', encoding='utf-8') as f:
                for item in data:
                    f.write(item.strip() + '\n')

            total_saved += len(data)

            if RICH_AVAILABLE:
                console.print(f"[dim]  保存 {difficulty}: {len(data)} 条 -> {file_path.name}[/dim]")

        return total_saved

    async def crawl_all(self):
        """爬取所有领域数据"""
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[bold cyan]MCP 数据爬虫[/bold cyan]",
                title="HiveMind",
                subtitle="阶段 2",
                box=box.DOUBLE_EDGE,
                border_style="bright_blue"
            ))
        else:
            print("\n" + "="*50)
            print("  HiveMind - 阶段2: MCP 数据爬虫")
            print("="*50 + "\n")

        results = {}

        for domain_key, domain_config in self.config.domains.items():
            domain_data = await self.crawl_domain(domain_key, domain_config)
            count = self.save_domain_data(domain_key, domain_data)
            results[domain_key] = count

        # 打印总结
        if RICH_AVAILABLE:
            console.print("\n[bold green]✓ 数据爬取完成！[/bold green]")
            table = Table(title="[bold yellow]数据统计[/bold yellow]", box=box.ROUNDED)
            table.add_column("[cyan]领域[/cyan]")
            table.add_column("[yellow]数量[/yellow]")

            for domain, count in results.items():
                table.add_row(domain_config["name"], str(count))

            console.print(table)
        else:
            print("\n✓ 数据爬取完成！")
            for domain, count in results.items():
                print(f"  {domain_config['name']}: {count} 条")


def create_sample_data():
    """创建示例数据（当 MCP 不可用时）"""
    config = CrawlConfig()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    sample_data = {
        "code": [
            "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + [pivot] + quick_sort(right)",
            "class BinarySearchTree:\n    def __init__(self):\n        self.root = None\n    def insert(self, val):\n        if not self.root:\n            self.root = TreeNode(val)\n        else:\n            self._insert(self.root, val)",
        "import numpy as np\n\ndef matrix_multiply(A, B):\n    return np.dot(A, B)\n\n# 使用示例\nA = np.random.rand(3, 3)\nB = np.random.rand(3, 3)\nC = matrix_multiply(A, B)",
        "from collections import deque\n\ndef bfs(graph, start):\n    visited = set()\n    queue = deque([start])\n    while queue:\n        node = queue.popleft()\n        if node not in visited:\n            visited.add(node)\n            queue.extend(graph[node] - visited)",
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "class LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n            return self.cache[key]\n        return None",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b\n\n# 动态规划版本\nfib_cache = {}\ndef fib_dp(n):\n    if n in fib_cache:\n        return fib_cache[n]\n    if n <= 1:\n        return n\n    fib_cache[n] = fib_dp(n-1) + fib_dp(n-2)\n    return fib_cache[n]",
        "import threading\n\nclass ThreadPool:\n    def __init__(self, max_workers):\n        self.max_workers = max_workers\n        self.workers = []\n    def submit(self, fn, *args):\n        # 提交任务到线程池\n        thread = threading.Thread(target=fn, args=args)\n        thread.start()\n        return thread",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
        "def depth_first_search(graph, start, goal):\n    stack = [(start, [start])]\n    while stack:\n        node, path = stack.pop()\n        if node == goal:\n            return path\n        for neighbor in graph.get(node, []):\n            if neighbor not in path:\n                stack.append((neighbor, path + [neighbor]))\n    return None",
        ] * 100,  # 重复生成足够数据
        "writing": [
            "清晨的阳光透过薄雾洒落在湖面上，波光粼粼。远山如黛，近水含烟，构成一幅宁静致远的山水画卷。",
            "秋日的午后，我独自坐在窗前，看着落叶随风起舞。每一片叶子都承载着季节的故事，轻轻诉说着时光的流逝。",
            "雨后的森林格外清新，泥土的芬芳混合着青草的气息。偶尔传来几声鸟鸣，更增添了几分幽静与祥和。",
            "夜幕降临，繁星点点。仰望星空，不禁让人思考：在这浩瀚宇宙中，我们不过是沧海一粟，却依然拥有探索未知的勇气。",
            "春风拂过大地，万物复苏。嫩绿的新芽从土壤中探出头来，宣告着生命的顽强与美丽。这是希望的季节，是梦想启航的时刻。",
            "记忆如同一条长河，有时平静如镜，有时波澜壮阔。我们每个人都是自己故事的作者，用经历书写着独特的篇章。",
            "古城的街巷里，青石板路诉说着岁月的沧桑。老屋的檐角下，燕子依然归来，仿佛在诉说着时光的轮回与永恒。",
            "海边的日出总是那么壮观。当第一缕金光刺破云层，整个世界仿佛都被重新点亮，充满了生机与希望。",
            "书中自有黄金屋，文字有着独特的魔力。一行行诗句，一个个故事，都在我们心中播撒着思想的种子。",
            "生活的美好往往藏在细节之中：一个微笑，一声问候，一次邂逅。珍惜这些瞬间，就是珍惜生活本身。",
        ] * 100,
        "math": [
            "求证: 对于任意正整数 n，有 1 + 2 + ... + n = n(n+1)/2\n\n证明: 使用数学归纳法。\n当 n=1 时，左边=1，右边=1(2)/2=1，等式成立。\n假设 n=k 时成立，即 1+2+...+k = k(k+1)/2\n则当 n=k+1 时，1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2\n得证。",
            "微积分基本定理: ∫[a,b] f(x)dx = F(b) - F(a)，其中 F 是 f 的原函数。\n\n这个定理建立了微分和积分之间的联系，是微积分的核心定理之一。",
            "线性代数中，矩阵乘法满足结合律但不满足交换律。\n即 (AB)C = A(BC)，但通常 AB ≠ BA。\n这是因为矩阵乘法是行向量与列向量的点积运算。",
            "概率论中的贝叶斯定理: P(A|B) = P(B|A)P(A) / P(B)\n\n这个定理描述了在已知 B 发生的情况下 A 发生的条件概率，是统计推断的重要基础。",
            "图论中的欧拉公式: V - E + F = 2\n对于任意连通平面图，顶点数 V 减去边数 E 加上面数 F 等于 2。",
            "拉格朗日中值定理: 若 f(x) 在 [a,b] 上连续且在 (a,b) 上可导，则存在 c∈(a,b) 使得 f'(c) = (f(b)-f(a))/(b-a)。\n几何意义是曲线上至少有一点，其切线平行于连接两端点的弦。",
            "群论的基本概念: 群 G 是一个非空集合，配上一 个二元运算 *，满足封闭性、结合律、存在单位元、存在逆元四个性质。",
            "傅里叶级数展开: f(x) = a₀/2 + Σ(n=1 to ∞)[aₙcos(nx) + bₙsin(nx)]\n它可以将周期函数表示为正弦和余弦函数的无穷级数之和。",
            "拓扑学中的同伦概念: 两个连续映射 f,g: X→Y 称为同伦的，如果存在连续映射 H: X×[0,1]→Y 使得 H(x,0)=f(x), H(x,1)=g(x)。",
            "复变函数的柯西积分公式: f(z₀) = (1/2πi)∮_C f(z)/(z-z₀) dz\n这个公式表明解析函数在区域内某点的值可以通过其在边界上的积分来计算。",
        ] * 100,
        "knowledge": [
            "量子力学是描述微观物质世界的物理学理论，与相对论一起构成现代物理学的理论基础。",
            "DNA（脱氧核糖核酸）是生物体内储存遗传信息的生物大分子，由四种核苷酸（A、T、G、C）组成。",
            "光合作用是植物、藻类利用光能将二氧化碳和水转化为有机物并释放氧气的过程，是地球生命存在的基础。",
            "黑洞是广义相对论预言的一种天体，其引力场极强，连光都无法逃脱。黑洞的存在已被天文观测证实。",
            "神经网络是受人脑神经元结构启发而设计的数学模型，深度学习就是基于多层神经网络的机器学习方法。",
            "熵是热力学中描述系统无序程度的物理量。根据热力学第二定律，孤立系统的熵永不减少。",
            "板块构造理论认为地球表面由多个板块组成，这些板块在地质历史上不断运动，形成了大陆、海洋、山脉等地理特征。",
            "CRISPR-Cas9 是一种革命性的基因编辑技术，可以精确地修改生物体的DNA序列，在医学和生物学中有广泛应用。",
            "暗物质是一种理论上存在的物质，不发光也不反射光，只能通过引力效应被探测到。它可能占宇宙物质总量的85%。",
            "表观遗传学是研究基因表达的可遗传变化的学科，这些变化不涉及DNA序列的改变，而是通过DNA甲基化、组蛋白修饰等方式实现。",
        ] * 100,
    }

    for domain, texts in sample_data.items():
        file_path = Path(config.output_dir) / f"{domain}_sample.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            for text in texts[:config.samples_per_domain]:
                f.write(text + '\n\n')

        if RICH_AVAILABLE:
            console.print(f"[dim]  创建示例数据: {domain} -> {len(texts[:config.samples_per_domain])} 条[/dim]")

    return config.samples_per_domain * len(sample_data)


async def main():
    """主函数"""
    config = CrawlConfig()
    crawler = DataCrawler(config)

    # 检查 MCP 是否可用
    mcp_available = False
    try:
        # 简单测试 MCP 模块是否可导入
        import importlib
        importlib.import_module("mcp_web_search_prime__webSearchPrime")
        mcp_available = True
    except:
        mcp_available = False

    if mcp_available:
        # 使用 MCP 爬虫
        try:
            await crawler.crawl_all()
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[yellow]MCP 爬取失败，使用示例数据: {e}[/yellow]")
            create_sample_data()
    else:
        # MCP 不可用，直接使用示例数据
        if RICH_AVAILABLE:
            console.print("[yellow]MCP 未配置，使用示例数据[/yellow]")
        else:
            print("MCP 未配置，使用示例数据")
        create_sample_data()

    # 显示结果
    if RICH_AVAILABLE:
        console.print(f"\n[dim]数据已保存到: {config.output_dir}/[/dim]")
        import os
        files = os.listdir(config.output_dir)
        total_lines = 0
        for f in files:
            if f.endswith('.txt'):
                file_path = os.path.join(config.output_dir, f)
                with open(file_path, 'r') as file:
                    lines = len(file.readlines())
                    total_lines += lines
        console.print(f"[dim]总样本数: {total_lines}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())

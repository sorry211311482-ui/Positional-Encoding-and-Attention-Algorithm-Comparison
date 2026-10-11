import argparse
import json
import re

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


PRO_ROUTE = Path(__file__).resolve().parent.parent
DATA_DIR = PRO_ROUTE / "data"
TOKENIZER_SAVE_ROUTE = DATA_DIR / "tokenizer.json"
Tokenizer_SAVE_ROUTE = TOKENIZER_SAVE_ROUTE
ARTICLE_DIR = DATA_DIR / "articles"
ENCODED_DIR = DATA_DIR / "encoded"
SPLITS = ("train", "validation", "test")
TRAIN_ARTICLE_ROUTE = ARTICLE_DIR / "train_articles.json"
VALID_ARTICLE_ROUTE = ARTICLE_DIR / "valid_articles.json"
TEST_ARTICLE_ROUTE = ARTICLE_DIR / "test_articles.json"

seq_len = 256


def _cache_path(directory, split, suffix):
    """兼容 validation 和之前使用的 valid。"""
    if split == "valid":
        split = "validation"
    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}. Expected train, validation (valid), test.")
    stem = "valid" if split == "validation" else split
    return directory / f"{stem}_{suffix}.json"


def _save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)


def _load_json(path):
    # 文件缺失时直接报错，不自动下载或重新生成。
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_article_title(text):
    text = text.strip()

    # 第一步：检查整行是否符合一级标题结构
    match = re.fullmatch(r"=\s+([^=]*)\s+=", text)

    if match is None:
        return False

    # 第二步：检查标题内容是否非空
    title = match.group(1).strip()

    return bool(title)


def build_articles(rows):
    articles = []
    current_lines = []

    # 保存当前文章
    def save_article(lines):
        article_text = "\n".join(
            line.rstrip("\r\n") for line in lines
        )

        article = {
            "article_id": len(articles),
            "text": article_text
        }

        articles.append(article)

    # 遍历数据
    for i, row in enumerate(rows):
        text = row['text']

        # 情况一：遇到新文章标题
        if is_article_title(text):

            # 先保存上一篇文章
            if current_lines:
                save_article(current_lines)

            # 开始新文章
            current_lines = [text]

        # 情况二：当前文章已经开始
        elif current_lines:
            current_lines.append(text)

        # 情况三：文章开始前出现空白记录
        elif not text.strip():
            continue

        # 情况四：文章开始前出现非空正文
        else:
            raise ValueError(
                f"第 {i} 条记录出现异常：{repr(text)}"
            )

    # 保存最后一篇文章
    if current_lines:
        save_article(current_lines)

    return articles


#Tokenizer 是整个分词流程的管理器，models.BPE() 是它使用的分词算法。
def train_tokenizer(train_articles):
    # 1. 创建 BPE 分词模型
    tokenizer = Tokenizer(models.BPE())

    # 2. 设置 ByteLevel 预分词器
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    # 3. 设置 ByteLevel 解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 4. 配置 BPE 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=8192,
        min_frequency=2,
        special_tokens=["<pad>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # 5. 只读取训练集文章
    texts = (article["text"] for article in train_articles)

    # 6. 训练分词器
    tokenizer.train_from_iterator(
        texts,
        trainer=trainer,
        length=len(train_articles)
    )

    return tokenizer


def load_tokenizer():
    """直接加载固定 tokenizer，不训练、不覆盖文件。"""
    return Tokenizer.from_file(str(TOKENIZER_SAVE_ROUTE))


def save_tokenizer(tokenizer):
    TOKENIZER_SAVE_ROUTE.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_SAVE_ROUTE))


def save_articles(split, articles):
    """保存一个划分的完整文章，每篇包含 article_id 和 text。"""
    _save_json(articles, _cache_path(ARTICLE_DIR, split, "articles"))


def load_articles(split):
    return _load_json(_cache_path(ARTICLE_DIR, split, "articles"))


def save_encoded_articles(split, encoded_articles):
    """保存完整文章 ID 序列，保留末尾 EOS，不切窗口或补 PAD。"""
    _save_json(encoded_articles, _cache_path(ENCODED_DIR, split, "encoded"))


def load_encoded_articles(split):
    return _load_json(_cache_path(ENCODED_DIR, split, "encoded"))

def encode_articles(articles, tokenizer):
    encoded_articles = []
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer 缺少 <eos>，无法编码文章结束标记。")
    for article in articles:
        encoded_article = tokenizer.encode(article["text"])
        ids = encoded_article.ids
        ids.append(eos_id)
        encoded_articles.append({
            "article_id": article["article_id"],
            "ids": ids,
        })
    return encoded_articles

def prepare_data(*, retrain_tokenizer=False):
    """显式重建三组缓存。默认复用已保存 tokenizer；训练需明确指定。

    本函数会覆盖文章和编码缓存。日常开发使用 load_prepared_data。
    """
    tokenizer = None if retrain_tokenizer else load_tokenizer()
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    articles = {split: build_articles(ds[split]) for split in SPLITS}
    if retrain_tokenizer:
        tokenizer = train_tokenizer(articles["train"])
        save_tokenizer(tokenizer)

    for split in SPLITS:
        encoded = encode_articles(articles[split], tokenizer)
        save_articles(split, articles[split])
        save_encoded_articles(split, encoded)
        print(f"保存 {split}: {len(encoded)} 篇文章")


def load_prepared_data():
    """日常入口：返回 (tokenizer, 三个划分的完整编码字典)。"""
    tokenizer = load_tokenizer()
    encoded = {split: load_encoded_articles(split) for split in SPLITS}
    return tokenizer, encoded

def make_article_samples(article, seq_len, pad_id):
    if seq_len <= 0:
        raise ValueError("seq_len 必须大于 0")

    article_id = article["article_id"]
    ids = article["ids"]

    samples = []

    for start_offset in range(0, len(ids) - 1, seq_len):

        # 1. 取最多 seq_len + 1 个 token
        chunk = ids[start_offset:start_offset + seq_len + 1]

        # 2. 构造错位输入和标签
        input_ids = chunk[:-1]
        labels = chunk[1:]

        # 3. 记录有效长度
        valid_length = len(labels)

        # 4. 计算 padding 长度
        pad_length = seq_len - valid_length

        # 5. 分别填充
        input_ids = input_ids + [pad_id] * pad_length
        labels = labels + [-100] * pad_length

        # 6. 构造 padding mask
        attention_mask = (
            [1] * valid_length
            + [0] * pad_length
        )

        # 7. 保存样本
        sample = {
            "article_id": article_id,
            "start_offset": start_offset,
            "valid_length": valid_length,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

        samples.append(sample)

    return samples

def make_samples(articles, seq_len, pad_id):
    all_samples = []
    for article in articles:
        all_samples.extend(make_article_samples(article, seq_len, pad_id))
    return all_samples

class LanguageModelDataset(Dataset):
    def __init__(self,samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        res = self.samples[index]
        input_ids = torch.tensor(res["input_ids"],dtype=torch.long)
        labels = torch.tensor(res['labels'],dtype=torch.long)
        attention_mask = torch.tensor(res['attention_mask'],dtype=torch.bool)
        return {
            'input_ids':input_ids,
            'labels':labels,
            'attention_mask':attention_mask
        }


def main():
    parser = argparse.ArgumentParser(description="准备或直接加载 WikiText-2 数据")
    parser.add_argument("command", nargs="?", choices=("load", "prepare"), default="load")
    parser.add_argument(
        "--train-tokenizer", action="store_true",
        help="仅用于 prepare：重新训练 tokenizer，并重建全部文章编码",
    )
    args = parser.parse_args()
    if args.train_tokenizer and args.command != "prepare":
        parser.error("--train-tokenizer 只能和 prepare 一起使用")
    if args.command == "prepare":
        prepare_data(retrain_tokenizer=args.train_tokenizer)
    else:
        tokenizer, encoded = load_prepared_data()
        print(f"加载 tokenizer，词表大小：{tokenizer.get_vocab_size()}")
        for split, articles in encoded.items():
            print(f"加载 {split}: {len(articles)} 篇文章")
        dataset = make_samples(encoded["train"], seq_len, tokenizer.token_to_id("<pad>"))
        dataset = LanguageModelDataset(dataset)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        print(batch['input_ids'].shape)
        print(len(dataset))

if __name__ == "__main__":
    main()


import math
import torch
from torch.onnx.ops import attention
from torch.utils.data import DataLoader
from tqdm.asyncio import tqdm
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from position_lab.data import (
    load_prepared_data,
    make_samples,
    LanguageModelDataset,
)
from position_lab.model import languageModel




def evaluate(model,loader,device):
    state = model.training
    model.eval()
    eval_criterion = torch.nn.CrossEntropyLoss(
        reduction="sum", ignore_index=-100
    )
    with torch.no_grad():
        total_loss = 0.0
        total_tokens = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids)
            loss = eval_criterion(output.view(-1, output.size(-1)), labels.view(-1))
            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    model.train(state)
    return avg_loss, perplexity

def train_one_epoch(model, loader, optimizer, device):
    total_loss = 0
    total_tokens = 0
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for batch in tqdm(loader,desc="Training"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
    avg_loss = total_loss / total_tokens
    return avg_loss

def main():
    seed = 42
    seq_len = 64
    batch_size = 128
    epochs = 5
    dropout = 0.0
    learning_rate = 3e-4
    weight_decay = 0.0
    train_article_limit = None
    position_type = "rope" # rope or sinusoidal
    attention_type = 'linear' # sdpa or linear
    torch.manual_seed(seed)
    tokenizer, encoded = load_prepared_data()
    samples = make_samples(
        encoded["train"][:train_article_limit],
        seq_len=seq_len,
        pad_id=tokenizer.token_to_id("<pad>"),
    )
    train_articles = encoded["train"][:train_article_limit]
    dataset = LanguageModelDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = languageModel(
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layer=4,
        max_len=1024,
        num_embeddings=tokenizer.get_vocab_size(),
        dropout=dropout,
        position_type=position_type,
        attention_type=attention_type
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)
    valid_samples = make_samples(
        encoded["validation"],
        seq_len=seq_len,
        pad_id=tokenizer.token_to_id("<pad>"),
    )
    valid_dataset = LanguageModelDataset(valid_samples)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print("位置编码：", model.position_type)
    print("各层注意力：", [
        block.attention.position_type
        for block in model.blocks
    ])
    print([
        (block.attention.position_type, block.attention.attention_type)
        for block in model.blocks
    ])
    print(
        f'configuration:'
        f'文章数:{len(train_articles)}'
        f'\n 训练样本:{train_article_limit if train_article_limit else len(samples)}  \n '
        f'验证样本数:{len(valid_samples)} '
        f'\n 设备:{device}')
    log_root = Path(__file__).resolve().parents[1] / "runs"

    run_name = (
        f"{position_type}_{attention_type}"
        f"_seed{seed}_L{seq_len}_B{batch_size}"
        f"_{datetime.now():%Y%m%d_%H%M%S}"
    )

    writer = SummaryWriter(log_dir=str(log_root / run_name))
    writer.add_text(
        "Config",
        f"position={position_type}, attention={attention_type}, "
        f"seed={seed}, seq_len={seq_len}, batch_size={batch_size}, "
        f"lr={learning_rate}, dropout={dropout}",
        0,
    )
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loader, optim, device)
        valid_loss, valid_ppl = evaluate(model, valid_loader, device)
        step = epoch + 1
        writer.add_scalar("Loss/train", train_loss, step)
        writer.add_scalar("Loss/validation", valid_loss, step)
        writer.flush()
        print(
            f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Perplexity: {valid_ppl:.4f}')

    writer.close()
if __name__ == "__main__":
    main()
# coding: utf-8
# @Time : 2023/3/12 17:48
# @Author : wowbat
# @File : fune-tuning.py
# @Describe: 


import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer, TextGenerationPipeline
from torch.nn import CrossEntropyLoss

# 构造数据集
class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def setup_args():
    """
    初始化参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="model/Wenzhong2.0", type=str, help='模型路径')
    parser.add_argument('--vocab_path', default="model/Wenzhong2.0/vocab.txt", type=str, help='模型词库路径')
    parser.add_argument('--save_model_path', default="save_model", type=str, help='微调后保存模型路径')
    parser.add_argument('--final_model_path', default="final_model", type=str, help='')
    parser.add_argument('--train_raw_path', default='data/白皮书_train.txt', type=str, help='训练文本路径')
    parser.add_argument('--eval_raw_path', default='data/白皮书_test.txt', type=str, help='验证文本路径')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='批大小')
    parser.add_argument('--epochs', default=2, type=int, required=False, help='训练轮次')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='热身步数')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='打印log日志的步数')
    return parser.parse_args()


def load_model(model_path, vocab_path):
    """
    加载模型
    :param model_path: 模型路径
    :param vocab_path: 词表路径
    :return: 返回模型和分词器
    """
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = BertTokenizer(vocab_file=vocab_path)
    return model, tokenizer


def calculate_loss_and_accuracy(outputs, labels, device):
    # 计算损失和准确度

    # 获取输出的最后logits层数据,
    # 如果最后补全的是335位,那size就是[batchSize,335, 21128]
    logits = outputs.logits

    # 对数据进行偏移预测, 提供的数据是从[0-334], 预测的标签位置是[1-335], 即用[0,1,2,...,n-1]的词预测第[n]个词
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()  # size为[batchSize, 334, 21128]
    shift_labels = labels[..., 1:].contiguous().to(device)  # size为[batch, 334]

    # Flatten the tokens
    # 定义交叉熵损失函数
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    # 把token展平, shift_logits由原来的[2,334,21128]展平成[668,21128], 而shift_labels由原来的[2,334]变成[668]
    # 展平之后计算交叉熵损失
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # 根据最后一个维度21128找到最大值的value,index, 抛弃value, 只留index作为预测值
    # preds的size为[2, 334]
    _, preds = shift_logits.max(dim=-1)
    # 由于有些位数会用pading填充, 所以需要找到不能忽略的值, 得到的值跟shift_labels维度一样,都是[2,334],
    # 但是这是一个布尔矩阵, 都是true/false, true表示不为pad的位置, false表示为pading的位置
    not_ignore = shift_labels.ne(tokenizer.pad_token_id)
    # not_ignore.long()将布尔矩阵转化为整数矩阵,其中true为1, false为0
    # sum()则是把矩阵里面的值全部加起来, 由于true为1, false为0, 那么这个值就是true的值的总个数
    # item()就是取得标量值, 这个值作为总数进行后续判断用
    num_targets = not_ignore.long().sum().item()
    # 由于部分shift_labels用padding进行了填充, 所以需要计算标签与预测一样的位置, 同时该位置还是不能忽略的
    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()  # 获得正确的数量
    # 准确率=正确的数量/目标总数
    accuracy = correct / num_targets
    loss = loss / num_targets  # 获得平均交叉熵

    return loss, accuracy


# dataloader每次迭代的时候会运行该回调函数
# 决定了dataloader怎么从数据集中整理数据, 这里主要按max数据长度使用padding进行对齐
def collate_fn(batch):
    # batch就是每一批的数据, 已经encode好了, 例如batch_num=2,那么这就是一个size为2的list,其中每个值都是一个encode之后的input_id的list
    input_ids = []  # 用来放最后的inputs, 保证对齐, 空数据用pading填充
    input_lens_list = [len(w) for w in batch]  # 得到每一批数据的长度
    max_input_len = max(input_lens_list)  # 得到每一批数据长度中的最大值
    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        # 每一批中数据中每组用pading进行填充
        input_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def data_loader(args, train_data_path, tokenizer, shuffle):
    data_list = []
    # 读取训练数据
    with open(train_data_path, 'rb') as f:
        # 一次性全部读取之后使用utf-8解码
        data = f.read().decode("utf-8")
        # 按换行进行分隔
        train_data = data.split("\r\n")
        print("数据总行数:{}".format(len(train_data)))
        for text in tqdm(train_data):
            # text_split = text.split("\t")
            # if len(text_split) != 3:
            #     continue
            # product_word, title, wenan = text_split
            # title_ids = tokenizer.encode(title)
            # wenan_ids = tokenizer.encode(wenan)
            # inputs_ids = title_ids + wenan_ids[1:]
            inputs_ids = tokenizer.encode(text)
            data_list.append(inputs_ids)
    dataset = MyDataset(data_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn,
                            num_workers=0)

    return dataloader


def train(args, model, dataloader):
    # 参数微调
    num_training_steps = args.epochs * len(dataloader)  # 获取训练总步数
    optimizer = AdamW(model.parameters(), lr=args.lr)  # 优化器
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )  # 学习率调度器
    # 训练设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 将模型复制到设备上,开启训练模式
    model.to(device)
    model.train()

    # 批处理步数
    batch_steps = 0
    for epoch in range(args.epochs):
        # 获取每一批数据, 数据已经对齐了, 数据长度为每批最大的
        for batch in dataloader:
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            # loss = outputs.loss
            loss, acc = calculate_loss_and_accuracy(outputs, batch.to(device), device)
            loss.backward()  # 反向传播损失
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪, 防止梯度爆炸
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if batch_steps % args.log_step == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}, accuracy {}".format(
                    epoch+1, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc))

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.save_model_path)
    # torch.save(model, os.path.join(args.final_model_path, 'gpt2_WenAn.pth'))


def evaluate(dataloader, args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, _ = load_model(args.save_model_path, args.vocab_path)
    model.to(device)
    model.eval()
    loss_list, acc_list = [], [],
    batch_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            loss, acc = calculate_loss_and_accuracy(outputs, batch.to(device), device)
            loss_list.append(float(loss))
            acc_list.append(float(acc))
            print("eval batch {}/{}, loss {}, accuracy {}".format(
                batch_steps,
                len(dataloader),
                loss, acc))
    print("loss: {},".format(np.mean(loss_list)))


def predict(args, text="国家建设取得了"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, tokenizer = load_model(args.save_model_path, args.vocab_path)
    text_generator = TextGenerationPipeline(model, tokenizer)
    print(text_generator(text, max_length=100, do_sample=True))
    # model.to(device)
    # model.eval()
    # time1 = time.time()
    # max_length = 30
    # input_ids = tokenizer.encode(text)
    # wenan = ""
    # for i in range(max_length):
    #     input_tensor = torch.tensor([input_ids])
    #     inputs = {"input_ids": input_tensor.to(device)}
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     last_token_id = int(np.argmax(logits[0][-1].detach().to('cpu').numpy()))
    #     if last_token_id == tokenizer.sep_token_id:
    #         break
    #     last_token = tokenizer.convert_ids_to_tokens(last_token_id)
    #     input_ids.append(last_token_id)
    #     wenan += last_token
    # print("time cost: {}".format(time.time()-time1))
    # print(text)
    # print(wenan)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()  # 初始化参数
    # model, tokenizer = load_model(args.model_path, args.vocab_path)  # 加载现有模型和分词器
    # train_dataloader = data_loader(args, args.train_raw_path, tokenizer=tokenizer, shuffle=True)
    # eval_dataloader = data_loader(args, args.eval_raw_path, tokenizer=tokenizer, shuffle=False)
    # train(args, model, train_dataloader)
    # evaluate(eval_dataloader, args=args)

    # 预测
    predict(args, "为介绍中国应对气候变化进展，分享中国应对气候变化")



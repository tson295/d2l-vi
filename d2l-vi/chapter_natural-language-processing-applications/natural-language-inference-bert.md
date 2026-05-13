# Suy luận ngôn ngữ tự nhiên: Fine-tune BERT
<a id="sec_natural-language-inference-bert"></a>

Trong các phần trước của chương này,
chúng ta đã thiết kế một kiến trúc dựa trên attention
(trong [sec_natural-language-inference-attention](#sec_natural-language-inference-attention))
cho tác vụ suy luận ngôn ngữ tự nhiên
trên bộ dữ liệu SNLI (như mô tả trong [sec_natural-language-inference-and-dataset](#sec_natural-language-inference-and-dataset)).
Bây giờ chúng ta quay lại tác vụ này bằng cách fine-tune BERT.
Như đã thảo luận trong [sec_finetuning-bert](#sec_finetuning-bert),
suy luận ngôn ngữ tự nhiên là một bài toán phân loại cặp văn bản cấp chuỗi,
và fine-tune BERT chỉ cần một kiến trúc bổ sung dựa trên MLP,
như minh họa trong [fig_nlp-map-nli-bert](#fig_nlp-map-nli-bert).

![Phần này đưa BERT đã tiền huấn luyện vào một kiến trúc dựa trên MLP cho suy luận ngôn ngữ tự nhiên.](../img/nlp-map-nli-bert.svg)
<a id="fig_nlp-map-nli-bert"></a>

Trong phần này,
chúng ta sẽ tải xuống một phiên bản BERT nhỏ đã tiền huấn luyện,
rồi fine-tune nó
cho suy luận ngôn ngữ tự nhiên trên bộ dữ liệu SNLI.

```python
#@tab mxnet
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```python
#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
```

## [**Nạp BERT đã tiền huấn luyện**]

Chúng ta đã giải thích cách tiền huấn luyện BERT trên bộ dữ liệu WikiText-2 trong
[sec_bert-dataset](#sec_bert-dataset) và [sec_bert-pretraining](#sec_bert-pretraining)
(lưu ý rằng mô hình BERT gốc được tiền huấn luyện trên các kho ngữ liệu lớn hơn nhiều).
Như đã thảo luận trong [sec_bert-pretraining](#sec_bert-pretraining),
mô hình BERT gốc có hàng trăm triệu tham số.
Trong phần sau,
chúng tôi cung cấp hai phiên bản BERT đã tiền huấn luyện:
"bert.base" có kích thước xấp xỉ mô hình BERT base gốc và cần nhiều tài nguyên tính toán để fine-tune,
trong khi "bert.small" là một phiên bản nhỏ để thuận tiện cho minh họa.

```python
#@tab mxnet
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```

```python
#@tab pytorch
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```

Cả hai mô hình BERT đã tiền huấn luyện đều chứa một file "vocab.json" định nghĩa tập vocabulary
và một file "pretrained.params" chứa các tham số đã tiền huấn luyện.
Chúng ta triển khai hàm `load_pretrained_model` sau để [**nạp các tham số BERT đã tiền huấn luyện**].

```python
#@tab mxnet
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, 
                         num_blks, dropout, max_len)
    # Load pretrained BERT parameters
    bert.load_parameters(os.path.join(data_dir, 'pretrained.params'),
                         ctx=devices)
    return bert, vocab
```

```python
#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(
        len(vocab), num_hiddens, ffn_num_hiddens=ffn_num_hiddens, num_heads=4,
        num_blks=2, dropout=0.2, max_len=max_len)
    # Load pretrained BERT parameters
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab
```

Để thuận tiện minh họa trên hầu hết máy tính,
trong phần này chúng ta sẽ nạp và fine-tune phiên bản nhỏ ("bert.small") của BERT đã tiền huấn luyện.
Trong bài tập, chúng ta sẽ chỉ ra cách fine-tune "bert.base" lớn hơn nhiều để cải thiện đáng kể độ chính xác kiểm tra.

```python
#@tab all
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_blks=2, dropout=0.1, max_len=512, devices=devices)
```

## [**Bộ dữ liệu để fine-tune BERT**]

Với tác vụ downstream suy luận ngôn ngữ tự nhiên trên bộ dữ liệu SNLI,
chúng ta định nghĩa một lớp dataset tùy biến `SNLIBERTDataset`.
Trong mỗi ví dụ,
tiền đề và giả thuyết tạo thành một cặp chuỗi văn bản
và được đóng gói vào một chuỗi đầu vào BERT như minh họa trong [fig_bert-two-seqs](#fig_bert-two-seqs).
Nhắc lại [subsec_bert_input_rep](#subsec_bert_input_rep) rằng các segment ID
được dùng để phân biệt tiền đề và giả thuyết trong một chuỗi đầu vào BERT.
Với độ dài tối đa được định trước của một chuỗi đầu vào BERT (`max_len`),
token cuối cùng của văn bản dài hơn trong cặp văn bản đầu vào sẽ tiếp tục bị loại bỏ cho đến khi
đáp ứng `max_len`.
Để tăng tốc việc sinh bộ dữ liệu SNLI
cho fine-tune BERT,
chúng ta dùng 4 tiến trình worker để sinh song song các ví dụ huấn luyện hoặc kiểm tra.

```python
#@tab mxnet
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (np.array(all_token_ids, dtype='int32'),
                np.array(all_segments, dtype='int32'), 
                np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```python
#@tab pytorch
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), 
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

Sau khi tải xuống bộ dữ liệu SNLI,
chúng ta [**sinh các ví dụ huấn luyện và kiểm tra**]
bằng cách khởi tạo lớp `SNLIBERTDataset`.
Các ví dụ như vậy sẽ được đọc theo minibatch trong quá trình huấn luyện và kiểm tra
suy luận ngôn ngữ tự nhiên.

```python
#@tab mxnet
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```python
#@tab pytorch
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

## Fine-tune BERT

Như [fig_bert-two-seqs](#fig_bert-two-seqs) cho thấy,
fine-tune BERT cho suy luận ngôn ngữ tự nhiên
chỉ cần một MLP bổ sung gồm hai tầng kết nối đầy đủ
(xem `self.hidden` và `self.output` trong lớp `BERTClassifier` sau).
[**MLP này biến đổi
biểu diễn BERT của token đặc biệt “&lt;cls&gt;”**],
vốn mã hóa thông tin của cả tiền đề và giả thuyết,
(**thành ba đầu ra của suy luận ngôn ngữ tự nhiên**):
kéo theo, mâu thuẫn và trung lập.

```python
#@tab mxnet
class BERTClassifier(nn.Block):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```python
#@tab pytorch
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.LazyLinear(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

Sau đây,
mô hình BERT đã tiền huấn luyện `bert` được đưa vào instance `BERTClassifier` `net` cho
ứng dụng downstream.
Trong các triển khai fine-tuning BERT phổ biến,
chỉ tham số của tầng đầu ra của MLP bổ sung (`net.output`) sẽ được học từ đầu.
Tất cả tham số của BERT encoder đã tiền huấn luyện (`net.encoder`) và tầng ẩn của MLP bổ sung (`net.hidden`) sẽ được fine-tune.

```python
#@tab mxnet
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```

```python
#@tab pytorch
net = BERTClassifier(bert)
```

Nhắc lại rằng
trong [sec_bert](#sec_bert)
cả lớp `MaskLM` và lớp `NextSentencePred`
đều có tham số trong các MLP được chúng sử dụng.
Các tham số này là một phần của các tham số trong mô hình BERT đã tiền huấn luyện
`bert`, và do đó là một phần của các tham số trong `net`.
Tuy nhiên, các tham số như vậy chỉ dùng để tính
loss masked language modeling
và loss next sentence prediction
trong quá trình tiền huấn luyện.
Hai hàm loss này không liên quan đến fine-tune các ứng dụng downstream,
do đó các tham số của các MLP được dùng trong
`MaskLM` và `NextSentencePred` không được cập nhật (staled) khi BERT được fine-tune.

Để cho phép các tham số có gradient stale,
cờ `ignore_stale_grad=True` được đặt trong hàm `step` của `d2l.train_batch_ch13`.
Chúng ta dùng hàm này để huấn luyện và đánh giá mô hình `net` bằng tập huấn luyện
(`train_iter`) và tập kiểm tra (`test_iter`) của SNLI.
Do tài nguyên tính toán hạn chế, [**độ chính xác huấn luyện**] và kiểm tra
có thể được cải thiện thêm: chúng ta để phần thảo luận này trong bài tập.

```python
#@tab mxnet
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               d2l.split_batch_multi_inputs)
```

```python
#@tab pytorch
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
net(next(iter(train_iter))[0])
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Tóm tắt

* Chúng ta có thể fine-tune mô hình BERT đã tiền huấn luyện cho các ứng dụng downstream, chẳng hạn như suy luận ngôn ngữ tự nhiên trên bộ dữ liệu SNLI.
* Trong quá trình fine-tune, mô hình BERT trở thành một phần của mô hình cho ứng dụng downstream. Các tham số chỉ liên quan đến loss tiền huấn luyện sẽ không được cập nhật trong quá trình fine-tune. 


## Bài tập

1. Fine-tune một mô hình BERT đã tiền huấn luyện lớn hơn nhiều, có kích thước xấp xỉ mô hình BERT base gốc, nếu tài nguyên tính toán của bạn cho phép. Đặt các đối số trong hàm `load_pretrained_model` như sau: thay 'bert.small' bằng 'bert.base', tăng các giá trị `num_hiddens=256`, `ffn_num_hiddens=512`, `num_heads=4` và `num_blks=2` lần lượt lên 768, 3072, 12 và 12. Bằng cách tăng số epoch fine-tuning (và có thể điều chỉnh các siêu tham số khác), bạn có thể đạt độ chính xác kiểm tra cao hơn 0.86 không?
1. Làm thế nào để cắt ngắn một cặp chuỗi theo tỉ lệ độ dài của chúng? So sánh phương pháp cắt ngắn cặp này với phương pháp được dùng trong lớp `SNLIBERTDataset`. Ưu và nhược điểm của chúng là gì?


[Thảo luận](https://discuss.d2l.ai/t/1526)

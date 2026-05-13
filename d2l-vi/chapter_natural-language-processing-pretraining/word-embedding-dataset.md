# Tập Dữ Liệu Cho Tiền Huấn Luyện Word Embedding
<a id="sec_word2vec_data"></a>

Bây giờ khi đã biết các chi tiết kỹ thuật của
các mô hình word2vec và các phương pháp huấn luyện xấp xỉ,
hãy đi qua phần hiện thực của chúng.
Cụ thể,
chúng ta sẽ lấy mô hình skip-gram trong [sec_word2vec](#sec_word2vec)
và negative sampling trong [sec_approx_train](#sec_approx_train)
làm ví dụ.
Trong mục này,
chúng ta bắt đầu với tập dữ liệu
để tiền huấn luyện mô hình word embedding:
định dạng dữ liệu ban đầu
sẽ được biến đổi
thành các minibatch
có thể được lặp qua trong quá trình huấn luyện.

```python
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```python
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import os
import random
```

## Đọc Tập Dữ Liệu

Tập dữ liệu mà chúng ta dùng ở đây
là [Penn Tree Bank (PTB)]( https://catalog.ldc.upenn.edu/LDC99T42).
Kho ngữ liệu này được lấy mẫu
từ các bài viết của Wall Street Journal,
được chia thành tập huấn luyện, kiểm định, và kiểm tra.
Trong định dạng ban đầu,
mỗi dòng của tệp văn bản
biểu diễn một câu gồm các từ được phân tách bằng dấu cách.
Ở đây chúng ta xem mỗi từ là một token.

```python
#@tab all
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

Sau khi đọc tập huấn luyện,
chúng ta xây dựng một từ vựng cho kho ngữ liệu,
trong đó bất kỳ từ nào xuất hiện
ít hơn 10 lần sẽ được thay bằng
token "&lt;unk&gt;".
Lưu ý rằng tập dữ liệu gốc
cũng chứa các token "&lt;unk&gt;" biểu diễn các từ hiếm (không biết).

```python
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## Lấy Mẫu Con

Dữ liệu văn bản
thường có các từ tần suất cao
như "the", "a", và "in":
chúng thậm chí có thể xuất hiện hàng tỷ lần trong
các kho ngữ liệu rất lớn.
Tuy nhiên,
các từ này thường đồng xuất hiện
với nhiều từ khác nhau trong
các cửa sổ ngữ cảnh, cung cấp ít tín hiệu hữu ích.
Chẳng hạn,
xét từ "chip" trong một cửa sổ ngữ cảnh:
theo trực giác,
việc nó đồng xuất hiện với một từ tần suất thấp "intel"
hữu ích hơn trong huấn luyện
so với
việc đồng xuất hiện với một từ tần suất cao "a".
Hơn nữa, huấn luyện với lượng lớn từ (tần suất cao)
thì chậm.
Do đó, khi huấn luyện các mô hình word embedding,
các từ tần suất cao có thể được *lấy mẫu con* [Mikolov.Sutskever.Chen.ea.2013].
Cụ thể,
mỗi từ đã được đánh chỉ số $w_i$
trong tập dữ liệu sẽ bị loại bỏ với xác suất


$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

trong đó $f(w_i)$ là tỉ lệ giữa
số lượng từ $w_i$
và tổng số từ trong tập dữ liệu,
và hằng số $t$ là một siêu tham số
($10^{-4}$ trong thí nghiệm).
Chúng ta có thể thấy rằng chỉ khi
tần suất tương đối
$f(w_i) > t$ thì từ (tần suất cao) $w_i$ mới có thể bị loại bỏ,
và tần suất tương đối của từ càng cao,
xác suất bị loại bỏ càng lớn.

```python
#@tab all
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

Đoạn code sau
vẽ histogram của
số token trên mỗi câu
trước và sau khi lấy mẫu con.
Như kỳ vọng,
lấy mẫu con rút ngắn câu đáng kể
bằng cách bỏ các từ tần suất cao,
điều này sẽ tăng tốc huấn luyện.

```python
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

Với từng token riêng lẻ, tỉ lệ lấy mẫu của từ tần suất cao "the" nhỏ hơn 1/20.

```python
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

Ngược lại,
các từ tần suất thấp "join" được giữ lại hoàn toàn.

```python
#@tab all
compare_counts('join')
```

Sau khi lấy mẫu con, chúng ta ánh xạ token sang chỉ số của chúng cho kho ngữ liệu.

```python
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## Trích Xuất Từ Trung Tâm và Từ Ngữ Cảnh


Hàm `get_centers_and_contexts` sau
trích xuất tất cả
từ trung tâm và các từ ngữ cảnh của chúng
từ `corpus`.
Nó lấy mẫu đều một số nguyên giữa 1 và `max_window_size`
một cách ngẫu nhiên làm kích thước cửa sổ ngữ cảnh.
Với bất kỳ từ trung tâm nào,
những từ
có khoảng cách tới nó
không vượt quá
kích thước cửa sổ ngữ cảnh đã lấy mẫu
là các từ ngữ cảnh của nó.

```python
#@tab all
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

Tiếp theo, chúng ta tạo một tập dữ liệu nhân tạo chứa hai câu có lần lượt 7 và 3 từ.
Đặt kích thước cửa sổ ngữ cảnh tối đa là 2
và in tất cả từ trung tâm cùng từ ngữ cảnh của chúng.

```python
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

Khi huấn luyện trên tập dữ liệu PTB,
chúng ta đặt kích thước cửa sổ ngữ cảnh tối đa là 5.
Đoạn sau trích xuất tất cả từ trung tâm và từ ngữ cảnh của chúng trong tập dữ liệu.

```python
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## Negative Sampling

Chúng ta dùng negative sampling cho huấn luyện xấp xỉ.
Để lấy mẫu từ nhiễu theo
một phân bố định trước,
chúng ta định nghĩa lớp `RandomGenerator` sau,
trong đó phân bố lấy mẫu (có thể chưa chuẩn hóa) được truyền vào
qua đối số `sampling_weights`.

```python
#@tab all
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

Ví dụ,
chúng ta có thể lấy 10 biến ngẫu nhiên $X$
trong các chỉ số 1, 2, và 3
với các xác suất lấy mẫu $P(X=1)=2/9, P(X=2)=3/9$, và $P(X=3)=4/9$ như sau.

```python
#@tab mxnet
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

Với một cặp từ trung tâm và từ ngữ cảnh,
chúng ta lấy mẫu ngẫu nhiên `K` (5 trong thí nghiệm) từ nhiễu. Theo các gợi ý trong bài báo word2vec,
xác suất lấy mẫu $P(w)$ của
một từ nhiễu $w$
được
đặt bằng tần suất tương đối của nó
trong từ điển
nâng lên
lũy thừa 0.75 [Mikolov.Sutskever.Chen.ea.2013].

```python
#@tab all
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## Tải Ví Dụ Huấn Luyện Theo Minibatch
<a id="subsec_word2vec-minibatch-loading"></a>

Sau khi
tất cả từ trung tâm
cùng với
từ ngữ cảnh và từ nhiễu đã lấy mẫu của chúng được trích xuất,
chúng sẽ được biến đổi thành
các minibatch ví dụ
có thể được tải lặp lại
trong quá trình huấn luyện.


Trong một minibatch,
ví dụ thứ $i$ bao gồm một từ trung tâm
và $n_i$ từ ngữ cảnh cùng $m_i$ từ nhiễu của nó.
Do kích thước cửa sổ ngữ cảnh thay đổi,
$n_i+m_i$ khác nhau với các $i$ khác nhau.
Do đó,
với mỗi ví dụ,
chúng ta nối các từ ngữ cảnh và từ nhiễu của nó trong
biến `contexts_negatives`,
và đệm các số không cho đến khi độ dài phép nối
đạt $\max_i n_i+m_i$ (`max_len`).
Để loại trừ các phần đệm
trong phép tính mất mát,
chúng ta định nghĩa biến mask `masks`.
Có một tương ứng một-một
giữa các phần tử trong `masks` và các phần tử trong `contexts_negatives`,
trong đó các số không (ngược lại là số một) trong `masks` tương ứng với phần đệm trong `contexts_negatives`.


Để phân biệt giữa ví dụ dương và ví dụ âm,
chúng ta tách các từ ngữ cảnh khỏi các từ nhiễu trong `contexts_negatives` thông qua một biến `labels`.
Tương tự `masks`,
cũng có một tương ứng một-một
giữa các phần tử trong `labels` và các phần tử trong `contexts_negatives`,
trong đó các số một (ngược lại là số không) trong `labels` tương ứng với các từ ngữ cảnh (ví dụ dương) trong `contexts_negatives`.


Ý tưởng trên được hiện thực trong hàm `batchify` sau.
Đầu vào `data` của nó là một danh sách có độ dài
bằng kích thước batch,
trong đó mỗi phần tử là một ví dụ
gồm
từ trung tâm `center`, các từ ngữ cảnh `context`, và các từ nhiễu `negative` của nó.
Hàm này trả về
một minibatch có thể được tải để tính toán
trong quá trình huấn luyện,
chẳng hạn bao gồm biến mask.

```python
#@tab all
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

Hãy kiểm tra hàm này bằng một minibatch gồm hai ví dụ.

```python
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## Gộp Tất Cả Lại

Cuối cùng, chúng ta định nghĩa hàm `load_data_ptb` để đọc tập dữ liệu PTB và trả về iterator dữ liệu cùng từ vựng.

```python
#@tab mxnet
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```python
#@tab pytorch
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

Hãy in minibatch đầu tiên của iterator dữ liệu.

```python
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## Tóm Tắt

* Các từ tần suất cao có thể không quá hữu ích trong huấn luyện. Chúng ta có thể lấy mẫu con chúng để tăng tốc huấn luyện.
* Để tính toán hiệu quả, chúng ta tải ví dụ theo minibatch. Ta có thể định nghĩa các biến khác để phân biệt phần đệm với phần không đệm, và ví dụ dương với ví dụ âm.


## Bài Tập

1. Thời gian chạy của code trong mục này thay đổi như thế nào nếu không dùng lấy mẫu con?
1. Lớp `RandomGenerator` lưu đệm `k` kết quả lấy mẫu ngẫu nhiên. Đặt `k` thành các giá trị khác và xem nó ảnh hưởng thế nào đến tốc độ tải dữ liệu.
1. Những siêu tham số nào khác trong code của mục này có thể ảnh hưởng đến tốc độ tải dữ liệu?


[Thảo luận](https://discuss.d2l.ai/t/1330)

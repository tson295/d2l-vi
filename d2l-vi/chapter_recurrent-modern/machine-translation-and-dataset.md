# Dịch Máy và Tập Dữ Liệu
<a id="sec_machine_translation"></a>

Trong số những đột phá lớn đã thúc đẩy
sự quan tâm rộng rãi đến các RNN hiện đại
là một bước tiến lớn trong lĩnh vực ứng dụng của
*dịch máy* thống kê.
Ở đây, mô hình được cung cấp một câu trong một ngôn ngữ
và phải dự đoán câu tương ứng trong ngôn ngữ khác.
Lưu ý rằng ở đây các câu có thể có độ dài khác nhau,
và các từ tương ứng trong hai câu
có thể không xuất hiện theo cùng thứ tự,
do sự khác biệt
trong cấu trúc ngữ pháp của hai ngôn ngữ.


Nhiều vấn đề có tính chất ánh xạ
giữa hai chuỗi "không căn chỉnh" như vậy.
Các ví dụ bao gồm ánh xạ
từ các gợi ý hội thoại sang phản hồi
hoặc từ câu hỏi sang câu trả lời.
Nhìn chung, các vấn đề như vậy được gọi là
các vấn đề *chuỗi-sang-chuỗi* (seq2seq)
và chúng là trọng tâm của chúng ta cho
phần còn lại của chương này
và nhiều phần trong [chap_attention-and-transformers](#chap_attention-and-transformers).

Trong phần này, chúng ta giới thiệu bài toán dịch máy
và một tập dữ liệu ví dụ mà chúng ta sẽ sử dụng trong các ví dụ tiếp theo.
Trong nhiều thập kỷ, các công thức thống kê về dịch thuật giữa các ngôn ngữ
đã rất phổ biến [Brown.Cocke.Della-Pietra.ea.1988, Brown.Cocke.Della-Pietra.ea.1990],
ngay cả trước khi các nhà nghiên cứu làm cho các phương pháp mạng nơ-ron hoạt động
(các phương pháp thường được gộp chung dưới thuật ngữ *dịch máy bằng mạng nơ-ron*).


Đầu tiên chúng ta sẽ cần một số mã mới để xử lý dữ liệu của mình.
Không giống như mô hình hóa ngôn ngữ mà chúng ta đã thấy trong [sec_language-model](#sec_language-model),
ở đây mỗi ví dụ gồm hai chuỗi văn bản riêng biệt,
một trong ngôn ngữ nguồn và một (bản dịch) trong ngôn ngữ đích.
Các đoạn mã sau sẽ hiển thị cách
tải dữ liệu đã được tiền xử lý vào các minibatch để huấn luyện.


```python
from d2l import torch as d2l
import torch
import os
```


## [**Tải Xuống và Tiền Xử Lý Tập Dữ Liệu**]

Để bắt đầu, chúng ta tải xuống tập dữ liệu Anh--Pháp
gồm [các cặp câu song ngữ từ Dự án Tatoeba](http://www.manythings.org/anki/).
Mỗi dòng trong tập dữ liệu là một cặp được phân tách bằng tab
gồm một chuỗi văn bản tiếng Anh (nguồn)
và chuỗi văn bản tiếng Pháp đã dịch (đích).
Lưu ý rằng mỗi chuỗi văn bản
có thể chỉ là một câu,
hoặc một đoạn gồm nhiều câu.

```python
class MTFraEng(d2l.DataModule):  
    """The English-French dataset."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
```

```python
data = MTFraEng() 
raw_text = data._download()
print(raw_text[:75])
```

Sau khi tải xuống tập dữ liệu,
chúng ta [**tiến hành một số bước tiền xử lý**]
cho dữ liệu văn bản thô.
Ví dụ, chúng ta thay thế ký tự không ngắt dòng bằng khoảng trắng,
chuyển đổi chữ hoa thành chữ thường,
và chèn khoảng trắng giữa từ và dấu câu.

```python
@d2l.add_to_class(MTFraEng)  
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace(' ', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)
```

```python
text = data._preprocess(raw_text)
print(text[:80])
```

## [**Phân Token**]

Không giống như phân token theo cấp ký tự
trong [sec_language-model](#sec_language-model),
để dịch máy
chúng ta ưu tiên phân token theo cấp từ ở đây
(các mô hình tiên tiến nhất ngày nay sử dụng
các kỹ thuật phân token phức tạp hơn).
Phương thức `_tokenize` sau
phân token `max_examples` cặp chuỗi văn bản đầu tiên,
trong đó mỗi token là một từ hoặc dấu câu.
Chúng ta thêm token đặc biệt "&lt;eos&gt;"
vào cuối mỗi chuỗi để chỉ ra
phần cuối của chuỗi.
Khi mô hình đang dự đoán
bằng cách tạo ra một chuỗi token sau token,
việc tạo ra token "&lt;eos&gt;"
có thể gợi ý rằng chuỗi đầu ra đã hoàn chỉnh.
Cuối cùng, phương thức dưới đây trả về
hai danh sách các danh sách token: `src` và `tgt`.
Cụ thể, `src[i]` là danh sách token từ
chuỗi văn bản thứ $i$ trong ngôn ngữ nguồn (tiếng Anh ở đây)
và `tgt[i]` là trong ngôn ngữ đích (tiếng Pháp ở đây).

```python
@d2l.add_to_class(MTFraEng)  
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt
```

```python
src, tgt = data._tokenize(text)
src[:6], tgt[:6]
```

Hãy [**vẽ biểu đồ tần số của số token trên mỗi chuỗi văn bản.**]
Trong tập dữ liệu Anh--Pháp đơn giản này,
hầu hết các chuỗi văn bản có ít hơn 20 token.

```python
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
```

```python
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt);
```

## Tải Chuỗi Có Độ Dài Cố Định
<a id="subsec_loading-seq-fixed-len"></a>

Nhớ lại rằng trong mô hình hóa ngôn ngữ
[**mỗi chuỗi ví dụ**],
dù là một đoạn của một câu
hay một khoảng trải qua nhiều câu,
(**có độ dài cố định.**)
Điều này được chỉ định bởi tham số `num_steps`
(số bước thời gian hoặc token) từ [sec_language-model](#sec_language-model).
Trong dịch máy, mỗi ví dụ là
một cặp chuỗi văn bản nguồn và đích,
trong đó hai chuỗi văn bản có thể có độ dài khác nhau.

Để tính toán hiệu quả,
chúng ta vẫn có thể xử lý một minibatch các chuỗi văn bản
cùng một lúc bằng cách *cắt xén* và *đệm*.
Giả sử rằng mỗi chuỗi trong cùng một minibatch
nên có cùng độ dài `num_steps`.
Nếu một chuỗi văn bản có ít hơn `num_steps` token,
chúng ta sẽ tiếp tục thêm token đặc biệt "&lt;pad&gt;"
vào cuối nó cho đến khi độ dài của nó đạt `num_steps`.
Ngược lại, chúng ta sẽ cắt bớt chuỗi văn bản
bằng cách chỉ lấy `num_steps` token đầu tiên của nó
và loại bỏ phần còn lại.
Bằng cách này, mỗi chuỗi văn bản
sẽ có cùng độ dài
để được tải trong các minibatch cùng hình dạng.
Hơn nữa, chúng ta cũng ghi lại độ dài của chuỗi nguồn không tính các token đệm.
Thông tin này sẽ được cần bởi một số mô hình mà chúng ta sẽ đề cập sau.


Vì tập dữ liệu dịch máy
gồm các cặp ngôn ngữ,
chúng ta có thể xây dựng hai từ vựng cho
cả ngôn ngữ nguồn và
ngôn ngữ đích riêng biệt.
Với phân token theo cấp từ,
kích thước từ vựng sẽ lớn hơn đáng kể
so với khi sử dụng phân token theo cấp ký tự.
Để giảm nhẹ điều này,
ở đây chúng ta coi các token không thường xuyên
xuất hiện ít hơn hai lần
là cùng một token không xác định ("&lt;unk&gt;").
Như chúng ta sẽ giải thích sau ([fig_seq2seq](#fig_seq2seq)),
khi huấn luyện với các chuỗi đích,
đầu ra của bộ giải mã (token nhãn)
có thể giống như đầu vào của bộ giải mã (token đích),
dịch chuyển một token;
và token đặc biệt bắt đầu-chuỗi "&lt;bos&gt;"
sẽ được sử dụng làm token đầu vào đầu tiên
để dự đoán chuỗi đích ([fig_seq2seq_predict](#fig_seq2seq_predict)).

```python
@d2l.add_to_class(MTFraEng)  
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())
```

```python
@d2l.add_to_class(MTFraEng)  
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## [**Đọc Tập Dữ Liệu**]

Cuối cùng, chúng ta định nghĩa phương thức `get_dataloader`
để trả về trình lặp dữ liệu.

```python
@d2l.add_to_class(MTFraEng)  
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

Hãy [**đọc minibatch đầu tiên từ tập dữ liệu Anh--Pháp.**]

```python
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('decoder input:', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

Chúng ta hiển thị một cặp chuỗi nguồn và đích
được xử lý bởi phương thức `_build_arrays` ở trên
(ở định dạng chuỗi).

```python
@d2l.add_to_class(MTFraEng)  
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```python
src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## Tóm Tắt

Trong xử lý ngôn ngữ tự nhiên, *dịch máy* đề cập đến nhiệm vụ tự động ánh xạ từ một chuỗi đại diện cho một chuỗi văn bản trong ngôn ngữ *nguồn* sang một chuỗi đại diện cho một bản dịch hợp lý trong ngôn ngữ *đích*. Sử dụng phân token theo cấp từ, kích thước từ vựng sẽ lớn hơn đáng kể so với khi sử dụng phân token theo cấp ký tự, nhưng độ dài chuỗi sẽ ngắn hơn nhiều. Để giảm kích thước từ vựng lớn, chúng ta có thể coi các token không thường xuyên là một số token "không xác định". Chúng ta có thể cắt xén và đệm các chuỗi văn bản sao cho tất cả chúng sẽ có cùng độ dài để được tải trong các minibatch. Các triển khai hiện đại thường nhóm các chuỗi có độ dài tương tự để tránh lãng phí quá nhiều tính toán vào đệm.


## Bài Tập

1. Thử các giá trị khác nhau của tham số `max_examples` trong phương thức `_tokenize`. Điều này ảnh hưởng như thế nào đến kích thước từ vựng của ngôn ngữ nguồn và ngôn ngữ đích?
1. Văn bản trong một số ngôn ngữ như tiếng Trung và tiếng Nhật không có các chỉ thị ranh giới từ (ví dụ: khoảng trắng). Phân token theo cấp từ có còn là ý tưởng tốt cho những trường hợp như vậy không? Tại sao hay tại sao không?


[Thảo luận](https://discuss.d2l.ai/t/1060)

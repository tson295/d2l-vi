# Subword Embedding
<a id="sec_fasttext"></a>

Trong tiếng Anh,
các từ như
"helps", "helped", và "helping" là
các dạng biến tố của cùng một từ "help".
Quan hệ
giữa "dog" và "dogs"
giống với
quan hệ giữa "cat" và "cats",
và
quan hệ
giữa "boy" và "boyfriend"
giống với
quan hệ giữa "girl" và "girlfriend".
Trong các ngôn ngữ khác
như tiếng Pháp và tiếng Tây Ban Nha,
nhiều động từ có hơn 40 dạng biến tố,
trong khi trong tiếng Phần Lan,
một danh từ có thể có tới 15 cách.
Trong ngôn ngữ học,
hình thái học nghiên cứu sự hình thành từ và quan hệ giữa các từ.
Tuy nhiên,
cấu trúc bên trong của từ
không được khai thác trong word2vec
cũng như trong GloVe.

## Mô Hình fastText

Nhắc lại cách các từ được biểu diễn trong word2vec.
Trong cả mô hình skip-gram
và mô hình continuous bag-of-words,
các dạng biến tố khác nhau của cùng một từ
được biểu diễn trực tiếp bằng các vector khác nhau
mà không chia sẻ tham số.
Để dùng thông tin hình thái,
mô hình *fastText*
đề xuất một cách tiếp cận *subword embedding*,
trong đó một subword là một $n$-gram ký tự [Bojanowski.Grave.Joulin.ea.2017].
Thay vì học các biểu diễn vector ở mức từ,
fastText có thể được xem như
skip-gram ở mức subword,
trong đó mỗi *từ trung tâm* được biểu diễn bằng tổng
các vector subword của nó.

Hãy minh họa cách lấy
subword cho mỗi từ trung tâm trong fastText
bằng từ "where".
Trước tiên, thêm các ký tự đặc biệt “&lt;” và “&gt;”
vào đầu và cuối từ để phân biệt tiền tố và hậu tố với các subword khác.
Sau đó, trích xuất các $n$-gram ký tự từ từ này.
Ví dụ, khi $n=3$,
ta thu được tất cả subword độ dài 3: "&lt;wh", "whe", "her", "ere", "re&gt;", và subword đặc biệt "&lt;where&gt;".


Trong fastText, với bất kỳ từ $w$ nào,
ký hiệu $\mathcal{G}_w$
là hợp của tất cả subword của nó có độ dài từ 3 đến 6
và subword đặc biệt của nó.
Từ vựng
là hợp các subword của tất cả các từ.
Gọi $\mathbf{z}_g$
là vector của subword $g$ trong từ điển,
vector $\mathbf{v}_w$ cho
từ $w$ khi làm từ trung tâm
trong mô hình skip-gram
là tổng các vector subword của nó:

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

Phần còn lại của fastText giống với mô hình skip-gram. So với mô hình skip-gram,
từ vựng trong fastText lớn hơn,
dẫn tới nhiều tham số mô hình hơn.
Bên cạnh đó,
để tính biểu diễn của một từ,
tất cả các vector subword của nó
phải được cộng lại,
dẫn tới độ phức tạp tính toán cao hơn.
Tuy nhiên,
nhờ các tham số được chia sẻ từ subword giữa những từ có cấu trúc tương tự,
các từ hiếm và thậm chí các từ ngoài từ vựng
có thể nhận được biểu diễn vector tốt hơn trong fastText.


## Byte Pair Encoding
<a id="subsec_Byte_Pair_Encoding"></a>

Trong fastText, tất cả các subword được trích xuất phải có độ dài đã chỉ định, chẳng hạn từ $3$ đến $6$, do đó kích thước từ vựng không thể được định trước.
Để cho phép subword có độ dài biến đổi trong một từ vựng kích thước cố định,
chúng ta có thể áp dụng một thuật toán nén
gọi là *byte pair encoding* (BPE) để trích xuất subword [Sennrich.Haddow.Birch.2015].

Byte pair encoding thực hiện phân tích thống kê tập dữ liệu huấn luyện để khám phá các ký hiệu phổ biến bên trong một từ,
chẳng hạn các chuỗi ký tự liên tiếp có độ dài bất kỳ.
Bắt đầu từ các ký hiệu độ dài 1,
byte pair encoding lặp lại việc gộp cặp ký hiệu liên tiếp xuất hiện thường xuyên nhất để tạo ra các ký hiệu mới dài hơn.
Lưu ý rằng để hiệu quả, các cặp vượt qua ranh giới từ không được xét.
Cuối cùng, chúng ta có thể dùng các ký hiệu này như subword để phân đoạn từ.
Byte pair encoding và các biến thể của nó đã được dùng cho biểu diễn đầu vào trong các mô hình tiền huấn luyện xử lý ngôn ngữ tự nhiên phổ biến như GPT-2 [Radford.Wu.Child.ea.2019] và RoBERTa [Liu.Ott.Goyal.ea.2019].
Sau đây, chúng ta sẽ minh họa cách byte pair encoding hoạt động.

Trước tiên, chúng ta khởi tạo từ vựng ký hiệu gồm tất cả ký tự chữ thường tiếng Anh, một ký hiệu kết thúc từ đặc biệt `'_'`, và một ký hiệu không biết đặc biệt `'[UNK]'`.

```python
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

Vì chúng ta không xét các cặp ký hiệu vượt qua ranh giới từ,
chúng ta chỉ cần một dictionary `raw_token_freqs` ánh xạ từ sang tần suất của chúng (số lần xuất hiện)
trong một tập dữ liệu.
Lưu ý rằng ký hiệu đặc biệt `'_'` được thêm vào cuối mỗi từ để
chúng ta có thể dễ dàng khôi phục một chuỗi từ (ví dụ, "a taller man")
từ một chuỗi ký hiệu đầu ra (ví dụ, "a_ tall er_ man").
Vì chúng ta bắt đầu quá trình gộp từ một từ vựng chỉ gồm các ký tự đơn và ký hiệu đặc biệt, dấu cách được chèn giữa mọi cặp ký tự liên tiếp trong mỗi từ (các khóa của dictionary `token_freqs`).
Nói cách khác, dấu cách là dấu phân tách giữa các ký hiệu trong một từ.

```python
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

Chúng ta định nghĩa hàm `get_max_freq_pair` sau,
trả về cặp ký hiệu liên tiếp xuất hiện thường xuyên nhất trong một từ,
trong đó các từ đến từ khóa của dictionary đầu vào `token_freqs`.

```python
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

Là một cách tiếp cận tham lam dựa trên tần suất của các ký hiệu liên tiếp,
byte pair encoding sẽ dùng hàm `merge_symbols` sau để gộp cặp ký hiệu liên tiếp xuất hiện thường xuyên nhất nhằm tạo ra ký hiệu mới.

```python
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

Bây giờ chúng ta lặp thuật toán byte pair encoding trên các khóa của dictionary `token_freqs`. Trong lần lặp đầu tiên, cặp ký hiệu liên tiếp thường gặp nhất là `'t'` và `'a'`, nên byte pair encoding gộp chúng để tạo ra ký hiệu mới `'ta'`. Trong lần lặp thứ hai, byte pair encoding tiếp tục gộp `'ta'` và `'l'` để tạo ra một ký hiệu mới khác `'tal'`.

```python
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

Sau 10 lần lặp byte pair encoding, ta có thể thấy danh sách `symbols` hiện chứa thêm 10 ký hiệu được gộp lặp lại từ các ký hiệu khác.

```python
#@tab all
print(symbols)
```

Với cùng tập dữ liệu được chỉ định trong các khóa của dictionary `raw_token_freqs`,
mỗi từ trong tập dữ liệu hiện được phân đoạn bằng các subword "fast_", "fast", "er_", "tall_", và "tall"
như kết quả của thuật toán byte pair encoding.
Chẳng hạn, các từ "faster_" và "taller_" lần lượt được phân đoạn thành "fast er_" và "tall er_".

```python
#@tab all
print(list(token_freqs.keys()))
```

Lưu ý rằng kết quả của byte pair encoding phụ thuộc vào tập dữ liệu được dùng.
Chúng ta cũng có thể dùng các subword học được từ một tập dữ liệu
để phân đoạn từ của một tập dữ liệu khác.
Là một cách tiếp cận tham lam, hàm `segment_BPE` sau cố gắng tách từ thành các subword dài nhất có thể từ đối số đầu vào `symbols`.

```python
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

Sau đây, chúng ta dùng các subword trong danh sách `symbols`, được học từ tập dữ liệu nói trên,
để phân đoạn `tokens` biểu diễn một tập dữ liệu khác.

```python
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## Tóm Tắt

* Mô hình fastText đề xuất cách tiếp cận subword embedding. Dựa trên mô hình skip-gram trong word2vec, nó biểu diễn một từ trung tâm bằng tổng các vector subword của nó.
* Byte pair encoding thực hiện phân tích thống kê tập dữ liệu huấn luyện để khám phá các ký hiệu phổ biến bên trong một từ. Là một cách tiếp cận tham lam, byte pair encoding lặp lại việc gộp cặp ký hiệu liên tiếp xuất hiện thường xuyên nhất.
* Subword embedding có thể cải thiện chất lượng biểu diễn của các từ hiếm và các từ ngoài từ điển.

## Bài Tập

1. Ví dụ, có khoảng $3\times 10^8$ $6$-gram khả dĩ trong tiếng Anh. Vấn đề gì xảy ra khi có quá nhiều subword? Làm thế nào để xử lý vấn đề này? Gợi ý: tham khảo phần cuối Mục 3.2 của bài báo fastText [Bojanowski.Grave.Joulin.ea.2017].
1. Làm thế nào để thiết kế một mô hình subword embedding dựa trên mô hình continuous bag-of-words?
1. Để có một từ vựng kích thước $m$, cần bao nhiêu phép gộp khi kích thước từ vựng ký hiệu ban đầu là $n$?
1. Làm thế nào để mở rộng ý tưởng byte pair encoding để trích xuất cụm từ?


[Thảo luận](https://discuss.d2l.ai/t/4587)

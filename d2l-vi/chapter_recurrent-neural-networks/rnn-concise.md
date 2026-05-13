# Triển khai Súc tích Mạng Nơ-ron Hồi quy
<a id="sec_rnn-concise"></a>

Giống như hầu hết các triển khai từ đầu của chúng ta,
[sec_rnn-scratch](#sec_rnn-scratch) được thiết kế
để cung cấp cái nhìn sâu sắc về cách mỗi thành phần hoạt động.
Nhưng khi bạn sử dụng RNN hàng ngày
hoặc viết mã sản xuất,
bạn sẽ muốn dựa vào các thư viện nhiều hơn
để giảm cả thời gian triển khai
(bằng cách cung cấp mã thư viện cho các mô hình và hàm phổ biến)
và thời gian tính toán
(bằng cách tối ưu hóa triệt để các triển khai thư viện này).
Phần này sẽ chỉ cho bạn cách triển khai
cùng mô hình ngôn ngữ hiệu quả hơn
sử dụng API cấp cao được cung cấp
bởi framework deep learning của bạn.
Chúng ta bắt đầu, như trước, bằng cách tải
tập dữ liệu *The Time Machine*.


```python
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```


## [**Định nghĩa Mô hình**]

Chúng ta định nghĩa lớp sau
sử dụng RNN được triển khai
bởi các API cấp cao.


```python
class RNN(d2l.Module):  
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```


Kế thừa từ lớp `RNNLMScratch` trong [sec_rnn-scratch](#sec_rnn-scratch),
lớp `RNNLM` sau đây định nghĩa một mô hình ngôn ngữ hoàn chỉnh dựa trên RNN.
Lưu ý rằng chúng ta cần tạo một lớp đầu ra kết nối đầy đủ riêng biệt.

```python
class RNNLM(d2l.RNNLMScratch):  
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```


## Huấn luyện và Dự đoán

Trước khi huấn luyện mô hình, hãy [**thực hiện dự đoán
với mô hình được khởi tạo với các trọng số ngẫu nhiên.**]
Vì chúng ta chưa huấn luyện mạng,
nó sẽ tạo ra các dự đoán vô nghĩa.


Tiếp theo, chúng ta [**huấn luyện mô hình, tận dụng API cấp cao**].


So với [sec_rnn-scratch](#sec_rnn-scratch),
mô hình này đạt được perplexity có thể so sánh,
nhưng chạy nhanh hơn do các triển khai được tối ưu hóa.
Như trước, chúng ta có thể tạo ra các token dự đoán
theo chuỗi tiền tố đã chỉ định.


## Tóm tắt

Các API cấp cao trong các framework deep learning cung cấp các triển khai của các RNN chuẩn.
Các thư viện này giúp bạn tránh lãng phí thời gian triển khai lại các mô hình chuẩn.
Hơn nữa,
các triển khai framework thường được tối ưu hóa cao,
dẫn đến lợi ích hiệu suất (tính toán) đáng kể
khi so sánh với các triển khai từ đầu.

## Bài tập

1. Bạn có thể làm cho mô hình RNN quá khớp bằng cách sử dụng API cấp cao không?
1. Triển khai mô hình tự hồi quy của [sec_sequence](#sec_sequence) sử dụng RNN.


[Discussions](https://discuss.d2l.ai/t/1053)

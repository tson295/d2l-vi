# Triển khai Mạng Nơ-ron Hồi quy từ Đầu
<a id="sec_rnn-scratch"></a>

Bây giờ chúng ta đã sẵn sàng để triển khai RNN từ đầu.
Cụ thể, chúng ta sẽ huấn luyện RNN này để hoạt động
như một mô hình ngôn ngữ cấp ký tự
(xem [sec_rnn](#sec_rnn))
và huấn luyện nó trên kho ngữ liệu bao gồm
toàn bộ văn bản của *The Time Machine* của H. G. Wells,
theo các bước xử lý dữ liệu
được phác thảo trong [sec_text-sequence](#sec_text-sequence).
Chúng ta bắt đầu bằng cách tải tập dữ liệu.


```python
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```


## Mô hình RNN

Chúng ta bắt đầu bằng cách định nghĩa một lớp
để triển khai mô hình RNN
([subsec_rnn_w_hidden_states](#subsec_rnn_w_hidden_states)).
Lưu ý rằng số đơn vị ẩn `num_hiddens`
là một siêu tham số có thể điều chỉnh.


[**Phương thức `forward` dưới đây định nghĩa cách tính toán
đầu ra và trạng thái ẩn tại bất kỳ bước thời gian nào,
khi có đầu vào hiện tại và trạng thái của mô hình
tại bước thời gian trước đó.**]
Lưu ý rằng mô hình RNN lặp qua
chiều ngoài cùng của `inputs`,
cập nhật trạng thái ẩn
từng bước thời gian một.
Mô hình ở đây sử dụng hàm kích hoạt $\tanh$ ([subsec_tanh](#subsec_tanh)).


Chúng ta có thể đưa một minibatch các chuỗi đầu vào vào mô hình RNN như sau.


Hãy kiểm tra xem mô hình RNN
có tạo ra kết quả với các hình dạng đúng không
để đảm bảo rằng chiều của
trạng thái ẩn vẫn không thay đổi.

```python
def check_len(a, n):  
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'
    
def check_shape(a, shape):  
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## Mô hình Ngôn ngữ Dựa trên RNN

Lớp `RNNLMScratch` sau đây định nghĩa
một mô hình ngôn ngữ dựa trên RNN,
trong đó chúng ta truyền RNN của mình
qua đối số `rnn`
của phương thức `__init__`.
Khi huấn luyện các mô hình ngôn ngữ,
đầu vào và đầu ra là
từ cùng một từ vựng.
Do đó, chúng có cùng chiều,
bằng với kích thước từ vựng.
Lưu ý rằng chúng ta sử dụng perplexity để đánh giá mô hình.
Như đã thảo luận trong [subsec_perplexity](#subsec_perplexity), điều này đảm bảo
rằng các chuỗi có độ dài khác nhau có thể so sánh được.

```python
class RNNLMScratch(d2l.Classifier):  
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```


### [**Mã hóa One-Hot**]

Nhắc lại rằng mỗi token được biểu diễn
bởi một chỉ số số học chỉ ra vị trí
trong từ vựng của
từ/ký tự/mảnh từ tương ứng.
Bạn có thể bị cám dỗ để xây dựng mạng nơ-ron
với một nút đầu vào duy nhất (tại mỗi bước thời gian),
trong đó chỉ số có thể được đưa vào như một giá trị vô hướng.
Điều này hoạt động khi chúng ta xử lý các đầu vào số học
như giá hoặc nhiệt độ, trong đó bất kỳ hai giá trị nào
đủ gần nhau
nên được xử lý tương tự.
Nhưng điều này không thực sự có ý nghĩa.
Từ thứ $45^{\textrm{th}}$ và $46^{\textrm{th}}$
trong từ vựng của chúng ta là "their" và "said",
nghĩa của chúng không hề giống nhau.

Khi xử lý dữ liệu phân loại như vậy,
chiến lược phổ biến nhất là biểu diễn
mỗi mục bằng một *mã hóa one-hot*
(nhớ lại từ [subsec_classification-problem](#subsec_classification-problem)).
Mã hóa one-hot là một vector có độ dài
được cho bởi kích thước của từ vựng $N$,
trong đó tất cả các mục được đặt thành $0$,
ngoại trừ mục tương ứng
với token của chúng ta, được đặt thành $1$.
Ví dụ, nếu từ vựng có năm phần tử,
thì các vector one-hot tương ứng
với các chỉ số 0 và 2 sẽ như sau.


```python
F.one_hot(torch.tensor([0, 2]), 5)
```


(**Các minibatch mà chúng ta lấy mẫu tại mỗi lần lặp
sẽ có hình dạng (kích thước batch, số bước thời gian).
Sau khi biểu diễn mỗi đầu vào như một vector one-hot,
chúng ta có thể coi mỗi minibatch như một tensor ba chiều,
trong đó độ dài dọc theo trục thứ ba
được cho bởi kích thước từ vựng (`len(vocab)`).**)
Chúng ta thường hoán vị đầu vào để chúng ta sẽ thu được đầu ra
có hình dạng (số bước thời gian, kích thước batch, kích thước từ vựng).
Điều này sẽ cho phép chúng ta lặp thuận tiện hơn qua chiều ngoài cùng
để cập nhật trạng thái ẩn của một minibatch,
từng bước thời gian
(ví dụ: trong phương thức `forward` ở trên).

```python
@d2l.add_to_class(RNNLMScratch)  
def one_hot(self, X):    
    # Output shape: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### Biến đổi Đầu ra RNN

Mô hình ngôn ngữ sử dụng một lớp đầu ra kết nối đầy đủ
để biến đổi đầu ra RNN thành các dự đoán token tại mỗi bước thời gian.

```python
@d2l.add_to_class(RNNLMScratch)  
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

Hãy [**kiểm tra xem tính toán xuôi
có tạo ra đầu ra với hình dạng đúng không.**]


## [**Cắt Gradient**]


Mặc dù bạn đã quen với việc nghĩ đến các mạng nơ-ron
là "sâu" theo nghĩa là nhiều lớp
tách đầu vào và đầu ra
ngay cả trong một bước thời gian duy nhất,
độ dài của chuỗi giới thiệu
một khái niệm mới về độ sâu.
Ngoài việc đi qua mạng
theo hướng đầu vào-đến-đầu ra,
các đầu vào tại bước thời gian đầu tiên
phải đi qua một chuỗi $T$ lớp
dọc theo các bước thời gian để
ảnh hưởng đến đầu ra của mô hình
tại bước thời gian cuối cùng.
Nhìn theo chiều ngược lại, trong mỗi lần lặp,
chúng ta lan truyền ngược các gradient qua thời gian,
dẫn đến một chuỗi các tích ma trận
có độ dài $\mathcal{O}(T)$.
Như đề cập trong [sec_numerical_stability](#sec_numerical_stability),
điều này có thể dẫn đến bất ổn số học,
khiến các gradient bùng nổ hoặc biến mất,
tùy thuộc vào các tính chất của các ma trận trọng số.

Xử lý gradient biến mất và bùng nổ
là một vấn đề cơ bản khi thiết kế RNN
và đã truyền cảm hứng cho một số tiến bộ lớn nhất
trong các kiến trúc mạng nơ-ron hiện đại.
Trong chương tiếp theo, chúng ta sẽ nói về
các kiến trúc chuyên biệt được thiết kế
với hy vọng giảm thiểu vấn đề gradient biến mất.
Tuy nhiên, ngay cả các RNN hiện đại thường bị
gradient bùng nổ.
Một giải pháp không tao nhã nhưng phổ biến
là đơn giản cắt các gradient
ép buộc các gradient "bị cắt" kết quả
nhận các giá trị nhỏ hơn.


Nói chung, khi tối ưu hóa một số mục tiêu
bằng gradient descent, chúng ta cập nhật lặp đi lặp lại
tham số quan tâm, giả sử một vector $\mathbf{x}$,
nhưng đẩy nó theo hướng của
gradient âm $\mathbf{g}$
(trong gradient descent ngẫu nhiên,
chúng ta tính toán gradient này
trên một minibatch được lấy mẫu ngẫu nhiên).
Ví dụ, với tốc độ học $\eta > 0$,
mỗi cập nhật có dạng
$\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$.
Hãy tiếp tục giả định rằng hàm mục tiêu $f$
đủ mượt.
Chính thức, chúng ta nói rằng mục tiêu
là *Lipschitz liên tục* với hằng số $L$,
nghĩa là với bất kỳ $\mathbf{x}$ và $\mathbf{y}$, chúng ta có

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

Như bạn có thể thấy, khi chúng ta cập nhật vector tham số bằng cách trừ $\eta \mathbf{g}$,
sự thay đổi trong giá trị của mục tiêu
phụ thuộc vào tốc độ học,
chuẩn của gradient và $L$ như sau:

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

Nói cách khác, mục tiêu không thể
thay đổi nhiều hơn $L \eta \|\mathbf{g}\|$.
Có một giá trị nhỏ cho giới hạn trên này
có thể được coi là tốt hoặc xấu.
Mặt trái, chúng ta đang giới hạn tốc độ
mà chúng ta có thể giảm giá trị của mục tiêu.
Mặt sáng, điều này giới hạn bao nhiêu
chúng ta có thể đi sai trong bất kỳ bước gradient nào.


Khi chúng ta nói rằng gradient bùng nổ,
chúng ta có nghĩa là $\|\mathbf{g}\|$
trở nên quá lớn.
Trong trường hợp xấu nhất này, chúng ta có thể gây ra nhiều
thiệt hại trong một bước gradient duy nhất đến mức chúng ta
có thể hoàn tác tất cả tiến bộ đã đạt được trong
hàng nghìn lần lặp huấn luyện.
Khi gradient có thể lớn như vậy,
việc huấn luyện mạng nơ-ron thường phân kỳ,
không giảm giá trị của mục tiêu.
Những lúc khác, việc huấn luyện cuối cùng hội tụ
nhưng không ổn định do các đột biến lớn trong mất mát.


Một cách để giới hạn kích thước của $L \eta \|\mathbf{g}\|$
là thu nhỏ tốc độ học $\eta$ xuống giá trị rất nhỏ.
Điều này có ưu điểm là chúng ta không làm lệch hướng các cập nhật.
Nhưng nếu chúng ta chỉ *hiếm khi* nhận được gradient lớn thì sao?
Bước đi quyết liệt này làm chậm tiến trình của chúng ta ở tất cả các bước,
chỉ để xử lý các sự kiện gradient bùng nổ hiếm gặp.
Một giải pháp thay thế phổ biến là áp dụng heuristic *cắt gradient*
chiếu các gradient $\mathbf{g}$ lên một quả cầu
của một bán kính $\theta$ nhất định như sau:

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

Điều này đảm bảo rằng chuẩn gradient không bao giờ vượt quá $\theta$
và gradient được cập nhật hoàn toàn căn chỉnh
với hướng ban đầu của $\mathbf{g}$.
Nó cũng có tác dụng phụ mong muốn
là giới hạn ảnh hưởng mà bất kỳ minibatch nào
(và trong đó bất kỳ mẫu nào)
có thể tác động lên vector tham số.
Điều này mang lại một mức độ độ bền nhất định cho mô hình.
Để rõ ràng, đây là một thủ thuật.
Cắt gradient có nghĩa là chúng ta không phải lúc nào cũng
tuân theo gradient thực sự và khó
để lý luận phân tích về các tác dụng phụ có thể có.
Tuy nhiên, đây là một thủ thuật rất hữu ích,
và được áp dụng rộng rãi trong các triển khai RNN
trong hầu hết các framework deep learning.


Dưới đây chúng ta định nghĩa một phương thức để cắt gradient,
được gọi bởi phương thức `fit_epoch` của
lớp `d2l.Trainer` (xem [sec_linear_scratch](#sec_linear_scratch)).
Lưu ý rằng khi tính toán chuẩn gradient,
chúng ta đang ghép tất cả các tham số mô hình,
coi chúng như một vector tham số khổng lồ duy nhất.


```python
@d2l.add_to_class(d2l.Trainer)  
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```


## Huấn luyện

Sử dụng tập dữ liệu *The Time Machine* (`data`),
chúng ta huấn luyện một mô hình ngôn ngữ cấp ký tự (`model`)
dựa trên RNN (`rnn`) được triển khai từ đầu.
Lưu ý rằng trước tiên chúng ta tính toán các gradient,
sau đó cắt chúng, và cuối cùng
cập nhật các tham số mô hình
sử dụng các gradient đã cắt.

```python
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Giải mã

Khi một mô hình ngôn ngữ đã được học,
chúng ta có thể sử dụng nó không chỉ để dự đoán token tiếp theo
mà còn để tiếp tục dự đoán mỗi token tiếp theo,
coi token được dự đoán trước đó như thể
nó là token tiếp theo trong đầu vào.
Đôi khi chúng ta chỉ muốn tạo văn bản
như thể chúng ta đang bắt đầu từ đầu
của một tài liệu.
Tuy nhiên, thường hữu ích hơn khi điều kiện hóa
mô hình ngôn ngữ trên một tiền tố do người dùng cung cấp.
Ví dụ, nếu chúng ta đang phát triển
tính năng tự hoàn thành cho công cụ tìm kiếm
hoặc hỗ trợ người dùng trong việc viết email,
chúng ta sẽ muốn đưa vào những gì họ
đã viết cho đến nay (tiền tố),
và sau đó tạo ra một phần tiếp theo có thể.


[**Phương thức `predict` sau đây
tạo ra một phần tiếp theo, một ký tự tại một thời điểm,
sau khi tiếp nhận một `prefix` do người dùng cung cấp**].
Khi lặp qua các ký tự trong `prefix`,
chúng ta tiếp tục truyền trạng thái ẩn
đến bước thời gian tiếp theo
nhưng không tạo ra bất kỳ đầu ra nào.
Đây được gọi là giai đoạn *khởi động*.
Sau khi tiếp nhận tiền tố, bây giờ chúng ta
sẵn sàng bắt đầu phát ra các ký tự tiếp theo,
mỗi ký tự sẽ được đưa trở lại mô hình
như đầu vào tại bước thời gian tiếp theo.


Trong phần sau, chúng ta chỉ định tiền tố
và để nó tạo ra 20 ký tự bổ sung.


Mặc dù việc triển khai mô hình RNN trên từ đầu là có tính hướng dẫn, nhưng không thuận tiện.
Trong phần tiếp theo, chúng ta sẽ xem cách tận dụng các framework deep learning để xây dựng nhanh các RNN
sử dụng các kiến trúc chuẩn, và thu được các lợi ích hiệu suất
bằng cách dựa vào các hàm thư viện được tối ưu hóa cao.


## Tóm tắt

Chúng ta có thể huấn luyện các mô hình ngôn ngữ dựa trên RNN để tạo ra văn bản theo tiền tố văn bản do người dùng cung cấp.
Một mô hình ngôn ngữ RNN đơn giản bao gồm mã hóa đầu vào, mô hình hóa RNN, và tạo đầu ra.
Trong quá trình huấn luyện, cắt gradient có thể giảm thiểu vấn đề gradient bùng nổ nhưng không giải quyết vấn đề gradient biến mất. Trong thí nghiệm, chúng ta đã triển khai một mô hình ngôn ngữ RNN đơn giản và huấn luyện nó với cắt gradient trên các chuỗi văn bản, được token hóa ở cấp độ ký tự. Bằng cách điều kiện hóa trên một tiền tố, chúng ta có thể sử dụng mô hình ngôn ngữ để tạo ra các phần tiếp theo có thể, điều này tỏ ra hữu ích trong nhiều ứng dụng, ví dụ: tính năng tự hoàn thành.


## Bài tập

1. Mô hình ngôn ngữ được triển khai có dự đoán token tiếp theo dựa trên tất cả các token trong quá khứ đến token đầu tiên trong *The Time Machine* không?
1. Siêu tham số nào kiểm soát độ dài lịch sử được sử dụng để dự đoán?
1. Chứng minh rằng mã hóa one-hot tương đương với việc chọn một nhúng khác nhau cho mỗi đối tượng.
1. Điều chỉnh các siêu tham số (ví dụ: số epoch, số đơn vị ẩn, số bước thời gian trong một minibatch, và tốc độ học) để cải thiện perplexity. Bạn có thể đạt được mức thấp bao nhiêu trong khi vẫn sử dụng kiến trúc đơn giản này?
1. Thay thế mã hóa one-hot bằng nhúng có thể học được. Điều này có dẫn đến hiệu suất tốt hơn không?
1. Tiến hành thí nghiệm để xác định mô hình ngôn ngữ này
   được huấn luyện trên *The Time Machine* hoạt động tốt như thế nào trên các cuốn sách khác của H. G. Wells,
   ví dụ: *The War of the Worlds*.
1. Tiến hành một thí nghiệm khác để đánh giá perplexity của mô hình này
   trên các cuốn sách được viết bởi các tác giả khác.
1. Sửa đổi phương thức dự đoán để sử dụng lấy mẫu
   thay vì chọn ký tự có xác suất cao nhất tiếp theo.
    * Điều gì xảy ra?
    * Làm lệch mô hình về phía các đầu ra có xác suất cao hơn, ví dụ:
    bằng cách lấy mẫu từ $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ với $\alpha > 1$.
1. Chạy mã trong phần này mà không cắt gradient. Điều gì xảy ra?
1. Thay thế hàm kích hoạt được sử dụng trong phần này bằng ReLU
   và lặp lại các thí nghiệm trong phần này. Chúng ta có còn cần cắt gradient không? Tại sao?


[Discussions](https://discuss.d2l.ai/t/486)

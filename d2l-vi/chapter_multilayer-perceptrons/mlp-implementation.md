# Cài đặt Perceptron Đa lớp
<a id="sec_mlp-implementation"></a>

Perceptron đa lớp (MLP) không phức tạp hơn nhiều để cài đặt so với các mô hình tuyến tính đơn giản. Sự khác biệt khái niệm chính
là chúng ta bây giờ ghép nối nhiều lớp.


```python
from d2l import torch as d2l
import torch
from torch import nn
```


## Cài đặt từ Đầu

Hãy bắt đầu lại bằng cách cài đặt mạng như vậy từ đầu.

### Khởi tạo Tham số Mô hình

Nhớ lại rằng Fashion-MNIST chứa 10 lớp,
và mỗi ảnh bao gồm lưới pixel xám $28 \times 28 = 784$.
Như trước đây, chúng ta sẽ bỏ qua cấu trúc không gian
giữa các pixel hiện tại,
vì vậy chúng ta có thể coi đây là bộ dữ liệu phân loại
với 784 đặc trưng đầu vào và 10 lớp.
Để bắt đầu, chúng ta sẽ [**cài đặt MLP
với một lớp ẩn và 256 đơn vị ẩn.**]
Cả số lớp lẫn chiều rộng của chúng đều có thể điều chỉnh
(chúng được coi là siêu tham số).
Thường thì chúng ta chọn chiều rộng lớp chia hết cho lũy thừa lớn hơn của 2.
Điều này hiệu quả về mặt tính toán do cách
bộ nhớ được cấp phát và địa chỉ hóa trong phần cứng.

Một lần nữa, chúng ta sẽ biểu diễn các tham số với nhiều tensor.
Lưu ý rằng *với mỗi lớp*, chúng ta phải theo dõi
một ma trận trọng số và một vector hệ số chặn.
Như thường lệ, chúng ta cấp phát bộ nhớ
cho gradient của mất mát theo các tham số này.


Trong code bên dưới, chúng ta dùng `nn.Parameter`
để tự động đăng ký
một thuộc tính lớp như là tham số được theo dõi bởi `autograd` ([sec_autograd](#sec_autograd)).


```python
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```


### Mô hình

Để đảm bảo chúng ta hiểu mọi thứ hoạt động như thế nào,
chúng ta sẽ [**tự cài đặt kích hoạt ReLU**]
thay vì gọi trực tiếp hàm `relu` tích hợp.


```python
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```


Vì chúng ta bỏ qua cấu trúc không gian,
chúng ta `reshape` mỗi ảnh hai chiều thành
vector phẳng có độ dài `num_inputs`.
Cuối cùng, chúng ta (**cài đặt mô hình**)
chỉ với vài dòng code. Vì chúng ta sử dụng autograd tích hợp của framework, đây là tất cả những gì cần thiết.

```python
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### Huấn luyện

May mắn thay, [**vòng lặp huấn luyện cho MLP
giống hệt như cho hồi quy softmax.**] Chúng ta định nghĩa mô hình, dữ liệu, và trainer, rồi cuối cùng gọi phương thức `fit` trên mô hình và dữ liệu.

```python
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Cài đặt Gọn

Như bạn có thể kỳ vọng, bằng cách dựa vào API cấp cao, chúng ta có thể cài đặt MLP thậm chí gọn hơn.

### Mô hình

So với cài đặt gọn
của cài đặt hồi quy softmax
([sec_softmax_concise](#sec_softmax_concise)),
sự khác biệt duy nhất là chúng ta thêm
*hai* lớp kết nối đầy đủ khi trước đây chúng ta chỉ thêm *một*.
Cái đầu tiên là [**lớp ẩn**],
cái thứ hai là lớp đầu ra.


```python
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```


Trước đây, chúng ta đã định nghĩa các phương thức `forward` cho mô hình để biến đổi đầu vào bằng các tham số mô hình.
Các phép toán này về cơ bản là một pipeline:
bạn lấy một đầu vào và
áp dụng một phép biến đổi (ví dụ:
nhân ma trận với trọng số tiếp theo là cộng hệ số chặn),
sau đó liên tục sử dụng đầu ra của phép biến đổi hiện tại như là
đầu vào cho phép biến đổi tiếp theo.
Tuy nhiên, bạn có thể nhận thấy rằng
không có phương thức `forward` nào được định nghĩa ở đây.
Thực ra, `MLP` kế thừa phương thức `forward` từ lớp `Module` ([subsec_oo-design-models](#subsec_oo-design-models)) để
đơn giản gọi `self.net(X)` (`X` là đầu vào),
hiện được định nghĩa là một chuỗi phép biến đổi
thông qua lớp `Sequential`.
Lớp `Sequential` trừu tượng hóa quá trình forward
cho phép chúng ta tập trung vào các phép biến đổi.
Chúng ta sẽ thảo luận thêm về cách lớp `Sequential` hoạt động trong [subsec_model-construction-sequential](#subsec_model-construction-sequential).


### Huấn luyện

[**Vòng lặp huấn luyện**] hoàn toàn giống
như khi chúng ta cài đặt hồi quy softmax.
Tính mô-đun này cho phép chúng ta tách
các vấn đề liên quan đến kiến trúc mô hình
khỏi các xem xét trực giao.

```python
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## Tóm tắt

Bây giờ chúng ta có nhiều kinh nghiệm hơn trong việc thiết kế các mạng sâu, bước từ một lớp đơn đến nhiều lớp của mạng sâu không còn đặt ra thách thức đáng kể nữa. Đặc biệt, chúng ta có thể tái sử dụng thuật toán huấn luyện và data loader. Tuy nhiên, lưu ý rằng việc cài đặt MLP từ đầu vẫn rộn ràng: đặt tên và theo dõi các tham số mô hình khiến việc mở rộng mô hình trở nên khó khăn. Ví dụ, hãy tưởng tượng muốn chèn thêm một lớp giữa lớp 42 và 43. Điều này bây giờ có thể là lớp 42b, trừ khi chúng ta sẵn sàng thực hiện đổi tên tuần tự. Hơn nữa, nếu chúng ta cài đặt mạng từ đầu, framework sẽ khó thực hiện các tối ưu hóa hiệu suất có ý nghĩa hơn.

Tuy nhiên, bây giờ bạn đã đạt đến trình độ nghệ thuật của cuối những năm 1980 khi các mạng sâu kết nối đầy đủ là phương pháp được lựa chọn cho mô hình hóa mạng nơ-ron. Bước khái niệm tiếp theo của chúng ta sẽ là xem xét ảnh. Trước khi làm điều đó, chúng ta cần xem xét một số kiến thức cơ bản về thống kê và chi tiết về cách tính toán mô hình hiệu quả.


## Bài tập

1. Thay đổi số đơn vị ẩn `num_hiddens` và vẽ đồ thị cách số này ảnh hưởng đến độ chính xác của mô hình. Giá trị tốt nhất của siêu tham số này là gì?
1. Thử thêm một lớp ẩn để xem nó ảnh hưởng như thế nào đến kết quả.
1. Tại sao việc chèn một lớp ẩn chỉ có một nơ-ron là ý tưởng tệ? Điều gì có thể xảy ra?
1. Việc thay đổi tốc độ học thay đổi kết quả của bạn như thế nào? Với tất cả các tham số khác cố định, tốc độ học nào cho kết quả tốt nhất? Điều này liên quan như thế nào đến số epoch?
1. Hãy tối ưu hóa trên tất cả siêu tham số cùng nhau, tức là tốc độ học, số epoch, số lớp ẩn, và số đơn vị ẩn mỗi lớp.
    1. Kết quả tốt nhất bạn có thể đạt được bằng cách tối ưu hóa tất cả chúng là gì?
    1. Tại sao việc xử lý nhiều siêu tham số lại khó khăn hơn nhiều?
    1. Mô tả một chiến lược hiệu quả để tối ưu hóa nhiều tham số cùng nhau.
1. So sánh tốc độ của framework và cài đặt từ đầu cho một bài toán thách thức. Nó thay đổi như thế nào với độ phức tạp của mạng?
1. Đo tốc độ của phép nhân tensor-ma trận cho các ma trận căn chỉnh tốt và không căn chỉnh tốt. Ví dụ: kiểm tra với ma trận có chiều 1024, 1025, 1026, 1028, và 1032.
    1. Điều này thay đổi như thế nào giữa GPU và CPU?
    1. Xác định độ rộng bus bộ nhớ của CPU và GPU của bạn.
1. Thử các hàm kích hoạt khác nhau. Hàm nào hoạt động tốt nhất?
1. Có sự khác biệt giữa các cách khởi tạo trọng số của mạng không? Nó có quan trọng không?


[Discussions](https://discuss.d2l.ai/t/93)

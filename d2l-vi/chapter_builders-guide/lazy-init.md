# Khởi tạo Lười biếng
<a id="sec_lazy_init"></a>

Cho đến nay, có vẻ như chúng ta đã thoát khỏi
việc thiếu cẩn thận trong việc thiết lập mạng của mình.
Cụ thể, chúng ta đã làm những việc không trực quan sau đây,
có vẻ như không nên hoạt động:

* Chúng ta đã định nghĩa kiến trúc mạng
  mà không chỉ định chiều đầu vào.
* Chúng ta đã thêm các lớp mà không chỉ định
  chiều đầu ra của lớp trước.
* Chúng ta thậm chí đã "khởi tạo" các tham số này
  trước khi cung cấp đủ thông tin để xác định
  mô hình của chúng ta nên chứa bao nhiêu tham số.

Bạn có thể ngạc nhiên khi code của chúng ta chạy được.
Xét cho cùng, không có cách nào framework deep learning
có thể biết chiều đầu vào của một mạng sẽ là bao nhiêu.
Thủ thuật ở đây là framework *hoãn khởi tạo*,
chờ cho đến lần đầu tiên chúng ta truyền dữ liệu qua mô hình,
để suy ra kích thước của mỗi lớp một cách nhanh chóng.


Về sau, khi làm việc với các mạng nơ-ron tích chập,
kỹ thuật này sẽ trở nên thậm chí tiện lợi hơn
vì chiều đầu vào
(ví dụ: độ phân giải của hình ảnh)
sẽ ảnh hưởng đến chiều
của mỗi lớp tiếp theo.
Do đó, khả năng đặt tham số
mà không cần biết,
tại thời điểm viết code,
giá trị của chiều
có thể đơn giản hóa đáng kể nhiệm vụ chỉ định
và sau đó sửa đổi các mô hình của chúng ta.
Tiếp theo, chúng ta đi sâu hơn vào cơ chế khởi tạo.


```python
from d2l import torch as d2l
import torch
from torch import nn
```


Để bắt đầu, hãy khởi tạo một MLP.


```python
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```


Ở thời điểm này, mạng không thể biết
chiều của trọng số lớp đầu vào
vì chiều đầu vào vẫn chưa biết.


```python
net[0].weight
```


Tiếp theo hãy truyền dữ liệu qua mạng
để framework cuối cùng khởi tạo các tham số.


```python
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```


Ngay khi chúng ta biết chiều đầu vào,
20,
framework có thể xác định hình dạng của ma trận trọng số lớp đầu tiên bằng cách thay vào giá trị 20.
Sau khi nhận ra hình dạng của lớp đầu tiên, framework tiến hành
đến lớp thứ hai,
và cứ thế qua đồ thị tính toán
cho đến khi tất cả các hình dạng được biết.
Lưu ý rằng trong trường hợp này,
chỉ lớp đầu tiên yêu cầu khởi tạo lười biếng,
nhưng framework khởi tạo tuần tự.
Một khi tất cả các hình dạng tham số được biết,
framework cuối cùng có thể khởi tạo các tham số.

Phương thức sau
truyền các đầu vào giả
qua mạng
để chạy thử
suy ra tất cả các hình dạng tham số
và sau đó khởi tạo các tham số.
Nó sẽ được sử dụng sau này khi không muốn khởi tạo ngẫu nhiên mặc định.


```python
@d2l.add_to_class(d2l.Module)  
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```


## Tóm tắt

Khởi tạo lười biếng có thể thuận tiện, cho phép framework tự động suy ra hình dạng tham số, giúp dễ dàng sửa đổi kiến trúc và loại bỏ một nguồn lỗi phổ biến.
Chúng ta có thể truyền dữ liệu qua mô hình để framework cuối cùng khởi tạo các tham số.


## Bài tập

1. Điều gì xảy ra nếu bạn chỉ định chiều đầu vào cho lớp đầu tiên nhưng không cho các lớp tiếp theo? Bạn có nhận được khởi tạo ngay lập tức không?
1. Điều gì xảy ra nếu bạn chỉ định các chiều không khớp?
1. Bạn cần làm gì nếu có đầu vào có chiều thay đổi? Gợi ý: hãy xem ràng buộc tham số.


[Discussions](https://discuss.d2l.ai/t/8092)

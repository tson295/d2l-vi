# Khởi tạo Tham số

Bây giờ khi chúng ta biết cách truy cập các tham số,
hãy xem cách khởi tạo chúng đúng cách.
Chúng ta đã thảo luận về nhu cầu khởi tạo đúng cách trong [sec_numerical_stability](#sec_numerical_stability).
Framework deep learning cung cấp khởi tạo ngẫu nhiên mặc định cho các lớp của nó.
Tuy nhiên, chúng ta thường muốn khởi tạo trọng số của mình
theo các giao thức khác nhau khác. Framework cung cấp hầu hết các
giao thức thường được sử dụng, và cũng cho phép tạo bộ khởi tạo tùy chỉnh.


```python
import torch
from torch import nn
```


Theo mặc định, PyTorch khởi tạo các ma trận trọng số và hệ số chặn
đồng đều bằng cách lấy mẫu từ một phạm vi được tính toán theo chiều đầu vào và đầu ra.
Mô-đun `nn.init` của PyTorch cung cấp nhiều
phương pháp khởi tạo được đặt sẵn.


```python
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```


## [**Khởi tạo Tích hợp sẵn**]

Hãy bắt đầu bằng cách gọi các bộ khởi tạo tích hợp sẵn.
Code dưới đây khởi tạo tất cả các tham số trọng số
như các biến ngẫu nhiên Gaussian
với độ lệch chuẩn 0.01, trong khi các tham số hệ số chặn được xóa về không.


```python
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```


Chúng ta cũng có thể khởi tạo tất cả các tham số
về một giá trị hằng số nhất định (ví dụ: 1).


```python
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```


[**Chúng ta cũng có thể áp dụng các bộ khởi tạo khác nhau cho các block nhất định.**]
Ví dụ, dưới đây chúng ta khởi tạo lớp đầu tiên
với bộ khởi tạo Xavier
và khởi tạo lớp thứ hai
về một giá trị hằng số 42.


```python
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```


### [**Khởi tạo Tùy chỉnh**]

Đôi khi, các phương pháp khởi tạo mà chúng ta cần
không được cung cấp bởi framework deep learning.
Trong ví dụ dưới đây, chúng ta định nghĩa một bộ khởi tạo
cho bất kỳ tham số trọng số $w$ nào bằng cách sử dụng phân phối kỳ lạ sau:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ với xác suất } \frac{1}{4} \\
            0    & \textrm{ với xác suất } \frac{1}{2} \\
        U(-10, -5) & \textrm{ với xác suất } \frac{1}{4}
    \end{cases}
\end{aligned}
$$


Một lần nữa, chúng ta cài đặt hàm `my_init` để áp dụng cho `net`.


```python
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```


```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```


## Tóm tắt

Chúng ta có thể khởi tạo tham số bằng các bộ khởi tạo tích hợp sẵn và tùy chỉnh.

## Bài tập

Tra cứu tài liệu trực tuyến để biết thêm về các bộ khởi tạo tích hợp sẵn.


[Discussions](https://discuss.d2l.ai/t/8090)

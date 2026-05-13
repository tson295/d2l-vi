# Quản lý Tham số

Khi chúng ta đã chọn một kiến trúc
và thiết lập các siêu tham số,
chúng ta tiến hành vòng lặp huấn luyện,
nơi mục tiêu là tìm các giá trị tham số
giảm thiểu hàm mất mát của chúng ta.
Sau khi huấn luyện, chúng ta sẽ cần những tham số này
để thực hiện các dự đoán trong tương lai.
Ngoài ra, đôi khi chúng ta sẽ muốn
trích xuất các tham số
có thể để tái sử dụng chúng trong một bối cảnh khác,
để lưu mô hình vào đĩa để
nó có thể được thực thi trong phần mềm khác,
hoặc để kiểm tra với hy vọng
đạt được sự hiểu biết khoa học.

Hầu hết thời gian, chúng ta sẽ có thể
bỏ qua các chi tiết tỉ mỉ
về cách các tham số được khai báo
và thao tác, dựa vào các framework deep learning
để làm công việc nặng nhọc.
Tuy nhiên, khi chúng ta rời khỏi
các kiến trúc chồng xếp với các lớp tiêu chuẩn,
đôi khi chúng ta sẽ cần đi sâu vào
việc khai báo và thao tác tham số.
Trong phần này, chúng ta đề cập đến:

* Truy cập các tham số để gỡ lỗi, chẩn đoán, và trực quan hóa.
* Chia sẻ tham số trên các thành phần mô hình khác nhau.


```python
import torch
from torch import nn
```


(**Chúng ta bắt đầu bằng cách tập trung vào một MLP với một lớp ẩn.**)


```python
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```


## [**Truy cập Tham số**]
<a id="subsec_param-access"></a>

Hãy bắt đầu với cách truy cập các tham số
từ các mô hình mà bạn đã biết.


Chúng ta có thể kiểm tra các tham số của lớp kết nối đầy đủ thứ hai như sau.


```python
net[2].state_dict()
```


Chúng ta có thể thấy rằng lớp kết nối đầy đủ này
chứa hai tham số,
tương ứng với trọng số và hệ số chặn
của lớp đó.


### [**Tham số Mục tiêu**]

Lưu ý rằng mỗi tham số được đại diện
như một thực thể của lớp tham số.
Để làm bất cứ điều gì hữu ích với các tham số,
trước tiên chúng ta cần truy cập các giá trị số bên dưới.
Có một số cách để làm điều này.
Một số cách đơn giản hơn trong khi các cách khác tổng quát hơn.
Code sau trích xuất hệ số chặn
từ lớp mạng nơ-ron thứ hai, trả về một thực thể lớp tham số, và
truy cập thêm giá trị của tham số đó.


```python
type(net[2].bias), net[2].bias.data
```


```python
net[2].weight.grad == None
```

### [**Tất cả Tham số cùng một lúc**]

Khi chúng ta cần thực hiện các phép toán trên tất cả tham số,
việc truy cập chúng từng cái một có thể trở nên tẻ nhạt.
Tình huống có thể trở nên đặc biệt khó xử
khi chúng ta làm việc với các mô-đun phức tạp hơn, ví dụ: lồng nhau,
vì chúng ta sẽ cần đệ quy
qua toàn bộ cây để trích xuất
các tham số của mỗi mô-đun con. Dưới đây chúng ta minh họa việc truy cập các tham số của tất cả các lớp.


```python
[(name, param.shape) for name, param in net.named_parameters()]
```


## [**Tham số Ràng buộc**]

Thường xuyên, chúng ta muốn chia sẻ tham số trên nhiều lớp.
Hãy xem cách làm điều này một cách thanh lịch.
Trong phần sau chúng ta phân bổ một lớp kết nối đầy đủ
và sau đó sử dụng các tham số của nó một cách cụ thể
để thiết lập tham số của lớp khác.
Ở đây chúng ta cần chạy lan truyền xuôi
`net(X)` trước khi truy cập các tham số.


```python
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```


Ví dụ này cho thấy rằng các tham số
của lớp thứ hai và thứ ba được ràng buộc.
Chúng không chỉ bằng nhau, chúng
được đại diện bởi cùng một tensor chính xác.
Do đó, nếu chúng ta thay đổi một trong các tham số,
cái kia cũng thay đổi theo.


## Tóm tắt

Chúng ta có một số cách truy cập và ràng buộc tham số mô hình.


## Bài tập

1. Sử dụng mô hình `NestMLP` được định nghĩa trong [sec_model_construction](#sec_model_construction) và truy cập các tham số của các lớp khác nhau.
1. Xây dựng một MLP chứa một lớp tham số chia sẻ và huấn luyện nó. Trong quá trình huấn luyện, quan sát các tham số mô hình và gradient của mỗi lớp.
1. Tại sao chia sẻ tham số là một ý tưởng tốt?


[Discussions](https://discuss.d2l.ai/t/57)

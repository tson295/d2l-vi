# Gộp
<a id="sec_pooling"></a>

Trong nhiều trường hợp tác vụ cuối cùng của chúng ta đặt ra một số câu hỏi toàn cục về hình ảnh,
ví dụ: *nó có chứa một con mèo không?* Do đó, các đơn vị của lớp cuối cùng
nên nhạy cảm với toàn bộ đầu vào.
Bằng cách dần dần tổng hợp thông tin, tạo ra các bản đồ ngày càng thô hơn,
chúng ta hoàn thành mục tiêu cuối cùng là học một biểu diễn toàn cục,
trong khi vẫn giữ tất cả các ưu điểm của các lớp tích chập ở các lớp xử lý trung gian.
Chúng ta càng đi sâu vào mạng,
trường tiếp nhận (so với đầu vào) càng lớn
mà mỗi nút ẩn nhạy cảm. Việc giảm độ phân giải không gian
đẩy nhanh quá trình này,
vì các nhân tích chập bao phủ một vùng hiệu quả lớn hơn.

Hơn nữa, khi phát hiện các đặc trưng cấp thấp hơn, chẳng hạn như cạnh
(như đã thảo luận trong [sec_conv_layer](#sec_conv_layer)),
chúng ta thường muốn biểu diễn của mình phần nào bất biến với dịch chuyển.
Ví dụ, nếu chúng ta lấy hình ảnh `X`
với sự phân định sắc nét giữa đen và trắng
và dịch chuyển toàn bộ hình ảnh sang phải một điểm ảnh,
tức là `Z[i, j] = X[i, j + 1]`,
thì đầu ra của hình ảnh mới `Z` có thể hoàn toàn khác.
Cạnh sẽ dịch chuyển một điểm ảnh.
Trong thực tế, các đối tượng hiếm khi xuất hiện chính xác ở cùng một nơi.
Trên thực tế, ngay cả với chân máy ảnh và vật thể tĩnh,
độ rung của máy ảnh do chuyển động của màn trập
có thể dịch chuyển mọi thứ một điểm ảnh hoặc nhiều hơn
(máy ảnh cao cấp được trang bị các tính năng đặc biệt để giải quyết vấn đề này).

Phần này giới thiệu *các lớp gộp*,
phục vụ mục đích kép là
giảm thiểu sự nhạy cảm của các lớp tích chập với vị trí
và lấy mẫu thấp hơn về mặt không gian các biểu diễn.


```python
from d2l import torch as d2l
import torch
from torch import nn
```


## Gộp Tối đa và Gộp Trung bình

Cũng giống như các lớp tích chập, các toán tử *gộp*
bao gồm một cửa sổ hình dạng cố định được trượt qua
tất cả các vùng trong đầu vào theo sải bước của nó,
tính toán một đầu ra duy nhất cho mỗi vị trí được duyệt qua
bởi cửa sổ hình dạng cố định (đôi khi còn được gọi là *cửa sổ gộp*).
Tuy nhiên, khác với tính toán tương quan chéo
của đầu vào và nhân trong lớp tích chập,
lớp gộp không chứa tham số (không có *nhân*).
Thay vào đó, các toán tử gộp là tất định,
thường tính giá trị tối đa hoặc trung bình
của các phần tử trong cửa sổ gộp.
Các phép toán này được gọi là *gộp tối đa* (*max-pooling* cho ngắn gọn)
và *gộp trung bình*, tương ứng.

*Gộp trung bình* về cơ bản cũ như CNN. Ý tưởng tương tự như
lấy mẫu thấp hơn của hình ảnh. Thay vì chỉ lấy giá trị của mỗi điểm ảnh thứ hai (hoặc thứ ba)
cho hình ảnh độ phân giải thấp hơn, chúng ta có thể lấy trung bình qua các điểm ảnh lân cận để thu được
một hình ảnh với tỷ lệ tín hiệu-nhiễu tốt hơn vì chúng ta đang kết hợp thông tin
từ nhiều điểm ảnh lân cận. *Gộp tối đa* được giới thiệu trong
Riesenhuber.Poggio.1999 trong bối cảnh khoa học thần kinh nhận thức để mô tả
cách thông tin tổng hợp có thể được tổng hợp theo phân cấp cho mục đích
nhận dạng đối tượng; đã có một phiên bản trước đó trong nhận dạng giọng nói [Yamaguchi.Sakamoto.Akabane.ea.1990]. Trong hầu hết mọi trường hợp, max-pooling, như nó còn được gọi,
được ưu tiên hơn gộp trung bình.

Trong cả hai trường hợp, cũng như với toán tử tương quan chéo,
chúng ta có thể nghĩ về cửa sổ gộp
như bắt đầu từ góc trên bên trái của tensor đầu vào
và trượt qua nó từ trái sang phải và từ trên xuống dưới.
Tại mỗi vị trí mà cửa sổ gộp chạm tới,
nó tính toán giá trị tối đa hoặc trung bình
của tensor con đầu vào trong cửa sổ,
tùy thuộc vào việc gộp tối đa hay gộp trung bình được sử dụng.


![Gộp tối đa với hình dạng cửa sổ gộp là $2\times 2$. Các phần tô bóng là phần tử đầu ra đầu tiên cũng như các phần tử tensor đầu vào được sử dụng để tính toán đầu ra: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
<a id="fig_pooling"></a>

Tensor đầu ra trong [fig_pooling](#fig_pooling) có chiều cao là 2 và chiều rộng là 2.
Bốn phần tử được rút ra từ giá trị tối đa trong mỗi cửa sổ gộp:

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

Nói chung, chúng ta có thể định nghĩa một lớp gộp $p \times q$ bằng cách tổng hợp qua
một vùng có kích thước đã nói. Quay lại bài toán phát hiện cạnh,
chúng ta sử dụng đầu ra của lớp tích chập
làm đầu vào cho gộp tối đa $2\times 2$.
Ký hiệu `X` là đầu vào của lớp tích chập và `Y` là đầu ra của lớp gộp.
Bất kể các giá trị của `X[i, j]`, `X[i, j + 1]`,
`X[i+1, j]` và `X[i+1, j + 1]` có khác nhau hay không,
lớp gộp luôn xuất ra `Y[i, j] = 1`.
Nghĩa là, sử dụng lớp gộp tối đa $2\times 2$,
chúng ta vẫn có thể phát hiện nếu mẫu được nhận dạng bởi lớp tích chập
di chuyển không quá một phần tử về chiều cao hoặc chiều rộng.

Trong code dưới đây, chúng ta (**triển khai lan truyền xuôi
của lớp gộp**) trong hàm `pool2d`.
Hàm này tương tự như hàm `corr2d`
trong [sec_conv_layer](#sec_conv_layer).
Tuy nhiên, không cần nhân, tính toán đầu ra
là giá trị tối đa hoặc trung bình của mỗi vùng trong đầu vào.


Chúng ta có thể xây dựng tensor đầu vào `X` trong [fig_pooling](#fig_pooling) để [**xác thực đầu ra của lớp gộp tối đa hai chiều**].

```python
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

Ngoài ra, chúng ta có thể thử nghiệm với (**lớp gộp trung bình**).

```python
pool2d(X, (2, 2), 'avg')
```

## [**Đệm và Sải bước**]

Cũng như các lớp tích chập, các lớp gộp
thay đổi hình dạng đầu ra.
Và như trước, chúng ta có thể điều chỉnh phép toán để đạt được hình dạng đầu ra mong muốn
bằng cách đệm đầu vào và điều chỉnh sải bước.
Chúng ta có thể minh họa việc sử dụng đệm và sải bước
trong các lớp gộp thông qua lớp gộp tối đa hai chiều tích hợp sẵn từ framework deep learning.
Đầu tiên chúng ta xây dựng một tensor đầu vào `X` có hình dạng bốn chiều,
trong đó số lượng mẫu (kích thước batch) và số lượng kênh đều là 1.


Vì gộp tổng hợp thông tin từ một vùng, (**framework deep learning mặc định khớp kích thước cửa sổ gộp và sải bước.**) Ví dụ, nếu chúng ta sử dụng cửa sổ gộp hình dạng `(3, 3)`
chúng ta có hình dạng sải bước là `(3, 3)` theo mặc định.


```python
pool2d = nn.MaxPool2d(3)
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```


Không cần nói, [**đệm và sải bước có thể được chỉ định thủ công**] để ghi đè các giá trị mặc định của framework nếu cần.


```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```


Tất nhiên, chúng ta có thể chỉ định một cửa sổ gộp hình chữ nhật tùy ý với chiều cao và chiều rộng tùy ý, như ví dụ dưới đây cho thấy.


```python
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```


## Nhiều Kênh

Khi xử lý dữ liệu đầu vào đa kênh,
[**lớp gộp gộp riêng từng kênh đầu vào**],
thay vì tổng hợp đầu vào qua các kênh
như trong lớp tích chập.
Điều này có nghĩa là số kênh đầu ra của lớp gộp
bằng số kênh đầu vào.
Dưới đây, chúng ta sẽ nối các tensor `X` và `X + 1`
trên chiều kênh để xây dựng đầu vào với hai kênh.


Như chúng ta có thể thấy, số kênh đầu ra vẫn là hai sau khi gộp.


```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```


## Tóm tắt

Gộp là một phép toán vô cùng đơn giản. Nó thực hiện chính xác những gì tên của nó chỉ ra, tổng hợp kết quả qua một cửa sổ các giá trị. Tất cả các ngữ nghĩa tích chập, chẳng hạn như sải bước và đệm áp dụng theo cách tương tự như trước đây. Lưu ý rằng gộp không phân biệt kênh, tức là nó để số lượng kênh không thay đổi và nó áp dụng cho mỗi kênh riêng biệt. Cuối cùng, trong hai lựa chọn gộp phổ biến, gộp tối đa được ưu tiên hơn gộp trung bình, vì nó mang lại một mức độ bất biến nhất định đối với đầu ra. Một lựa chọn phổ biến là chọn kích thước cửa sổ gộp là $2 \times 2$ để chia tư độ phân giải không gian của đầu ra.

Lưu ý rằng có nhiều cách giảm độ phân giải hơn ngoài gộp. Ví dụ, trong gộp ngẫu nhiên [Zeiler.Fergus.2013] và gộp tối đa phân số [Graham.2014] tổng hợp được kết hợp với ngẫu nhiên hóa. Điều này có thể cải thiện độ chính xác một chút trong một số trường hợp. Cuối cùng, như chúng ta sẽ thấy sau này với cơ chế attention, có những cách tinh tế hơn để tổng hợp qua các đầu ra, ví dụ: bằng cách sử dụng sự căn chỉnh giữa một truy vấn và các vectơ biểu diễn.


## Bài tập

1. Triển khai gộp trung bình thông qua một phép tích chập.
1. Chứng minh rằng gộp tối đa không thể được triển khai chỉ thông qua một phép tích chập.
1. Gộp tối đa có thể được thực hiện bằng cách sử dụng các phép toán ReLU, tức là $\textrm{ReLU}(x) = \max(0, x)$.
    1. Biểu diễn $\max (a, b)$ chỉ bằng cách sử dụng các phép toán ReLU.
    1. Sử dụng điều này để triển khai gộp tối đa bằng cách sử dụng các lớp tích chập và ReLU.
    1. Bạn cần bao nhiêu kênh và lớp cho tích chập $2 \times 2$? Bao nhiêu cho tích chập $3 \times 3$?
1. Chi phí tính toán của lớp gộp là bao nhiêu? Giả sử đầu vào của lớp gộp có kích thước $c\times h\times w$, cửa sổ gộp có hình dạng $p_\textrm{h}\times p_\textrm{w}$ với đệm $(p_\textrm{h}, p_\textrm{w})$ và sải bước $(s_\textrm{h}, s_\textrm{w})$.
1. Tại sao bạn kỳ vọng gộp tối đa và gộp trung bình hoạt động khác nhau?
1. Chúng ta có cần một lớp gộp tối thiểu riêng biệt không? Bạn có thể thay thế nó bằng một phép toán khác không?
1. Chúng ta có thể sử dụng phép toán softmax để gộp. Tại sao nó có thể không phổ biến như vậy?


[Discussions](https://discuss.d2l.ai/t/72)

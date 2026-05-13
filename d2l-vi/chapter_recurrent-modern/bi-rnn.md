# Mạng Nơ-ron Hồi Tiếp Hai Chiều
<a id="sec_bi_rnn"></a>

Cho đến nay, ví dụ làm việc của chúng ta về nhiệm vụ học chuỗi là mô hình hóa ngôn ngữ,
nơi chúng ta nhằm dự đoán token tiếp theo dựa trên tất cả các token trước đó trong chuỗi.
Trong kịch bản này, chúng ta chỉ muốn điều kiện hóa trên ngữ cảnh bên trái,
và do đó việc kết nối một chiều của RNN tiêu chuẩn có vẻ phù hợp.
Tuy nhiên, có nhiều ngữ cảnh nhiệm vụ học chuỗi khác
nơi hoàn toàn ổn khi điều kiện hóa dự đoán tại mỗi bước thời gian
trên cả ngữ cảnh bên trái lẫn bên phải.
Xét, ví dụ, phát hiện từ loại.
Tại sao chúng ta không xét ngữ cảnh theo cả hai hướng
khi đánh giá từ loại liên quan đến một từ nhất định?

Một nhiệm vụ phổ biến khác---thường hữu ích như bài tập tiền huấn luyện
trước khi tinh chỉnh mô hình trên một nhiệm vụ quan tâm thực tế---là
che giấu các token ngẫu nhiên trong một tài liệu văn bản và sau đó huấn luyện
mô hình chuỗi để dự đoán giá trị của các token bị thiếu.
Lưu ý rằng tùy thuộc vào những gì xuất hiện sau chỗ trống,
giá trị có khả năng của token bị thiếu thay đổi đáng kể:

* I am `___`.
* I am `___` hungry.
* I am `___` hungry, and I can eat half a pig.

Trong câu đầu tiên "happy" có vẻ là ứng viên có khả năng.
Các từ "not" và "very" có vẻ hợp lý trong câu thứ hai,
nhưng "not" có vẻ không tương thích với câu thứ ba.


May mắn thay, một kỹ thuật đơn giản biến đổi bất kỳ RNN một chiều nào
thành một RNN hai chiều [Schuster.Paliwal.1997].
Chúng ta đơn giản lập trình hai lớp RNN một chiều
được kết nối với nhau theo hướng ngược lại
và hoạt động trên cùng một đầu vào ([fig_birnn](#fig_birnn)).
Đối với lớp RNN đầu tiên,
đầu vào đầu tiên là $\mathbf{x}_1$
và đầu vào cuối cùng là $\mathbf{x}_T$,
nhưng đối với lớp RNN thứ hai,
đầu vào đầu tiên là $\mathbf{x}_T$
và đầu vào cuối cùng là $\mathbf{x}_1$.
Để tạo ra đầu ra của lớp RNN hai chiều này,
chúng ta đơn giản nối với nhau các đầu ra tương ứng
của hai lớp RNN một chiều cơ bản.


![Kiến trúc của RNN hai chiều.](../img/birnn.svg)
<a id="fig_birnn"></a>


Về mặt hình thức cho bất kỳ bước thời gian $t$ nào,
chúng ta xét đầu vào minibatch $\mathbf{X}_t \in \mathbb{R}^{n \times d}$
(số ví dụ $=n$; số đầu vào trong mỗi ví dụ $=d$)
và gọi hàm kích hoạt lớp ẩn là $\phi$.
Trong kiến trúc hai chiều,
các trạng thái ẩn xuôi và ngược cho bước thời gian này
lần lượt là $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$
và $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$,
trong đó $h$ là số đơn vị ẩn.
Các cập nhật trạng thái ẩn xuôi và ngược như sau:


$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{\textrm{hh}}^{(f)}  + \mathbf{b}_\textrm{h}^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{\textrm{hh}}^{(b)}  + \mathbf{b}_\textrm{h}^{(b)}),
\end{aligned}
$$

trong đó các trọng số $\mathbf{W}_{\textrm{xh}}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{\textrm{xh}}^{(b)} \in \mathbb{R}^{d \times h}, \textrm{ và } \mathbf{W}_{\textrm{hh}}^{(b)} \in \mathbb{R}^{h \times h}$, và các hệ số chặn $\mathbf{b}_\textrm{h}^{(f)} \in \mathbb{R}^{1 \times h}$ và $\mathbf{b}_\textrm{h}^{(b)} \in \mathbb{R}^{1 \times h}$ là tất cả các tham số mô hình.

Tiếp theo, chúng ta nối các trạng thái ẩn xuôi và ngược
$\overrightarrow{\mathbf{H}}_t$ và $\overleftarrow{\mathbf{H}}_t$
để thu được trạng thái ẩn $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$ để đưa vào lớp đầu ra.
Trong RNN hai chiều sâu với nhiều lớp ẩn,
thông tin như vậy được truyền dưới dạng *đầu vào* cho lớp hai chiều tiếp theo.
Cuối cùng, lớp đầu ra tính toán đầu ra
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (số đầu ra $=q$):

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

Ở đây, ma trận trọng số $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{2h \times q}$
và hệ số chặn $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$
là các tham số mô hình của lớp đầu ra.
Mặc dù về mặt kỹ thuật, hai hướng có thể có số đơn vị ẩn khác nhau,
lựa chọn thiết kế này hiếm khi được thực hiện trong thực tế.
Bây giờ chúng ta minh họa một lập trình đơn giản của RNN hai chiều.


```python
from d2l import torch as d2l
import torch
from torch import nn
```


## Lập Trình Từ Đầu

Nếu muốn lập trình RNN hai chiều từ đầu, chúng ta có thể
bao gồm hai instance `RNNScratch` một chiều
với các tham số có thể học riêng biệt.


Các trạng thái của RNN xuôi và ngược
được cập nhật riêng biệt,
trong khi các đầu ra của hai RNN này được nối lại.

```python
@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs is not None else (None, None)
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
    outputs = [d2l.concat((f, b), -1) for f, b in zip(
        f_outputs, reversed(b_outputs))]
    return outputs, (f_H, b_H)
```

## Lập Trình Súc Tích


## Tóm Tắt

Trong RNN hai chiều, trạng thái ẩn cho mỗi bước thời gian được xác định đồng thời bởi dữ liệu trước và sau bước thời gian hiện tại. RNN hai chiều hầu hết hữu ích cho việc mã hóa chuỗi và ước tính quan sát dựa trên ngữ cảnh hai chiều. RNN hai chiều rất tốn kém để huấn luyện do chuỗi gradient dài.

## Bài Tập

1. Nếu các hướng khác nhau sử dụng số đơn vị ẩn khác nhau, hình dạng của $\mathbf{H}_t$ sẽ thay đổi như thế nào?
1. Thiết kế RNN hai chiều với nhiều lớp ẩn.
1. Đa nghĩa rất phổ biến trong ngôn ngữ tự nhiên. Ví dụ, từ "bank" có các nghĩa khác nhau trong ngữ cảnh "i went to the bank to deposit cash" và "i went to the bank to sit down". Làm thế nào chúng ta có thể thiết kế mô hình mạng nơ-ron sao cho khi được đưa một chuỗi ngữ cảnh và một từ, biểu diễn vector của từ trong ngữ cảnh đúng sẽ được trả về? Loại kiến trúc mạng nơ-ron nào được ưu tiên để xử lý đa nghĩa?


[Thảo luận](https://discuss.d2l.ai/t/1059)

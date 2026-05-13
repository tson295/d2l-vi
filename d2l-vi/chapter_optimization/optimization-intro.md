# Tối Ưu Hóa và Deep Learning
<a id="sec_optimization-intro"></a>

Trong phần này, chúng ta sẽ thảo luận về mối quan hệ giữa tối ưu hóa và deep learning cũng như những thách thức khi sử dụng tối ưu hóa trong deep learning.
Với một bài toán deep learning, chúng ta thường sẽ định nghĩa một *hàm mất mát* trước. Khi đã có hàm mất mát, chúng ta có thể sử dụng một thuật toán tối ưu hóa để cố gắng tối thiểu hóa mất mát.
Trong tối ưu hóa, hàm mất mát thường được gọi là *hàm mục tiêu* của bài toán tối ưu hóa. Theo truyền thống và quy ước, hầu hết các thuật toán tối ưu hóa đều liên quan đến *tối thiểu hóa*. Nếu chúng ta cần tối đa hóa một mục tiêu, có một giải pháp đơn giản: chỉ cần đổi dấu mục tiêu.

## Mục Tiêu của Tối Ưu Hóa

Mặc dù tối ưu hóa cung cấp một cách để tối thiểu hóa hàm mất mát cho
deep learning, về bản chất, các mục tiêu của tối ưu hóa và deep learning là
khác nhau về cơ bản.
Cái trước chủ yếu quan tâm đến việc tối thiểu hóa một
mục tiêu trong khi cái sau quan tâm đến việc tìm một mô hình phù hợp, với một
lượng dữ liệu hữu hạn.
Trong [sec_generalization_basics](#sec_generalization_basics),
chúng ta đã thảo luận chi tiết về sự khác biệt giữa hai mục tiêu này.
Ví dụ,
lỗi huấn luyện và lỗi tổng quát hóa thường khác nhau: vì hàm mục tiêu
của thuật toán tối ưu hóa thường là hàm mất mát dựa trên
tập dữ liệu huấn luyện, mục tiêu của tối ưu hóa là giảm lỗi huấn luyện.
Tuy nhiên, mục tiêu của deep learning (hoặc rộng hơn là suy luận thống kê) là
giảm lỗi tổng quát hóa.
Để đạt được điều sau, chúng ta cần chú ý
đến quá khớp ngoài việc sử dụng thuật toán tối ưu hóa để
giảm lỗi huấn luyện.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```python
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

Để minh họa các mục tiêu khác nhau đã đề cập,
hãy xem xét
rủi ro thực nghiệm và rủi ro.
Như được mô tả
trong [subsec_empirical-risk-and-risk](#subsec_empirical-risk-and-risk),
rủi ro thực nghiệm
là mất mát trung bình
trên tập dữ liệu huấn luyện
trong khi rủi ro là mất mát kỳ vọng
trên toàn bộ dân số dữ liệu.
Dưới đây chúng ta định nghĩa hai hàm:
hàm rủi ro `f`
và hàm rủi ro thực nghiệm `g`.
Giả sử rằng chúng ta chỉ có một lượng hữu hạn dữ liệu huấn luyện.
Kết quả là, ở đây `g` ít mượt mà hơn `f`.

```python
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

Đồ thị dưới đây minh họa rằng cực tiểu của rủi ro thực nghiệm trên một tập dữ liệu huấn luyện có thể ở một vị trí khác so với cực tiểu của rủi ro (lỗi tổng quát hóa).

```python
#@tab all
def annotate(text, xy, xytext):  
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## Những Thách Thức Tối Ưu Hóa trong Deep Learning

Trong chương này, chúng ta sẽ tập trung cụ thể vào hiệu suất của các thuật toán tối ưu hóa trong việc tối thiểu hóa hàm mục tiêu, thay vì
lỗi tổng quát hóa của mô hình.
Trong [sec_linear_regression](#sec_linear_regression)
chúng ta đã phân biệt giữa các giải pháp phân tích và giải pháp số trong
các bài toán tối ưu hóa.
Trong deep learning, hầu hết các hàm mục tiêu đều
phức tạp và không có giải pháp phân tích. Thay vào đó, chúng ta phải sử dụng các
thuật toán tối ưu hóa số.
Các thuật toán tối ưu hóa trong chương này
đều thuộc danh mục này.

Có nhiều thách thức trong tối ưu hóa deep learning. Một số thách thức phức tạp nhất là cực tiểu cục bộ, điểm yên ngựa và gradient biến mất.
Hãy xem xét chúng.


### Cực Tiểu Cục Bộ

Với bất kỳ hàm mục tiêu $f(x)$ nào,
nếu giá trị của $f(x)$ tại $x$ nhỏ hơn các giá trị của $f(x)$ tại bất kỳ điểm nào khác trong lân cận của $x$, thì $f(x)$ có thể là cực tiểu cục bộ.
Nếu giá trị của $f(x)$ tại $x$ là cực tiểu của hàm mục tiêu trên toàn bộ miền xác định,
thì $f(x)$ là cực tiểu toàn cục.

Ví dụ, cho hàm

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ với } -1.0 \leq x \leq 2.0,$$

chúng ta có thể xấp xỉ cực tiểu cục bộ và cực tiểu toàn cục của hàm này.

```python
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

Hàm mục tiêu của các mô hình deep learning thường có nhiều cực trị cục bộ.
Khi nghiệm số của một bài toán tối ưu hóa gần cực tiểu cục bộ, nghiệm số thu được bởi lần lặp cuối cùng có thể chỉ tối thiểu hóa hàm mục tiêu *cục bộ*, thay vì *toàn cục*, khi gradient của các nghiệm hàm mục tiêu tiếp cận hoặc trở thành không.
Chỉ một mức độ nhiễu nào đó mới có thể đẩy tham số ra khỏi cực tiểu cục bộ. Thực tế, đây là một trong những thuộc tính có lợi của
gradient giảm ngẫu nhiên minibatch trong đó sự biến đổi tự nhiên của gradient trên các minibatch có khả năng dịch chuyển các tham số khỏi cực tiểu cục bộ.


### Điểm Yên Ngựa

Ngoài cực tiểu cục bộ, điểm yên ngựa là lý do khác để gradient biến mất. *Điểm yên ngựa* là bất kỳ vị trí nào mà tất cả gradient của hàm biến mất nhưng không phải là cực tiểu toàn cục hay cực tiểu cục bộ.
Xét hàm $f(x) = x^3$. Đạo hàm bậc nhất và bậc hai của nó biến mất với $x=0$. Tối ưu hóa có thể bị đình trệ tại điểm này, mặc dù nó không phải là cực tiểu.

```python
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

Các điểm yên ngựa trong các chiều cao hơn còn nguy hiểm hơn, như ví dụ dưới đây cho thấy. Xét hàm $f(x, y) = x^2 - y^2$. Điểm yên ngựa của nó tại $(0, 0)$. Đây là cực đại theo $y$ và cực tiểu theo $x$. Hơn nữa, nó *trông* như một cái yên ngựa, đó là lý do tên gọi toán học này.

```python
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```python
#@tab pytorch, tensorflow
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

Chúng ta giả sử rằng đầu vào của một hàm là một vector $k$ chiều và
đầu ra của nó là một số vô hướng, vì vậy ma trận Hessian của nó sẽ có $k$ giá trị riêng.
Nghiệm của
hàm có thể là cực tiểu cục bộ, cực đại cục bộ, hoặc điểm yên ngựa tại
một vị trí mà gradient hàm bằng không:

* Khi các giá trị riêng của ma trận Hessian của hàm tại vị trí gradient bằng không đều dương, chúng ta có cực tiểu cục bộ cho hàm.
* Khi các giá trị riêng của ma trận Hessian của hàm tại vị trí gradient bằng không đều âm, chúng ta có cực đại cục bộ cho hàm.
* Khi các giá trị riêng của ma trận Hessian của hàm tại vị trí gradient bằng không vừa âm vừa dương, chúng ta có điểm yên ngựa cho hàm.

Đối với các bài toán nhiều chiều, khả năng ít nhất *một số* giá trị riêng là âm khá cao. Điều này làm cho điểm yên ngựa có nhiều khả năng hơn cực tiểu cục bộ. Chúng ta sẽ thảo luận một số ngoại lệ cho tình huống này trong phần tiếp theo khi giới thiệu tính lồi. Nói tóm lại, các hàm lồi là những hàm mà các giá trị riêng của Hessian không bao giờ âm. Đáng tiếc, hầu hết các bài toán deep learning không thuộc danh mục này. Tuy nhiên, đây là công cụ tuyệt vời để nghiên cứu các thuật toán tối ưu hóa.

### Gradient Biến Mất

Có lẽ vấn đề phức tạp nhất gặp phải là gradient biến mất.
Hãy nhớ lại các hàm kích hoạt thường dùng của chúng ta và đạo hàm của chúng trong [subsec_activation-functions](#subsec_activation-functions).
Ví dụ, giả sử rằng chúng ta muốn tối thiểu hóa hàm $f(x) = \tanh(x)$ và chúng ta bắt đầu tại $x = 4$. Như chúng ta có thể thấy, gradient của $f$ gần bằng không.
Cụ thể hơn, $f'(x) = 1 - \tanh^2(x)$ và do đó $f'(4) = 0.0013$.
Kết quả là, tối ưu hóa sẽ bị đình trệ trong một thời gian dài trước khi chúng ta tiến bộ. Hóa ra đây là một trong những lý do tại sao việc huấn luyện các mô hình deep learning khá phức tạp trước khi giới thiệu hàm kích hoạt ReLU.

```python
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

Như chúng ta đã thấy, tối ưu hóa cho deep learning đầy những thách thức. May mắn thay, tồn tại một loạt các thuật toán mạnh mẽ hoạt động tốt và dễ sử dụng ngay cả đối với người mới bắt đầu. Hơn nữa, thực sự không cần tìm *nghiệm* tốt nhất. Các cực trị cục bộ hoặc thậm chí là các xấp xỉ của chúng vẫn rất hữu ích.

## Tóm Tắt

* Tối thiểu hóa lỗi huấn luyện *không* đảm bảo rằng chúng ta tìm được tập tham số tốt nhất để tối thiểu hóa lỗi tổng quát hóa.
* Các bài toán tối ưu hóa có thể có nhiều cực tiểu cục bộ.
* Bài toán có thể có nhiều điểm yên ngựa hơn, vì thường các bài toán không lồi.
* Gradient biến mất có thể làm tối ưu hóa bị đình trệ. Thường thì việc tái tham số hóa bài toán sẽ giúp ích. Khởi tạo tốt các tham số cũng có thể có lợi.


## Bài Tập

1. Xem xét một MLP đơn giản với một lớp ẩn đơn có, giả sử, $d$ chiều trong lớp ẩn và một đầu ra đơn. Chỉ ra rằng với bất kỳ cực tiểu cục bộ nào, có ít nhất $d!$ nghiệm tương đương hoạt động giống hệt nhau.
1. Giả sử rằng chúng ta có một ma trận ngẫu nhiên đối xứng $\mathbf{M}$ trong đó các mục
   $M_{ij} = M_{ji}$ đều được rút từ một phân phối xác suất nào đó
   $p_{ij}$. Hơn nữa giả sử rằng $p_{ij}(x) = p_{ij}(-x)$, tức là,
   phân phối là đối xứng (xem ví dụ: Wigner.1958 để biết chi tiết).
    1. Chứng minh rằng phân phối trên các giá trị riêng cũng đối xứng. Tức là, với bất kỳ vector riêng $\mathbf{v}$ nào, xác suất giá trị riêng liên quan $\lambda$ thỏa mãn $P(\lambda > 0) = P(\lambda < 0)$.
    1. Tại sao điều trên *không* ngụ ý $P(\lambda > 0) = 0.5$?
1. Bạn có thể nghĩ đến những thách thức nào khác liên quan đến tối ưu hóa deep learning?
1. Giả sử rằng bạn muốn cân bằng một quả bóng (thực sự) trên một cái yên ngựa (thực sự).
    1. Tại sao điều này khó?
    1. Bạn có thể khai thác hiệu ứng này cho các thuật toán tối ưu hóa không?


[Discussions](https://discuss.d2l.ai/t/487)

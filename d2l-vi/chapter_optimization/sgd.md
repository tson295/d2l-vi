# Stochastic Gradient Descent
<a id="sec_sgd"></a>

Trong các chương trước, chúng ta liên tục dùng stochastic gradient descent trong quy trình huấn luyện, tuy nhiên chưa giải thích vì sao nó hoạt động.
Để làm rõ điều đó,
chúng ta vừa mô tả các nguyên lý cơ bản của gradient descent
trong [sec_gd](#sec_gd).
Trong phần này, chúng ta tiếp tục thảo luận
*stochastic gradient descent* chi tiết hơn.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```python
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## Cập Nhật Gradient Ngẫu Nhiên

Trong deep learning, hàm mục tiêu thường là trung bình của các hàm mất mát cho từng ví dụ trong tập huấn luyện.
Cho một tập huấn luyện gồm $n$ ví dụ,
chúng ta giả sử $f_i(\mathbf{x})$ là hàm mất mát
ứng với ví dụ huấn luyện có chỉ số $i$,
trong đó $\mathbf{x}$ là vector tham số.
Khi đó chúng ta thu được hàm mục tiêu

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

Gradient của hàm mục tiêu tại $\mathbf{x}$ được tính là

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

Nếu dùng gradient descent, chi phí tính toán cho mỗi vòng lặp của biến độc lập là $\mathcal{O}(n)$, tăng tuyến tính theo $n$. Do đó, khi tập huấn luyện lớn hơn, chi phí của gradient descent cho mỗi vòng lặp sẽ cao hơn.

Stochastic gradient descent (SGD) giảm chi phí tính toán ở mỗi vòng lặp. Tại mỗi vòng lặp của stochastic gradient descent, chúng ta lấy mẫu đều ngẫu nhiên một chỉ số $i\in\{1,\ldots, n\}$ cho các ví dụ dữ liệu, rồi tính gradient $\nabla f_i(\mathbf{x})$ để cập nhật $\mathbf{x}$:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

trong đó $\eta$ là tốc độ học. Chúng ta có thể thấy rằng chi phí tính toán cho mỗi vòng lặp giảm từ $\mathcal{O}(n)$ của gradient descent xuống hằng số $\mathcal{O}(1)$. Hơn nữa, chúng ta muốn nhấn mạnh rằng gradient ngẫu nhiên $\nabla f_i(\mathbf{x})$ là một ước lượng không chệch của gradient đầy đủ $\nabla f(\mathbf{x})$ vì

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

Điều này có nghĩa là, xét trung bình, gradient ngẫu nhiên là một ước lượng tốt của gradient.

Bây giờ, chúng ta sẽ so sánh nó với gradient descent bằng cách thêm nhiễu ngẫu nhiên có trung bình 0 và phương sai 1 vào gradient để mô phỏng stochastic gradient descent.

```python
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```python
#@tab mxnet
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```python
#@tab pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```python
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```python
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Như chúng ta có thể thấy, quỹ đạo của các biến trong stochastic gradient descent nhiễu hơn nhiều so với quỹ đạo đã quan sát trong gradient descent ở [sec_gd](#sec_gd). Điều này là do bản chất ngẫu nhiên của gradient. Tức là, ngay cả khi đã đến gần cực tiểu, chúng ta vẫn chịu sự bất định được đưa vào bởi gradient tức thời thông qua $\eta \nabla f_i(\mathbf{x})$. Ngay cả sau 50 bước, chất lượng vẫn chưa tốt lắm. Tệ hơn, nó sẽ không cải thiện sau các bước bổ sung (chúng tôi khuyến khích bạn thử nghiệm với số bước lớn hơn để xác nhận điều này). Điều này khiến chúng ta chỉ còn một lựa chọn: thay đổi tốc độ học $\eta$. Tuy nhiên, nếu chọn nó quá nhỏ, ban đầu chúng ta sẽ không có tiến triển đáng kể nào. Mặt khác, nếu chọn quá lớn, chúng ta sẽ không nhận được nghiệm tốt, như đã thấy ở trên. Cách duy nhất để giải quyết các mục tiêu mâu thuẫn này là giảm tốc độ học một cách *động* khi quá trình tối ưu hóa diễn ra.

Đây cũng là lý do thêm hàm tốc độ học `lr` vào hàm bước `sgd`. Trong ví dụ trên, mọi chức năng lập lịch tốc độ học đều ở trạng thái không hoạt động vì chúng ta đặt hàm `lr` liên quan là hằng số.

## Tốc Độ Học Động

Thay $\eta$ bằng một tốc độ học phụ thuộc thời gian $\eta(t)$ làm tăng độ phức tạp của việc kiểm soát hội tụ của một thuật toán tối ưu hóa. Cụ thể, chúng ta cần xác định $\eta$ nên suy giảm nhanh đến mức nào. Nếu nó quá nhanh, chúng ta sẽ dừng tối ưu hóa quá sớm. Nếu giảm nó quá chậm, chúng ta lãng phí quá nhiều thời gian cho tối ưu hóa. Sau đây là một vài chiến lược cơ bản được dùng để điều chỉnh $\eta$ theo thời gian (chúng ta sẽ thảo luận các chiến lược nâng cao hơn sau):

$$
\begin{aligned}
    \eta(t) & = \eta_i \textrm{ if } t_i \leq t \leq t_{i+1}  && \textrm{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \textrm{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \textrm{polynomial decay}
\end{aligned}
$$

Trong kịch bản *hằng từng đoạn* đầu tiên, chúng ta giảm tốc độ học, ví dụ mỗi khi tiến triển trong tối ưu hóa bị đình trệ. Đây là một chiến lược phổ biến để huấn luyện các mạng sâu. Ngoài ra, chúng ta có thể giảm nó mạnh hơn nhiều bằng *suy giảm hàm mũ*. Đáng tiếc là điều này thường dẫn đến dừng sớm trước khi thuật toán hội tụ. Một lựa chọn phổ biến là *suy giảm đa thức* với $\alpha = 0.5$. Trong trường hợp tối ưu hóa lồi, có một số chứng minh cho thấy tốc độ này có hành vi tốt.

Hãy xem suy giảm hàm mũ trông như thế nào trong thực tế.

```python
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

Đúng như kỳ vọng, phương sai trong các tham số giảm đáng kể. Tuy nhiên, điều này phải trả giá bằng việc không hội tụ đến nghiệm tối ưu $\mathbf{x} = (0, 0)$. Ngay cả sau 1000 bước lặp, chúng ta vẫn còn rất xa nghiệm tối ưu. Thật vậy, thuật toán hoàn toàn không hội tụ. Mặt khác, nếu dùng suy giảm đa thức trong đó tốc độ học suy giảm theo nghịch đảo căn bậc hai của số bước, hội tụ trở nên tốt hơn chỉ sau 50 bước.

```python
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Còn tồn tại nhiều lựa chọn khác để đặt tốc độ học. Chẳng hạn, chúng ta có thể bắt đầu với tốc độ nhỏ, sau đó tăng nhanh rồi lại giảm, dù giảm chậm hơn. Chúng ta thậm chí có thể luân phiên giữa các tốc độ học nhỏ hơn và lớn hơn. Có rất nhiều lịch như vậy. Hiện tại, hãy tập trung vào các lịch tốc độ học mà phân tích lý thuyết toàn diện là khả thi, tức là vào tốc độ học trong bối cảnh lồi. Với các bài toán không lồi tổng quát, rất khó thu được các đảm bảo hội tụ có ý nghĩa, vì nói chung việc tối thiểu hóa các bài toán phi tuyến không lồi là NP-hard. Để khảo sát, xem ví dụ [ghi chú bài giảng](https://www.stat.cmu.edu/%7Eryantibs/convexopt-F15/lectures/26-nonconvex.pdf) rất hay của Tibshirani 2015.


## Phân Tích Hội Tụ Cho Mục Tiêu Lồi

Phân tích hội tụ sau đây của stochastic gradient descent cho các hàm mục tiêu lồi
là tùy chọn và chủ yếu nhằm truyền đạt thêm trực giác về bài toán.
Chúng ta giới hạn ở một trong những chứng minh đơn giản nhất [Nesterov.Vial.2000].
Tồn tại các kỹ thuật chứng minh nâng cao hơn đáng kể, ví dụ khi hàm mục tiêu có hành vi đặc biệt tốt.


Giả sử hàm mục tiêu $f(\boldsymbol{\xi}, \mathbf{x})$ lồi theo $\mathbf{x}$
với mọi $\boldsymbol{\xi}$.
Cụ thể hơn,
chúng ta xét cập nhật stochastic gradient descent:

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

trong đó $f(\boldsymbol{\xi}_t, \mathbf{x})$
là hàm mục tiêu
ứng với ví dụ huấn luyện $\boldsymbol{\xi}_t$
được rút ra từ một phân phối nào đó
tại bước $t$ và $\mathbf{x}$ là tham số mô hình.
Ký hiệu

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

là rủi ro kỳ vọng và $R^*$ là giá trị nhỏ nhất của nó theo $\mathbf{x}$. Cuối cùng, đặt $\mathbf{x}^*$ là bộ tối thiểu hóa (chúng ta giả định rằng nó tồn tại trong miền mà $\mathbf{x}$ được định nghĩa). Trong trường hợp này, chúng ta có thể theo dõi khoảng cách giữa tham số hiện tại $\mathbf{x}_t$ tại thời điểm $t$ và bộ tối thiểu hóa rủi ro $\mathbf{x}^*$, rồi xem liệu nó có cải thiện theo thời gian hay không:

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$

Chúng ta giả định rằng chuẩn $\ell_2$ của gradient ngẫu nhiên $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ bị chặn bởi một hằng số $L$ nào đó, do đó chúng ta có

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$


Chúng ta chủ yếu quan tâm đến cách khoảng cách giữa $\mathbf{x}_t$ và $\mathbf{x}^*$ thay đổi *theo kỳ vọng*. Thực ra, với bất kỳ chuỗi bước cụ thể nào, khoảng cách hoàn toàn có thể tăng, tùy vào $\boldsymbol{\xi}_t$ mà chúng ta gặp. Vì vậy, chúng ta cần chặn tích vô hướng.
Vì với bất kỳ hàm lồi $f$ nào ta có
$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$
với mọi $\mathbf{x}$ và $\mathbf{y}$,
theo tính lồi chúng ta có

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$

Thay cả hai bất đẳng thức :eqref:`eq_sgd-L` và :eqref:`eq_sgd-f-xi-xstar` vào :eqref:`eq_sgd-xt+1-xstar`, chúng ta thu được một chặn cho khoảng cách giữa các tham số tại thời điểm $t+1$ như sau:

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$

Điều này có nghĩa là chúng ta có tiến triển miễn là chênh lệch giữa mất mát hiện tại và mất mát tối ưu lớn hơn $\eta_t L^2/2$. Vì chênh lệch này chắc chắn hội tụ về 0, suy ra tốc độ học $\eta_t$ cũng cần phải *triệt tiêu*.

Tiếp theo, chúng ta lấy kỳ vọng trên :eqref:`eqref_sgd-xt-diff`. Điều này cho ta

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

Bước cuối cùng bao gồm lấy tổng các bất đẳng thức với $t \in \{1, \ldots, T\}$. Vì tổng là dạng kính thiên văn và bỏ hạng thấp hơn, chúng ta thu được

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$

Lưu ý rằng chúng ta đã tận dụng việc $\mathbf{x}_1$ là đã cho, nên có thể bỏ kỳ vọng. Cuối cùng định nghĩa

$$\bar{\mathbf{x}} \stackrel{\textrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

Vì

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

theo bất đẳng thức Jensen (đặt $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ trong :eqref:`eq_jensens-inequality`) và tính lồi của $R$, suy ra $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$, do đó

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

Thay điều này vào bất đẳng thức :eqref:`eq_sgd-x1-xstar` cho ta chặn

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

trong đó $r^2 \stackrel{\textrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$ là một chặn cho khoảng cách giữa lựa chọn tham số ban đầu và kết quả cuối cùng. Tóm lại, tốc độ hội tụ phụ thuộc vào cách
chuẩn của gradient ngẫu nhiên được chặn ($L$) và giá trị tham số ban đầu cách nghiệm tối ưu bao xa ($r$). Lưu ý rằng chặn này được viết theo $\bar{\mathbf{x}}$ thay vì $\mathbf{x}_T$. Điều này xảy ra vì $\bar{\mathbf{x}}$ là một phiên bản được làm trơn của đường đi tối ưu hóa.
Bất cứ khi nào biết $r, L$ và $T$, chúng ta có thể chọn tốc độ học $\eta = r/(L \sqrt{T})$. Điều này cho chặn trên $rL/\sqrt{T}$. Tức là, chúng ta hội tụ với tốc độ $\mathcal{O}(1/\sqrt{T})$ đến nghiệm tối ưu.


## Gradient Ngẫu Nhiên và Mẫu Hữu Hạn

Cho đến nay, chúng ta đã hơi thiếu chặt chẽ khi nói về stochastic gradient descent. Chúng ta giả định rằng mình rút các mẫu $x_i$, thường cùng với nhãn $y_i$, từ một phân phối nào đó $p(x, y)$ và dùng điều này để cập nhật tham số mô hình theo một cách nào đó. Cụ thể, với kích thước mẫu hữu hạn, chúng ta chỉ lập luận rằng phân phối rời rạc $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$
với một số hàm $\delta_{x_i}$ và $\delta_{y_i}$
cho phép chúng ta thực hiện stochastic gradient descent trên nó.

Tuy nhiên, đây không thực sự là điều chúng ta đã làm. Trong các ví dụ đồ chơi ở phần hiện tại, chúng ta chỉ đơn giản thêm nhiễu vào một gradient vốn không ngẫu nhiên, tức là chúng ta giả vờ có các cặp $(x_i, y_i)$. Hóa ra điều này là hợp lý ở đây (xem phần bài tập để thảo luận chi tiết). Điều đáng bận tâm hơn là trong tất cả các thảo luận trước đó, rõ ràng chúng ta đã không làm như vậy. Thay vào đó, chúng ta lặp qua tất cả các mẫu *đúng một lần*. Để thấy vì sao điều này là đáng ưu tiên, hãy xét trường hợp ngược lại, tức là chúng ta lấy mẫu $n$ quan sát từ phân phối rời rạc *có hoàn lại*. Xác suất chọn ngẫu nhiên một phần tử $i$ là $1/n$. Do đó, xác suất chọn nó *ít nhất* một lần là

$$P(\textrm{choose~} i) = 1 - P(\textrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

Lập luận tương tự cho thấy xác suất chọn một mẫu nào đó (tức là ví dụ huấn luyện) *đúng một lần* được cho bởi

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

Lấy mẫu có hoàn lại dẫn đến phương sai tăng và hiệu quả dữ liệu giảm so với lấy mẫu *không hoàn lại*. Vì vậy, trong thực tế chúng ta thực hiện cách sau (và đây là lựa chọn mặc định xuyên suốt cuốn sách này). Cuối cùng, lưu ý rằng các lượt duyệt lặp lại qua tập huấn luyện sẽ duyệt nó theo một thứ tự ngẫu nhiên *khác*.


## Tóm Tắt

* Với các bài toán lồi, chúng ta có thể chứng minh rằng với nhiều lựa chọn tốc độ học, stochastic gradient descent sẽ hội tụ đến nghiệm tối ưu.
* Với deep learning, nói chung điều này không đúng. Tuy nhiên, phân tích các bài toán lồi cho chúng ta hiểu biết hữu ích về cách tiếp cận tối ưu hóa, cụ thể là giảm tốc độ học dần dần, nhưng không quá nhanh.
* Vấn đề xảy ra khi tốc độ học quá nhỏ hoặc quá lớn. Trong thực tế, tốc độ học phù hợp thường chỉ được tìm thấy sau nhiều thử nghiệm.
* Khi có nhiều ví dụ hơn trong tập huấn luyện, việc tính mỗi vòng lặp cho gradient descent tốn kém hơn, vì vậy stochastic gradient descent được ưu tiên trong các trường hợp này.
* Các đảm bảo tối ưu cho stochastic gradient descent nói chung không có trong các trường hợp không lồi, vì số cực tiểu cục bộ cần kiểm tra rất có thể tăng theo hàm mũ.


## Bài Tập

1. Thử nghiệm với các lịch tốc độ học khác nhau cho stochastic gradient descent và với các số vòng lặp khác nhau. Cụ thể, hãy vẽ khoảng cách đến nghiệm tối ưu $(0, 0)$ theo số vòng lặp.
1. Chứng minh rằng với hàm $f(x_1, x_2) = x_1^2 + 2 x_2^2$, việc thêm nhiễu chuẩn vào gradient tương đương với tối thiểu hóa một hàm mất mát $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ trong đó $\mathbf{x}$ được rút ra từ một phân phối chuẩn.
1. So sánh sự hội tụ của stochastic gradient descent khi bạn lấy mẫu từ $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ có hoàn lại và khi lấy mẫu không hoàn lại.
1. Bạn sẽ thay đổi bộ giải stochastic gradient descent như thế nào nếu một gradient nào đó (hay đúng hơn là một tọa độ nào đó gắn với nó) luôn lớn hơn tất cả các gradient khác?
1. Giả sử $f(x) = x^2 (1 + \sin x)$. $f$ có bao nhiêu cực tiểu cục bộ? Bạn có thể thay đổi $f$ theo cách khiến việc tối thiểu hóa nó cần phải đánh giá tất cả các cực tiểu cục bộ không?


[Discussions](https://discuss.d2l.ai/t/497)

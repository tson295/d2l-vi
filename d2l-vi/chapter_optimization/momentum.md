# Momentum
<a id="sec_momentum"></a>

Trong [sec_sgd](#sec_sgd), chúng ta đã xem xét điều gì xảy ra khi thực hiện stochastic gradient descent, tức là khi thực hiện tối ưu hóa trong đó chỉ có một biến thể nhiễu của gradient là khả dụng. Cụ thể, chúng ta nhận thấy rằng với các gradient nhiễu, cần đặc biệt thận trọng khi chọn tốc độ học trước sự hiện diện của nhiễu. Nếu giảm nó quá nhanh, hội tụ sẽ đình trệ. Nếu quá dễ dãi, chúng ta không hội tụ đến một nghiệm đủ tốt vì nhiễu tiếp tục đẩy chúng ta ra xa nghiệm tối ưu.

## Cơ Bản

Trong phần này, chúng ta sẽ khám phá các thuật toán tối ưu hóa hiệu quả hơn, đặc biệt cho một số loại bài toán tối ưu hóa thường gặp trong thực tế.


### Trung Bình Rò Rỉ

Phần trước đã thảo luận minibatch SGD như một cách để tăng tốc tính toán. Nó cũng có tác dụng phụ tốt là việc lấy trung bình các gradient làm giảm lượng phương sai. Minibatch stochastic gradient descent có thể được tính bằng:

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

Để giữ ký hiệu đơn giản, ở đây chúng ta dùng $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ làm stochastic gradient descent cho mẫu $i$ với các trọng số đã cập nhật tại thời điểm $t-1$.
Sẽ rất tốt nếu chúng ta có thể hưởng lợi từ hiệu ứng giảm phương sai thậm chí vượt ra ngoài việc lấy trung bình gradient trên một minibatch. Một lựa chọn để đạt được điều này là thay phép tính gradient bằng một "trung bình rò rỉ":

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

với một $\beta \in (0, 1)$ nào đó. Điều này về thực chất thay gradient tức thời bằng một gradient đã được lấy trung bình trên nhiều gradient *trước đó*. $\mathbf{v}$ được gọi là *vận tốc*. Nó tích lũy các gradient trước tương tự như cách một quả bóng nặng lăn xuống bề mặt của hàm mục tiêu tích hợp các lực trong quá khứ. Để xem điều gì đang xảy ra chi tiết hơn, hãy khai triển đệ quy $\mathbf{v}_t$ thành

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

$\beta$ lớn tương ứng với một trung bình dài hạn, trong khi $\beta$ nhỏ chỉ tương ứng với một hiệu chỉnh nhẹ so với phương pháp gradient. Đại lượng thay thế gradient mới không còn trỏ theo hướng giảm dốc nhất trên một mẫu cụ thể nữa, mà theo hướng của trung bình có trọng số của các gradient trước đó. Điều này cho phép chúng ta thu được phần lớn lợi ích của việc lấy trung bình trên một batch mà không phải chịu chi phí thực sự tính các gradient trên batch đó. Chúng ta sẽ quay lại quy trình lấy trung bình này chi tiết hơn sau.

Lập luận trên tạo nền tảng cho những gì hiện được gọi là các phương pháp gradient *tăng tốc*, chẳng hạn như gradient với momentum. Chúng có thêm lợi ích là hiệu quả hơn nhiều trong các trường hợp bài toán tối ưu hóa có điều kiện kém (tức là có một số hướng mà tiến triển chậm hơn nhiều so với các hướng khác, giống như một hẻm núi hẹp). Hơn nữa, chúng cho phép chúng ta lấy trung bình các gradient liên tiếp để thu được các hướng giảm ổn định hơn. Thật vậy, khía cạnh tăng tốc ngay cả với các bài toán lồi không nhiễu là một trong những lý do chính khiến momentum hoạt động và hoạt động rất tốt.

Như có thể kỳ vọng, nhờ hiệu quả của nó, momentum là một chủ đề được nghiên cứu kỹ trong tối ưu hóa cho deep learning và xa hơn nữa. Xem ví dụ [bài viết giải thích](https://distill.pub/2017/momentum/) rất đẹp của Goh.2017 để có phân tích sâu và hoạt ảnh tương tác. Nó được đề xuất bởi Polyak.1964. Nesterov.2018 có thảo luận lý thuyết chi tiết trong bối cảnh tối ưu hóa lồi. Momentum trong deep learning đã được biết là có lợi từ lâu. Xem ví dụ thảo luận của Sutskever.Martens.Dahl.ea.2013 để biết chi tiết.

### Một Bài Toán Có Điều Kiện Kém

Để hiểu rõ hơn các tính chất hình học của phương pháp momentum, chúng ta quay lại gradient descent, dù với một hàm mục tiêu kém dễ chịu hơn đáng kể. Nhớ rằng trong [sec_gd](#sec_gd), chúng ta đã dùng $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, tức là một mục tiêu ellipsoid bị méo vừa phải. Chúng ta làm méo hàm này hơn nữa bằng cách kéo giãn nó theo hướng $x_1$ thông qua

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Như trước, $f$ có cực tiểu tại $(0, 0)$. Hàm này *rất* phẳng theo hướng $x_1$. Hãy xem điều gì xảy ra khi chúng ta thực hiện gradient descent như trước trên hàm mới này. Chúng ta chọn tốc độ học là $0.4$.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```python
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

Theo cách xây dựng, gradient theo hướng $x_2$ *lớn hơn nhiều* và thay đổi nhanh hơn nhiều so với hướng ngang $x_1$. Do đó, chúng ta bị kẹt giữa hai lựa chọn không mong muốn: nếu chọn tốc độ học nhỏ, chúng ta đảm bảo nghiệm không phân kỳ theo hướng $x_2$ nhưng phải chịu hội tụ chậm theo hướng $x_1$. Ngược lại, với tốc độ học lớn, chúng ta tiến triển nhanh theo hướng $x_1$ nhưng phân kỳ theo $x_2$. Ví dụ dưới đây minh họa điều gì xảy ra ngay cả sau một mức tăng nhẹ tốc độ học từ $0.4$ lên $0.6$. Hội tụ theo hướng $x_1$ cải thiện nhưng chất lượng nghiệm tổng thể tệ hơn nhiều.

```python
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### Phương Pháp Momentum

Phương pháp momentum cho phép chúng ta giải bài toán gradient descent được mô tả
ở trên. Nhìn vào vết tối ưu hóa ở trên, chúng ta có thể trực giác rằng lấy trung bình gradient trong quá khứ sẽ hoạt động tốt. Sau cùng, theo hướng $x_1$, điều này sẽ tổng hợp các gradient được căn chỉnh tốt, nhờ đó tăng quãng đường chúng ta đi được ở mỗi bước. Ngược lại, theo hướng $x_2$ nơi gradient dao động, gradient tổng hợp sẽ giảm kích thước bước do các dao động triệt tiêu lẫn nhau.
Dùng $\mathbf{v}_t$ thay cho gradient $\mathbf{g}_t$ cho ta các phương trình cập nhật sau:

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

Lưu ý rằng với $\beta = 0$, chúng ta thu được gradient descent thông thường. Trước khi đi sâu hơn vào các tính chất toán học, hãy nhanh chóng xem thuật toán hoạt động ra sao trong thực tế.

```python
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Như chúng ta có thể thấy, ngay cả với cùng tốc độ học đã dùng trước đó, momentum vẫn hội tụ tốt. Hãy xem điều gì xảy ra khi chúng ta giảm tham số momentum. Giảm một nửa xuống $\beta = 0.25$ dẫn đến một quỹ đạo hầu như không hội tụ. Dù vậy, nó vẫn tốt hơn nhiều so với không có momentum (khi nghiệm phân kỳ).

```python
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Lưu ý rằng chúng ta có thể kết hợp momentum với stochastic gradient descent và đặc biệt là minibatch stochastic gradient descent. Thay đổi duy nhất là trong trường hợp đó chúng ta thay các gradient $\mathbf{g}_{t, t-1}$ bằng $\mathbf{g}_t$. Cuối cùng, để thuận tiện, chúng ta khởi tạo $\mathbf{v}_0 = 0$ tại thời điểm $t=0$. Hãy xem trung bình rò rỉ thực sự làm gì với các cập nhật.

### Trọng Số Mẫu Hiệu Dụng

Nhớ rằng $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$. Ở giới hạn, các hạng cộng lại thành $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$. Nói cách khác, thay vì đi một bước kích thước $\eta$ trong gradient descent hoặc stochastic gradient descent, chúng ta đi một bước kích thước $\frac{\eta}{1-\beta}$, đồng thời xử lý một hướng giảm có thể có hành vi tốt hơn nhiều. Đây là hai lợi ích trong một. Để minh họa cách trọng số hoạt động với các lựa chọn $\beta$ khác nhau, hãy xét sơ đồ dưới đây.

```python
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## Thí Nghiệm Thực Tế

Hãy xem momentum hoạt động ra sao trong thực tế, tức là khi được dùng trong bối cảnh của một bộ tối ưu hóa đúng nghĩa. Để làm điều này, chúng ta cần một cài đặt có khả năng mở rộng hơn một chút.

### Cài Đặt Từ Đầu

So với stochastic gradient descent (minibatch), phương pháp momentum cần duy trì một tập biến phụ, tức là vận tốc. Nó có cùng hình dạng với các gradient (và các biến của bài toán tối ưu hóa). Trong cài đặt dưới đây, chúng ta gọi các biến này là `states`.

```python
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```python
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```python
#@tab mxnet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```python
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```python
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

Hãy xem điều này hoạt động thế nào trong thực tế.

```python
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Khi chúng ta tăng siêu tham số momentum `momentum` lên 0.9, nó tương ứng với kích thước mẫu hiệu dụng lớn hơn đáng kể là $\frac{1}{1 - 0.9} = 10$. Chúng ta giảm nhẹ tốc độ học xuống $0.01$ để giữ mọi thứ trong tầm kiểm soát.

```python
#@tab all
train_momentum(0.01, 0.9)
```

Giảm tốc độ học hơn nữa sẽ xử lý mọi vấn đề của các bài toán tối ưu hóa không trơn. Đặt nó thành $0.005$ cho các tính chất hội tụ tốt.

```python
#@tab all
train_momentum(0.005, 0.9)
```

### Cài Đặt Ngắn Gọn

Trong Gluon, hầu như không có gì cần làm vì bộ giải `sgd` tiêu chuẩn đã tích hợp momentum. Đặt các tham số khớp nhau cho một quỹ đạo rất tương tự.

```python
#@tab mxnet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```python
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```python
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## Phân Tích Lý Thuyết

Cho đến nay, ví dụ 2D của $f(x) = 0.1 x_1^2 + 2 x_2^2$ có vẻ khá được dựng lên. Bây giờ chúng ta sẽ thấy rằng nó thực ra khá đại diện cho các loại bài toán có thể gặp, ít nhất trong trường hợp tối thiểu hóa các hàm mục tiêu bậc hai lồi.

### Hàm Bậc Hai Lồi

Xét hàm

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

Đây là một hàm bậc hai tổng quát. Với các ma trận xác định dương $\mathbf{Q} \succ 0$, tức là các ma trận có trị riêng dương, hàm này có một bộ tối thiểu hóa tại $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ với giá trị nhỏ nhất $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Do đó, chúng ta có thể viết lại $h$ là

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

Gradient được cho bởi $\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$. Tức là, nó được cho bởi khoảng cách giữa $\mathbf{x}$ và bộ tối thiểu hóa, nhân với $\mathbf{Q}$. Do đó, vận tốc cũng là một tổ hợp tuyến tính của các hạng $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$.

Vì $\mathbf{Q}$ là xác định dương, nó có thể được phân rã thành hệ trị riêng thông qua $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ với một ma trận trực giao (xoay) $\mathbf{O}$ và một ma trận đường chéo $\boldsymbol{\Lambda}$ gồm các trị riêng dương. Điều này cho phép chúng ta thực hiện đổi biến từ $\mathbf{x}$ sang $\mathbf{z} \stackrel{\textrm{def}}{=} \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ để thu được một biểu thức đơn giản hơn nhiều:

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

Ở đây $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Vì $\mathbf{O}$ chỉ là một ma trận trực giao, điều này không làm nhiễu gradient theo cách có ý nghĩa. Biểu diễn theo $\mathbf{z}$, gradient descent trở thành

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

Sự kiện quan trọng trong biểu thức này là gradient descent *không trộn lẫn* giữa các không gian riêng khác nhau. Tức là, khi biểu diễn theo hệ trị riêng của $\mathbf{Q}$, bài toán tối ưu hóa diễn ra theo từng tọa độ. Điều này cũng đúng với

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

Khi làm điều này, chúng ta vừa chứng minh định lý sau: gradient descent có và không có momentum cho một hàm bậc hai lồi phân rã thành tối ưu hóa theo từng tọa độ theo hướng của các vector riêng của ma trận bậc hai.

### Hàm Vô Hướng

Với kết quả trên, hãy xem điều gì xảy ra khi chúng ta tối thiểu hóa hàm $f(x) = \frac{\lambda}{2} x^2$. Với gradient descent, chúng ta có

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

Bất cứ khi nào $|1 - \eta \lambda| < 1$, quá trình tối ưu hóa này hội tụ với tốc độ hàm mũ vì sau $t$ bước, chúng ta có $x_t = (1 - \eta \lambda)^t x_0$. Điều này cho thấy tốc độ hội tụ ban đầu cải thiện như thế nào khi chúng ta tăng tốc độ học $\eta$ cho đến khi $\eta \lambda = 1$. Vượt quá mức đó, mọi thứ phân kỳ và với $\eta \lambda > 2$, bài toán tối ưu hóa phân kỳ.

```python
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

Để phân tích hội tụ trong trường hợp momentum, chúng ta bắt đầu bằng cách viết lại các phương trình cập nhật theo hai vô hướng: một cho $x$ và một cho vận tốc $v$. Điều này cho ta:

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

Chúng ta dùng $\mathbf{R}$ để ký hiệu ma trận $2 \times 2$ chi phối hành vi hội tụ. Sau $t$ bước, lựa chọn ban đầu $[v_0, x_0]$ trở thành $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$. Do đó, tốc độ hội tụ phụ thuộc vào các trị riêng của $\mathbf{R}$. Xem [bài viết Distill](https://distill.pub/2017/momentum/) của Goh.2017 để có hoạt ảnh tuyệt vời và Flammarion.Bach.2015 để có phân tích chi tiết. Có thể chứng minh rằng với $0 < \eta \lambda < 2 + 2 \beta$, vận tốc hội tụ. Đây là một miền tham số khả thi lớn hơn so với $0 < \eta \lambda < 2$ của gradient descent. Nó cũng gợi ý rằng nói chung các giá trị $\beta$ lớn là đáng mong muốn. Các chi tiết sâu hơn đòi hỏi khá nhiều kỹ thuật, và chúng tôi đề nghị độc giả quan tâm tham khảo các công trình gốc.

## Tóm Tắt

* Momentum thay gradient bằng một trung bình rò rỉ trên các gradient trước đó. Điều này tăng tốc hội tụ đáng kể.
* Nó đáng mong muốn cho cả gradient descent không nhiễu và stochastic gradient descent (có nhiễu).
* Momentum ngăn quá trình tối ưu hóa bị đình trệ, điều vốn có nhiều khả năng xảy ra hơn với stochastic gradient descent.
* Số gradient hiệu dụng được cho bởi $\frac{1}{1-\beta}$ do việc giảm trọng số theo hàm mũ của dữ liệu quá khứ.
* Trong trường hợp các bài toán bậc hai lồi, điều này có thể được phân tích tường minh chi tiết.
* Cài đặt khá đơn giản nhưng đòi hỏi chúng ta lưu một vector trạng thái bổ sung (vận tốc $\mathbf{v}$).

## Bài Tập

1. Dùng các kết hợp khác của siêu tham số momentum và tốc độ học, rồi quan sát và phân tích các kết quả thí nghiệm khác nhau.
1. Thử gradient descent và momentum cho một bài toán bậc hai trong đó bạn có nhiều trị riêng, tức là $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, ví dụ $\lambda_i = 2^{-i}$. Vẽ cách các giá trị của $x$ giảm với khởi tạo $x_i = 1$.
1. Dẫn xuất giá trị nhỏ nhất và bộ tối thiểu hóa cho $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$.
1. Điều gì thay đổi khi chúng ta thực hiện stochastic gradient descent với momentum? Điều gì xảy ra khi chúng ta dùng minibatch stochastic gradient descent với momentum? Hãy thử nghiệm với các tham số.


[Discussions](https://discuss.d2l.ai/t/1070)

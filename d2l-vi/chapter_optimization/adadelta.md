# Adadelta
<a id="sec_adadelta"></a>

Adadelta là một biến thể khác của AdaGrad ([sec_adagrad](#sec_adagrad)). Khác biệt chính nằm ở chỗ nó làm giảm mức độ mà tốc độ học thích nghi theo các tọa độ. Hơn nữa, theo truyền thống nó được xem là không có tốc độ học vì dùng chính lượng thay đổi làm chuẩn để hiệu chỉnh thay đổi trong tương lai. Thuật toán này được đề xuất trong Zeiler.2012. Với thảo luận về các thuật toán trước đó cho đến nay, nó khá đơn giản.

## Thuật Toán

Tóm lại, Adadelta dùng hai biến trạng thái, $\mathbf{s}_t$ để lưu một trung bình rò rỉ của moment bậc hai của gradient và $\Delta\mathbf{x}_t$ để lưu một trung bình rò rỉ của moment bậc hai của thay đổi tham số trong chính mô hình. Lưu ý rằng chúng ta dùng ký hiệu và cách đặt tên gốc của các tác giả để tương thích với các công bố và cài đặt khác (không có lý do thực sự nào khác để dùng các biến Hy Lạp khác nhau nhằm biểu thị một tham số phục vụ cùng mục đích trong momentum, Adagrad, RMSProp và Adadelta).

Sau đây là các chi tiết kỹ thuật của Adadelta. Với tham số hiện hành là $\rho$, chúng ta thu được các cập nhật rò rỉ sau, tương tự [sec_rmsprop](#sec_rmsprop):

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

Khác biệt so với [sec_rmsprop](#sec_rmsprop) là chúng ta thực hiện cập nhật với gradient đã đổi thang $\mathbf{g}_t'$, tức là

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Vậy gradient đã đổi thang $\mathbf{g}_t'$ là gì? Chúng ta có thể tính nó như sau:

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

trong đó $\Delta \mathbf{x}_{t-1}$ là trung bình rò rỉ của bình phương các gradient đã đổi thang $\mathbf{g}_t'$. Chúng ta khởi tạo $\Delta \mathbf{x}_{0}$ bằng $0$ và cập nhật nó ở mỗi bước bằng $\mathbf{g}_t'$, tức là

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

và $\epsilon$ (một giá trị nhỏ như $10^{-5}$) được thêm vào để duy trì ổn định số.


## Cài Đặt

Adadelta cần duy trì hai biến trạng thái cho mỗi biến, $\mathbf{s}_t$ và $\Delta\mathbf{x}_t$. Điều này cho cài đặt sau.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```python
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

Chọn $\rho = 0.9$ tương ứng với thời gian bán rã 10 cho mỗi cập nhật tham số. Điều này thường hoạt động khá tốt. Chúng ta thu được hành vi sau.

```python
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

Với cài đặt ngắn gọn, chúng ta chỉ cần dùng thuật toán Adadelta từ các API cấp cao. Điều này cho lệnh một dòng sau để gọi gọn hơn nhiều.

```python
#@tab mxnet
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```python
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```python
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it is converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## Tóm Tắt

* Adadelta không có tham số tốc độ học. Thay vào đó, nó dùng chính tốc độ thay đổi trong các tham số để thích nghi tốc độ học.
* Adadelta cần hai biến trạng thái để lưu moment bậc hai của gradient và thay đổi trong tham số.
* Adadelta dùng các trung bình rò rỉ để duy trì một ước lượng chạy của các thống kê phù hợp.

## Bài Tập

1. Điều chỉnh giá trị của $\rho$. Điều gì xảy ra?
1. Chỉ ra cách cài đặt thuật toán mà không dùng $\mathbf{g}_t'$. Vì sao đây có thể là một ý tưởng hay?
1. Adadelta có thật sự không cần tốc độ học không? Bạn có thể tìm các bài toán tối ưu hóa làm Adadelta thất bại không?
1. So sánh Adadelta với Adagrad và RMS prop để thảo luận hành vi hội tụ của chúng.


[Discussions](https://discuss.d2l.ai/t/1076)

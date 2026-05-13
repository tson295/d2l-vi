# RMSProp
<a id="sec_rmsprop"></a>


Một trong những vấn đề chính trong [sec_adagrad](#sec_adagrad) là tốc độ học giảm theo một lịch định trước, về thực chất là $\mathcal{O}(t^{-\frac{1}{2}})$. Mặc dù điều này nhìn chung phù hợp với các bài toán lồi, nó có thể không lý tưởng cho các bài toán không lồi, chẳng hạn như những bài toán gặp trong deep learning. Tuy vậy, khả năng thích nghi theo từng tọa độ của Adagrad là rất đáng mong muốn như một bộ tiền điều kiện hóa.

Tieleman.Hinton.2012 đề xuất thuật toán RMSProp như một cách sửa đơn giản để tách lịch tốc độ học khỏi các tốc độ học thích nghi theo tọa độ. Vấn đề là Adagrad tích lũy bình phương của gradient $\mathbf{g}_t$ vào một vector trạng thái $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$. Kết quả là $\mathbf{s}_t$ tiếp tục tăng không bị chặn do thiếu chuẩn hóa, về cơ bản là tuyến tính khi thuật toán hội tụ.

Một cách sửa vấn đề này là dùng $\mathbf{s}_t / t$. Với các phân phối hợp lý của $\mathbf{g}_t$, đại lượng này sẽ hội tụ. Đáng tiếc là có thể mất rất lâu cho đến khi hành vi giới hạn bắt đầu có ý nghĩa, vì quy trình này nhớ toàn bộ quỹ đạo giá trị. Một cách thay thế là dùng trung bình rò rỉ giống như cách chúng ta đã dùng trong phương pháp momentum, tức là $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ với một tham số $\gamma > 0$ nào đó. Giữ nguyên tất cả các phần khác cho ta RMSProp.

## Thuật Toán

Hãy viết chi tiết các phương trình.

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

Hằng số $\epsilon > 0$ thường được đặt là $10^{-6}$ để đảm bảo chúng ta không bị chia cho không hoặc có kích thước bước quá lớn. Với khai triển này, giờ đây chúng ta có thể tự do kiểm soát tốc độ học $\eta$ độc lập với phép đổi thang được áp dụng theo từng tọa độ. Theo trung bình rò rỉ, chúng ta có thể áp dụng cùng lập luận như đã áp dụng trước đó trong trường hợp phương pháp momentum. Khai triển định nghĩa của $\mathbf{s}_t$ cho ta

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

Như trước trong [sec_momentum](#sec_momentum), chúng ta dùng $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$. Do đó tổng trọng số được chuẩn hóa thành $1$ với thời gian bán rã của một quan sát là $\gamma^{-1}$. Hãy trực quan hóa các trọng số cho 40 bước thời gian trước đó với nhiều lựa chọn $\gamma$ khác nhau.

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
from d2l import torch as d2l
import torch
import math
```

```python
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```python
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## Cài Đặt Từ Đầu

Như trước, chúng ta dùng hàm bậc hai $f(\mathbf{x})=0.1x_1^2+2x_2^2$ để quan sát quỹ đạo của RMSProp. Nhớ rằng trong [sec_adagrad](#sec_adagrad), khi dùng Adagrad với tốc độ học 0.4, các biến chỉ di chuyển rất chậm ở các giai đoạn sau của thuật toán vì tốc độ học giảm quá nhanh. Vì $\eta$ được kiểm soát riêng, điều này không xảy ra với RMSProp.

```python
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Tiếp theo, chúng ta cài đặt RMSProp để dùng trong một mạng sâu. Điều này cũng đơn giản tương tự.

```python
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```python
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```python
#@tab mxnet
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```python
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```python
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Chúng ta đặt tốc độ học ban đầu là 0.01 và hạng trọng số $\gamma$ là 0.9. Tức là, $\mathbf{s}$ tổng hợp trung bình trên $1/(1-\gamma) = 10$ quan sát trước đó của bình phương gradient.

```python
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## Cài Đặt Ngắn Gọn

Vì RMSProp là một thuật toán khá phổ biến, nó cũng có sẵn trong thể hiện `Trainer`. Tất cả những gì chúng ta cần làm là khởi tạo nó bằng một thuật toán tên `rmsprop`, gán $\gamma$ cho tham số `gamma1`.

```python
#@tab mxnet
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```python
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```python
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## Tóm Tắt

* RMSProp rất giống Adagrad ở chỗ cả hai đều dùng bình phương của gradient để đổi thang các hệ số.
* RMSProp chia sẻ với momentum phép lấy trung bình rò rỉ. Tuy nhiên, RMSProp dùng kỹ thuật này để điều chỉnh bộ tiền điều kiện hóa theo từng hệ số.
* Trong thực tế, tốc độ học cần được người làm thí nghiệm lập lịch.
* Hệ số $\gamma$ xác định lịch sử dài bao lâu khi điều chỉnh thang đo theo từng tọa độ.

## Bài Tập

1. Điều gì xảy ra trong thí nghiệm nếu chúng ta đặt $\gamma = 1$? Tại sao?
1. Xoay bài toán tối ưu hóa để tối thiểu hóa $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Điều gì xảy ra với hội tụ?
1. Thử xem điều gì xảy ra với RMSProp trên một bài toán machine learning thực tế, chẳng hạn huấn luyện trên Fashion-MNIST. Thử nghiệm với các lựa chọn khác nhau để điều chỉnh tốc độ học.
1. Bạn có muốn điều chỉnh $\gamma$ khi quá trình tối ưu hóa diễn ra không? RMSProp nhạy với điều này đến mức nào?


[Discussions](https://discuss.d2l.ai/t/1074)

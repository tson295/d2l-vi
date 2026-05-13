# Adam
<a id="sec_adam"></a>

Trong các thảo luận dẫn đến phần này, chúng ta đã gặp một số kỹ thuật để tối ưu hóa hiệu quả. Hãy tóm tắt lại chi tiết ở đây:

* Chúng ta đã thấy rằng [sec_sgd](#sec_sgd) hiệu quả hơn Gradient Descent khi giải các bài toán tối ưu hóa, ví dụ nhờ khả năng chống chịu vốn có với dữ liệu dư thừa.
* Chúng ta đã thấy rằng [sec_minibatch_sgd](#sec_minibatch_sgd) đem lại hiệu quả bổ sung đáng kể phát sinh từ vector hóa, bằng cách dùng các tập quan sát lớn hơn trong một minibatch. Đây là chìa khóa cho xử lý đa máy, đa GPU và xử lý song song nói chung một cách hiệu quả.
* [sec_momentum](#sec_momentum) thêm một cơ chế tổng hợp lịch sử các gradient trước đó để tăng tốc hội tụ.
* [sec_adagrad](#sec_adagrad) dùng đổi thang theo từng tọa độ để cho phép một bộ tiền điều kiện hóa hiệu quả về tính toán.
* [sec_rmsprop](#sec_rmsprop) tách đổi thang theo từng tọa độ khỏi điều chỉnh tốc độ học.

Adam [Kingma.Ba.2014] kết hợp tất cả các kỹ thuật này thành một thuật toán học hiệu quả. Như kỳ vọng, đây là một thuật toán đã trở nên khá phổ biến như một trong những thuật toán tối ưu hóa mạnh mẽ và hiệu quả hơn để dùng trong deep learning. Dù vậy, nó không phải không có vấn đề. Cụ thể, [Reddi.Kale.Kumar.2019] cho thấy có những tình huống Adam có thể phân kỳ do kiểm soát phương sai kém. Trong một công trình tiếp theo, Zaheer.Reddi.Sachan.ea.2018 đề xuất một bản sửa nhanh cho Adam, gọi là Yogi, để xử lý các vấn đề này. Sẽ nói thêm về điều này sau. Hiện tại, hãy xem lại thuật toán Adam.

## Thuật Toán

Một trong những thành phần chính của Adam là nó dùng các trung bình động có trọng số hàm mũ (cũng gọi là trung bình rò rỉ) để thu được ước lượng của cả momentum và moment bậc hai của gradient. Tức là, nó dùng các biến trạng thái

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

Ở đây $\beta_1$ và $\beta_2$ là các tham số trọng số không âm. Các lựa chọn phổ biến là $\beta_1 = 0.9$ và $\beta_2 = 0.999$. Tức là, ước lượng phương sai di chuyển *chậm hơn nhiều* so với hạng momentum. Lưu ý rằng nếu khởi tạo $\mathbf{v}_0 = \mathbf{s}_0 = 0$, ban đầu chúng ta có một lượng chệch đáng kể về phía các giá trị nhỏ hơn. Điều này có thể được xử lý bằng cách dùng sự kiện $\sum_{i=0}^{t-1} \beta^i = \frac{1 - \beta^t}{1 - \beta}$ để chuẩn hóa lại các hạng. Tương ứng, các biến trạng thái đã chuẩn hóa được cho bởi

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \textrm{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

Với các ước lượng phù hợp trong tay, giờ chúng ta có thể viết các phương trình cập nhật. Trước hết, chúng ta đổi thang gradient theo cách rất giống RMSProp để thu được

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

Khác với RMSProp, cập nhật của chúng ta dùng momentum $\hat{\mathbf{v}}_t$ thay vì chính gradient. Hơn nữa, có một khác biệt nhỏ về hình thức vì việc đổi thang xảy ra bằng $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ thay vì $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$. Dạng trước được cho là hoạt động hơi tốt hơn trong thực tế, do đó có sự khác biệt so với RMSProp. Thông thường, chúng ta chọn $\epsilon = 10^{-6}$ để có đánh đổi tốt giữa ổn định số và độ trung thực.

Bây giờ chúng ta đã có đủ các mảnh để tính cập nhật. Điều này hơi thiếu cao trào và chúng ta có một cập nhật đơn giản có dạng

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Khi xem lại thiết kế của Adam, nguồn cảm hứng của nó rất rõ ràng. Momentum và thang đo thể hiện rõ trong các biến trạng thái. Định nghĩa khá đặc biệt của chúng buộc chúng ta khử chệch các hạng (điều này có thể được sửa bằng một cách khởi tạo và điều kiện cập nhật hơi khác). Thứ hai, việc kết hợp cả hai hạng khá trực tiếp, dựa trên RMSProp. Cuối cùng, tốc độ học tường minh $\eta$ cho phép chúng ta kiểm soát độ dài bước để xử lý các vấn đề hội tụ.

## Cài Đặt

Cài đặt Adam từ đầu không quá đáng ngại. Để thuận tiện, chúng ta lưu bộ đếm bước thời gian $t$ trong từ điển `hyperparams`. Ngoài ra, mọi thứ đều đơn giản.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```python
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Chúng ta đã sẵn sàng dùng Adam để huấn luyện mô hình. Chúng ta dùng tốc độ học $\eta = 0.01$.

```python
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

Cài đặt ngắn gọn hơn khá trực tiếp vì `adam` là một trong các thuật toán được cung cấp như một phần của thư viện tối ưu hóa `trainer` của Gluon. Do đó, chúng ta chỉ cần truyền các tham số cấu hình cho một cài đặt trong Gluon.

```python
#@tab mxnet
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```python
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```python
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Một trong những vấn đề của Adam là nó có thể không hội tụ ngay cả trong bối cảnh lồi khi ước lượng moment bậc hai trong $\mathbf{s}_t$ bùng lên. Để sửa, Zaheer.Reddi.Sachan.ea.2018 đề xuất một cập nhật (và khởi tạo) tinh chỉnh cho $\mathbf{s}_t$. Để hiểu điều gì đang diễn ra, hãy viết lại cập nhật Adam như sau:

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

Bất cứ khi nào $\mathbf{g}_t^2$ có phương sai cao hoặc các cập nhật thưa, $\mathbf{s}_t$ có thể quên các giá trị quá khứ quá nhanh. Một cách sửa khả dĩ là thay $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ bằng $\mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$. Bây giờ độ lớn của cập nhật không còn phụ thuộc vào lượng sai lệch. Điều này cho các cập nhật Yogi

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

Các tác giả còn khuyên khởi tạo momentum trên một batch ban đầu lớn hơn thay vì chỉ một ước lượng điểm ban đầu. Chúng ta lược bỏ chi tiết vì chúng không thiết yếu cho thảo luận và vì ngay cả khi không có điều này, hội tụ vẫn khá tốt.

```python
#@tab mxnet
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```python
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```python
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## Tóm Tắt

* Adam kết hợp các đặc điểm của nhiều thuật toán tối ưu hóa thành một quy tắc cập nhật khá mạnh mẽ.
* Được tạo trên cơ sở RMSProp, Adam cũng dùng EWMA trên minibatch stochastic gradient.
* Adam dùng hiệu chỉnh chệch để điều chỉnh cho giai đoạn khởi động chậm khi ước lượng momentum và moment bậc hai.
* Với các gradient có phương sai đáng kể, chúng ta có thể gặp vấn đề về hội tụ. Chúng có thể được khắc phục bằng cách dùng minibatch lớn hơn hoặc chuyển sang một ước lượng cải thiện cho $\mathbf{s}_t$. Yogi cung cấp một lựa chọn thay thế như vậy.

## Bài Tập

1. Điều chỉnh tốc độ học rồi quan sát và phân tích các kết quả thí nghiệm.
1. Bạn có thể viết lại các cập nhật momentum và moment bậc hai sao cho không cần hiệu chỉnh chệch không?
1. Vì sao bạn cần giảm tốc độ học $\eta$ khi chúng ta hội tụ?
1. Hãy thử xây dựng một trường hợp trong đó Adam phân kỳ còn Yogi hội tụ.


[Discussions](https://discuss.d2l.ai/t/1078)

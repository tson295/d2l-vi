# Hợp Lý Cực Đại
<a id="sec_maximum_likelihood"></a>

Một trong những cách suy nghĩ thường gặp nhất trong machine learning là góc nhìn hợp lý cực đại. Đây là khái niệm rằng khi làm việc với một mô hình xác suất có các tham số chưa biết, những tham số khiến dữ liệu có xác suất cao nhất là những tham số có khả năng đúng nhất.

## Nguyên Lý Hợp Lý Cực Đại

Điều này có một diễn giải Bayes có thể hữu ích để suy nghĩ. Giả sử ta có một mô hình với các tham số $\boldsymbol{\theta}$ và một tập các ví dụ dữ liệu $X$. Để cụ thể, ta có thể tưởng tượng $\boldsymbol{\theta}$ là một giá trị đơn biểu diễn xác suất một đồng xu rơi vào mặt ngửa khi được tung, và $X$ là một chuỗi các lần tung đồng xu độc lập. Ta sẽ xem xét ví dụ này kỹ hơn ở phần sau.

Nếu ta muốn tìm giá trị có khả năng nhất cho các tham số của mô hình, điều đó có nghĩa là ta muốn tìm

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$

Theo quy tắc Bayes, điều này giống với

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

Biểu thức $P(X)$, tức xác suất sinh ra dữ liệu mà không phụ thuộc vào tham số, hoàn toàn không phụ thuộc vào $\boldsymbol{\theta}$, nên có thể bỏ đi mà không thay đổi lựa chọn tốt nhất của $\boldsymbol{\theta}$. Tương tự, giờ ta có thể giả định rằng ta không có giả định tiên nghiệm nào về tập tham số nào tốt hơn tập tham số khác, nên ta có thể tuyên bố rằng $P(\boldsymbol{\theta})$ cũng không phụ thuộc vào theta! Chẳng hạn, điều này hợp lý trong ví dụ tung đồng xu của ta, nơi xác suất ra mặt ngửa có thể là bất kỳ giá trị nào trong $[0,1]$ mà không có niềm tin tiên nghiệm rằng đồng xu công bằng hay không (thường được gọi là *tiên nghiệm không thông tin*). Vì vậy ta thấy việc áp dụng quy tắc Bayes cho thấy lựa chọn tốt nhất của ta cho $\boldsymbol{\theta}$ là ước lượng hợp lý cực đại cho $\boldsymbol{\theta}$:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

Theo thuật ngữ thông dụng, xác suất của dữ liệu khi biết các tham số ($P(X \mid \boldsymbol{\theta})$) được gọi là *hợp lý*.

### Một Ví Dụ Cụ Thể

Hãy xem điều này hoạt động như thế nào trong một ví dụ cụ thể. Giả sử ta có một tham số đơn $\theta$ biểu diễn xác suất một lần tung đồng xu ra mặt ngửa. Khi đó xác suất ra mặt sấp là $1-\theta$, và nếu dữ liệu quan sát được $X$ là một chuỗi có $n_H$ lần ngửa và $n_T$ lần sấp, ta có thể dùng thực tế rằng các xác suất độc lập được nhân với nhau để thấy rằng

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

Nếu ta tung $13$ đồng xu và nhận được chuỗi "HHHTHTTHHHHHT", có $n_H = 9$ và $n_T = 4$, ta thấy rằng đây là

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

Một điều hay ở ví dụ này là ta biết trước câu trả lời. Thật vậy, nếu ta nói bằng lời: "Tôi tung 13 đồng xu, và 9 lần ra mặt ngửa, dự đoán tốt nhất của ta cho xác suất đồng xu ra mặt ngửa là bao nhiêu?", mọi người sẽ đoán đúng là $9/13$. Phương pháp hợp lý cực đại này sẽ cho ta một cách thu được con số đó từ các nguyên lý đầu tiên theo cách có thể tổng quát hóa sang những tình huống phức tạp hơn rất nhiều.

Với ví dụ của ta, đồ thị của $P(X \mid \theta)$ như sau:

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```python
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

Giá trị cực đại của nó nằm đâu đó gần giá trị ta kỳ vọng $9/13 \approx 0.7\ldots$. Để xem có đúng chính xác ở đó không, ta có thể dùng giải tích. Lưu ý rằng tại cực đại, gradient của hàm bằng phẳng. Vì vậy, ta có thể tìm ước lượng hợp lý cực đại :eqref:`eq_max_like` bằng cách tìm các giá trị của $\theta$ tại đó đạo hàm bằng không, rồi tìm giá trị cho xác suất cao nhất. Ta tính:

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

Phương trình này có ba nghiệm: $0$, $1$ và $9/13$. Hai nghiệm đầu rõ ràng là cực tiểu, không phải cực đại, vì chúng gán xác suất $0$ cho chuỗi của ta. Giá trị cuối cùng *không* gán xác suất bằng không cho chuỗi của ta, và do đó phải là ước lượng hợp lý cực đại $\hat \theta = 9/13$.

## Tối Ưu Hóa Số Và Negative Log-Likelihood

Ví dụ trước khá đẹp, nhưng nếu ta có hàng tỷ tham số và ví dụ dữ liệu thì sao?

Trước hết, lưu ý rằng nếu ta giả định tất cả các ví dụ dữ liệu là độc lập, ta không còn có thể xét trực tiếp hàm hợp lý một cách thực tế nữa, vì nó là tích của rất nhiều xác suất. Thật vậy, mỗi xác suất nằm trong $[0,1]$, chẳng hạn thường có giá trị khoảng $1/2$, và tích $(1/2)^{1000000000}$ thấp hơn độ chính xác máy rất nhiều. Ta không thể làm việc trực tiếp với nó.

Tuy nhiên, nhớ rằng logarit biến tích thành tổng, trong trường hợp đó

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

Con số này nằm hoàn toàn vừa trong cả một số thực dấu phẩy động $32$-bit độ chính xác đơn. Vì vậy, ta nên xét *log-likelihood*, tức là

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

Vì hàm $x \mapsto \log(x)$ là hàm tăng, cực đại hóa likelihood cũng giống như cực đại hóa log-likelihood. Thật vậy, trong [sec_naive_bayes](#sec_naive_bayes), ta sẽ thấy lập luận này được áp dụng khi làm việc với ví dụ cụ thể về bộ phân loại naive Bayes.

Ta thường làm việc với các hàm mất mát, nơi ta muốn cực tiểu hóa mất mát. Ta có thể biến hợp lý cực đại thành việc cực tiểu hóa một mất mát bằng cách lấy $-\log(P(X \mid \boldsymbol{\theta}))$, được gọi là *negative log-likelihood*.

Để minh họa điều này, hãy xét bài toán tung đồng xu trước đó và giả vờ rằng ta không biết nghiệm dạng đóng. Ta có thể tính rằng

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

Điều này có thể được viết thành code và được tối ưu hóa tự do ngay cả với hàng tỷ lần tung đồng xu.

```python
#@tab mxnet
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```python
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```python
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

Sự tiện lợi về mặt số học không phải là lý do duy nhất khiến người ta thích dùng negative log-likelihood. Có một số lý do khác khiến nó được ưa chuộng hơn.

Lý do thứ hai ta xét log-likelihood là việc áp dụng các quy tắc giải tích được đơn giản hóa. Như đã thảo luận ở trên, do các giả định độc lập, hầu hết các xác suất ta gặp trong machine learning là tích của các xác suất riêng lẻ.

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

Điều này có nghĩa là nếu ta trực tiếp áp dụng quy tắc tích để tính đạo hàm, ta nhận được

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

Điều này cần $n(n-1)$ phép nhân, cùng với $(n-1)$ phép cộng, nên tỷ lệ với thời gian bậc hai theo đầu vào! Nếu đủ khéo léo trong việc nhóm các hạng tử, ta có thể giảm điều này xuống thời gian tuyến tính, nhưng nó đòi hỏi phải suy nghĩ. Với negative log-likelihood, thay vào đó ta có

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

từ đó cho

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

Điều này chỉ cần $n$ phép chia và $n-1$ phép cộng, và do đó có thời gian tuyến tính theo đầu vào.

Lý do thứ ba và cuối cùng để xét negative log-likelihood là mối quan hệ với lý thuyết thông tin, mà chúng ta sẽ thảo luận chi tiết trong [sec_information_theory](#sec_information_theory). Đây là một lý thuyết toán học chặt chẽ, cung cấp cách đo mức độ thông tin hoặc tính ngẫu nhiên trong một biến ngẫu nhiên. Đối tượng nghiên cứu then chốt trong lĩnh vực đó là entropy, được định nghĩa là

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

đo mức độ ngẫu nhiên của một nguồn. Lưu ý rằng đây không gì khác hơn là xác suất $-\log$ trung bình, và do đó nếu ta lấy negative log-likelihood rồi chia cho số ví dụ dữ liệu, ta thu được một họ hàng của entropy được gọi là cross-entropy. Chỉ riêng diễn giải lý thuyết này cũng đã đủ thuyết phục để thúc đẩy việc báo cáo negative log-likelihood trung bình trên tập dữ liệu như một cách đo hiệu năng mô hình.

## Hợp Lý Cực Đại Cho Biến Liên Tục

Mọi điều ta đã làm đến giờ đều giả định ta đang làm việc với biến ngẫu nhiên rời rạc, nhưng nếu ta muốn làm việc với biến liên tục thì sao?

Tóm tắt ngắn gọn là không có gì thay đổi cả, ngoại trừ việc ta thay mọi trường hợp của xác suất bằng mật độ xác suất. Nhớ rằng ta viết mật độ bằng chữ thường $p$, điều này có nghĩa là, chẳng hạn, bây giờ ta nói

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

Câu hỏi trở thành: "Tại sao điều này lại ổn?" Sau cùng, lý do ta đưa mật độ vào là vì xác suất nhận các kết quả cụ thể tự thân bằng không, vậy chẳng phải xác suất sinh dữ liệu của ta với bất kỳ tập tham số nào cũng bằng không sao?

Quả thật là như vậy, và việc hiểu vì sao ta có thể chuyển sang mật độ là một bài tập theo dõi điều gì xảy ra với các epsilon.

Trước hết, hãy định nghĩa lại mục tiêu của ta. Giả sử với các biến ngẫu nhiên liên tục, ta không còn muốn tính xác suất nhận đúng chính xác giá trị, mà thay vào đó là khớp trong một phạm vi $\epsilon$ nào đó. Để đơn giản, ta giả định dữ liệu của mình là các quan sát lặp lại $x_1, \ldots, x_N$ của các biến ngẫu nhiên phân phối giống nhau $X_1, \ldots, X_N$. Như ta đã thấy trước đó, điều này có thể được viết là

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

Do đó, nếu ta lấy logarit âm của biểu thức này, ta thu được

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

Nếu ta khảo sát biểu thức này, nơi duy nhất $\epsilon$ xuất hiện là trong hằng số cộng $-N\log(\epsilon)$. Hằng số này hoàn toàn không phụ thuộc vào các tham số $\boldsymbol{\theta}$, nên lựa chọn tối ưu của $\boldsymbol{\theta}$ không phụ thuộc vào lựa chọn $\epsilon$ của ta! Dù ta yêu cầu bốn chữ số hay bốn trăm chữ số, lựa chọn tốt nhất của $\boldsymbol{\theta}$ vẫn như nhau, vì vậy ta có thể tự do bỏ epsilon để thấy rằng điều ta muốn tối ưu là

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

Vì vậy, ta thấy rằng góc nhìn hợp lý cực đại có thể hoạt động với biến ngẫu nhiên liên tục dễ dàng như với biến rời rạc bằng cách thay xác suất bằng mật độ xác suất.

## Tóm Tắt
* Nguyên lý hợp lý cực đại cho ta biết rằng mô hình khớp tốt nhất cho một tập dữ liệu cho trước là mô hình sinh dữ liệu với xác suất cao nhất.
* Người ta thường làm việc với negative log-likelihood thay thế vì nhiều lý do: ổn định số học, chuyển tích thành tổng (và sự đơn giản hóa tương ứng trong tính toán gradient), cùng các liên hệ lý thuyết với lý thuyết thông tin.
* Dù dễ thúc đẩy nhất trong bối cảnh rời rạc, nó cũng có thể được tổng quát hóa tự do sang bối cảnh liên tục bằng cách cực đại hóa mật độ xác suất được gán cho các điểm dữ liệu.

## Bài Tập
1. Giả sử bạn biết rằng một biến ngẫu nhiên không âm có mật độ $\alpha e^{-\alpha x}$ với một giá trị nào đó $\alpha>0$. Bạn thu được một quan sát duy nhất từ biến ngẫu nhiên, đó là số $3$. Ước lượng hợp lý cực đại cho $\alpha$ là gì?
2. Giả sử bạn có một tập dữ liệu gồm các mẫu $\{x_i\}_{i=1}^N$ được rút ra từ một phân phối Gaussian có trung bình chưa biết, nhưng phương sai bằng $1$. Ước lượng hợp lý cực đại cho trung bình là gì?


[Thảo luận](https://discuss.d2l.ai/t/1096)

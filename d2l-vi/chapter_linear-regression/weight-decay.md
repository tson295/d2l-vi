# Suy giảm Trọng số
<a id="sec_weight_decay"></a>

Bây giờ ta đã đặc trưng hóa bài toán quá khớp,
ta có thể giới thiệu kỹ thuật *chuẩn hóa* đầu tiên.
Hãy nhớ rằng ta luôn có thể giảm thiểu quá khớp
bằng cách thu thập thêm dữ liệu huấn luyện.
Tuy nhiên, điều đó có thể tốn kém, mất nhiều thời gian,
hoặc hoàn toàn nằm ngoài tầm kiểm soát của ta,
khiến nó không thể thực hiện được trong ngắn hạn.
Hiện tại, ta có thể giả định rằng ta đã có
nhiều dữ liệu chất lượng cao như nguồn lực cho phép
và tập trung vào các công cụ trong tầm tay
khi tập dữ liệu được xem là đã cho.

Hãy nhớ lại rằng trong ví dụ hồi quy đa thức
([subsec_polynomial-curve-fitting](#subsec_polynomial-curve-fitting))
ta có thể giới hạn năng lực mô hình của mình
bằng cách điều chỉnh bậc
của đa thức được khớp.
Thực ra, giới hạn số lượng đặc trưng
là một kỹ thuật phổ biến để giảm thiểu quá khớp.
Tuy nhiên, đơn giản là loại bỏ các đặc trưng
có thể là công cụ quá thô bạo.
Tiếp tục với ví dụ hồi quy đa thức,
hãy xem xét điều gì có thể xảy ra
với đầu vào nhiều chiều.
Các phần mở rộng tự nhiên của đa thức
cho dữ liệu đa biến được gọi là *đơn thức*,
chỉ đơn giản là tích của lũy thừa các biến.
Bậc của một đơn thức là tổng của các lũy thừa.
Chẳng hạn, $x_1^2 x_2$ và $x_3 x_5^2$
đều là các đơn thức bậc 3.

Lưu ý rằng số số hạng có bậc $d$
tăng nhanh chóng khi $d$ lớn hơn.
Cho $k$ biến, số đơn thức
bậc $d$ là ${k - 1 + d} \choose {k - 1}$.
Ngay cả những thay đổi nhỏ về bậc, chẳng hạn từ $2$ đến $3$,
làm tăng đáng kể độ phức tạp của mô hình.
Do đó ta thường cần một công cụ tinh tế hơn
để điều chỉnh độ phức tạp hàm.


```python
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```


## Chuẩn và Suy giảm Trọng số

(**Thay vì trực tiếp thao túng số lượng tham số,
*suy giảm trọng số*, hoạt động bằng cách hạn chế các giá trị
mà các tham số có thể nhận.**)
Thường được gọi là chuẩn hóa $\ell_2$
bên ngoài giới deep learning
khi được tối ưu hóa bởi minibatch stochastic gradient descent,
suy giảm trọng số có thể là kỹ thuật được sử dụng rộng rãi nhất
để chuẩn hóa các mô hình machine learning tham số.
Kỹ thuật được thúc đẩy bởi trực giác cơ bản
rằng trong tất cả các hàm $f$,
hàm $f = 0$
(gán giá trị $0$ cho tất cả đầu vào)
theo một nghĩa nào đó là *đơn giản nhất*,
và ta có thể đo độ phức tạp
của một hàm bằng khoảng cách các tham số của nó đến không.
Nhưng ta nên đo chính xác như thế nào
khoảng cách giữa một hàm và không?
Không có một câu trả lời đúng duy nhất.
Thực tế, toàn bộ các nhánh toán học,
bao gồm các phần của phân tích hàm
và lý thuyết không gian Banach,
được dành để giải quyết các vấn đề như vậy.

Một diễn giải đơn giản có thể là
đo độ phức tạp của một hàm tuyến tính
$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$
bằng một vài chuẩn của vector trọng số, ví dụ $\| \mathbf{w} \|^2$.
Hãy nhớ lại rằng ta đã giới thiệu chuẩn $\ell_2$ và chuẩn $\ell_1$,
là các trường hợp đặc biệt của chuẩn $\ell_p$ tổng quát hơn,
trong [subsec_lin-algebra-norms](#subsec_lin-algebra-norms).
Phương pháp phổ biến nhất để đảm bảo vector trọng số nhỏ
là thêm chuẩn của nó như một số hạng phạt
vào bài toán tối thiểu hóa hàm mất mát.
Do đó ta thay thế mục tiêu ban đầu,
*tối thiểu hóa mất mát dự đoán trên nhãn huấn luyện*,
bằng mục tiêu mới,
*tối thiểu hóa tổng của mất mát dự đoán và số hạng phạt*.
Bây giờ, nếu vector trọng số của ta lớn quá,
thuật toán học của ta có thể tập trung
vào tối thiểu hóa chuẩn trọng số $\| \mathbf{w} \|^2$
hơn là tối thiểu hóa sai số huấn luyện.
Đó chính xác là điều ta muốn.
Để minh họa trong code,
ta hồi sinh ví dụ trước của ta
từ [sec_linear_regression](#sec_linear_regression) cho hồi quy tuyến tính.
Ở đó, hàm mất mát của ta được cho bởi

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Hãy nhớ rằng $\mathbf{x}^{(i)}$ là các đặc trưng,
$y^{(i)}$ là nhãn cho bất kỳ mẫu dữ liệu $i$ nào, và $(\mathbf{w}, b)$
lần lượt là các tham số trọng số và hệ số chặn.
Để phạt kích thước của vector trọng số,
ta phải bằng cách nào đó thêm $\| \mathbf{w} \|^2$ vào hàm mất mát,
nhưng mô hình nên đánh đổi mất mát tiêu chuẩn
với số hạng phạt cộng thêm mới này như thế nào?
Trong thực tế, ta đặc trưng hóa sự đánh đổi này
thông qua *hằng số chuẩn hóa* $\lambda$,
một siêu tham số không âm
mà ta khớp bằng dữ liệu kiểm định:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$


Với $\lambda = 0$, ta phục hồi hàm mất mát ban đầu.
Với $\lambda > 0$, ta hạn chế kích thước của $\| \mathbf{w} \|$.
Ta chia cho $2$ theo quy ước:
khi ta lấy đạo hàm của một hàm bậc hai,
$2$ và $1/2$ triệt tiêu nhau, đảm bảo rằng biểu thức
cho phép cập nhật trông đẹp và đơn giản.
Độc giả sắc sảo có thể tự hỏi tại sao ta làm việc với chuẩn bình phương
chứ không phải chuẩn tiêu chuẩn (tức là khoảng cách Euclid).
Ta làm điều này để tiện tính toán.
Bằng cách bình phương chuẩn $\ell_2$, ta loại bỏ căn bậc hai,
để lại tổng bình phương của
từng thành phần của vector trọng số.
Điều này làm cho đạo hàm của số hạng phạt dễ tính:
tổng các đạo hàm bằng đạo hàm của tổng.


Hơn nữa, bạn có thể hỏi tại sao ta làm việc với chuẩn $\ell_2$
trước hết chứ không phải, chẳng hạn, chuẩn $\ell_1$.
Thực ra, các lựa chọn khác cũng hợp lệ và
phổ biến trong thống kê.
Trong khi các mô hình tuyến tính được chuẩn hóa $\ell_2$ tạo nên
thuật toán *hồi quy ridge* cổ điển,
hồi quy tuyến tính được chuẩn hóa $\ell_1$
là một phương pháp cơ bản tương tự trong thống kê,
thường được biết đến là *hồi quy lasso*.
Một lý do để làm việc với chuẩn $\ell_2$
là nó đặt hình phạt lớn
lên các thành phần lớn của vector trọng số.
Điều này làm lệch thuật toán học của ta
hướng đến các mô hình phân phối trọng số đều đặn
trên nhiều đặc trưng hơn.
Trong thực tế, điều này có thể làm cho chúng mạnh mẽ hơn
với sai số đo lường trong một biến đơn lẻ.
Ngược lại, hình phạt $\ell_1$ dẫn đến các mô hình
tập trung trọng số vào một tập nhỏ các đặc trưng
bằng cách đặt các trọng số khác về không.
Điều này cho ta một phương pháp hiệu quả để *lựa chọn đặc trưng*,
điều có thể mong muốn vì các lý do khác.
Chẳng hạn, nếu mô hình của ta chỉ dựa vào một vài đặc trưng,
thì ta có thể không cần thu thập, lưu trữ, hoặc truyền dữ liệu
cho các đặc trưng khác (bị loại bỏ).

Sử dụng ký hiệu tương tự trong :eqref:`eq_linreg_batch_update`,
các cập nhật minibatch stochastic gradient descent
cho hồi quy được chuẩn hóa $\ell_2$ như sau:

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

Như trước, ta cập nhật $\mathbf{w}$ dựa trên lượng
mà ước lượng của ta khác với quan sát.
Tuy nhiên, ta cũng thu nhỏ kích thước của $\mathbf{w}$ về không.
Đó là lý do tại sao phương pháp đôi khi được gọi là "suy giảm trọng số":
với số hạng phạt một mình,
thuật toán tối ưu hóa của ta *phân rã*
trọng số ở mỗi bước huấn luyện.
Trái ngược với lựa chọn đặc trưng,
suy giảm trọng số cung cấp cho ta một cơ chế để liên tục điều chỉnh độ phức tạp của hàm.
Các giá trị nhỏ hơn của $\lambda$ tương ứng
với $\mathbf{w}$ ít bị hạn chế hơn,
trong khi các giá trị lớn hơn của $\lambda$
hạn chế $\mathbf{w}$ nhiều hơn.
Việc ta có bao gồm hình phạt hệ số chặn tương ứng $b^2$ hay không
có thể thay đổi giữa các cài đặt,
và có thể thay đổi giữa các lớp của mạng nơ-ron.
Thường ta không chuẩn hóa số hạng hệ số chặn.
Ngoài ra,
mặc dù chuẩn hóa $\ell_2$ có thể không tương đương với suy giảm trọng số đối với các thuật toán tối ưu hóa khác,
ý tưởng chuẩn hóa thông qua
thu nhỏ kích thước trọng số
vẫn đúng.

## Hồi quy Tuyến tính Nhiều chiều

Ta có thể minh họa lợi ích của suy giảm trọng số
thông qua một ví dụ tổng hợp đơn giản.

Đầu tiên, ta [**tạo một số dữ liệu như trước**]:

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ trong đó }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

Trong tập dữ liệu tổng hợp này, nhãn của ta được cho
bởi một hàm tuyến tính nền tảng của đầu vào,
bị nhiễm loạn bởi nhiễu Gauss
với trung bình không và độ lệch chuẩn 0.01.
Để minh họa,
ta có thể làm cho các hiệu ứng quá khớp rõ rệt,
bằng cách tăng chiều số của bài toán lên $d = 200$
và làm việc với một tập huấn luyện nhỏ chỉ 20 mẫu.

```python
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        if tab.selected('jax'):
            self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
            noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## Cài đặt từ Đầu

Bây giờ, hãy thử cài đặt suy giảm trọng số từ đầu.
Vì minibatch stochastic gradient descent
là bộ tối ưu hóa của ta,
ta chỉ cần thêm số hạng phạt bình phương $\ell_2$
vào hàm mất mát ban đầu.

### (**Định nghĩa Phạt Chuẩn $\ell_2$**)

Cách thuận tiện nhất để cài đặt số hạng phạt này
có lẽ là bình phương tất cả các số hạng tại chỗ và tính tổng chúng.

```python
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### Định nghĩa Mô hình

Trong mô hình cuối cùng,
hồi quy tuyến tính và mất mát bình phương không thay đổi kể từ [sec_linear_scratch](#sec_linear_scratch),
vì vậy ta sẽ chỉ định nghĩa một lớp con của `d2l.LinearRegressionScratch`. Thay đổi duy nhất ở đây là hàm mất mát của ta bây giờ bao gồm số hạng phạt.


Code sau khớp mô hình trên tập huấn luyện với 20 mẫu và đánh giá nó trên tập kiểm định với 100 mẫu.

```python
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        print('L2 norm of w:', float(l2_penalty(model.w)))
    if tab.selected('jax'):
        print('L2 norm of w:',
              float(l2_penalty(trainer.state.params['w'])))
```

### [**Huấn luyện không có Chuẩn hóa**]

Ta chạy code này với `lambd = 0`,
tắt suy giảm trọng số.
Lưu ý rằng ta quá khớp tệ,
giảm sai số huấn luyện nhưng không giảm
sai số kiểm định---một trường hợp điển hình của quá khớp.

```python
train_scratch(0)
```

### [**Sử dụng Suy giảm Trọng số**]

Dưới đây, ta chạy với suy giảm trọng số đáng kể.
Lưu ý rằng sai số huấn luyện tăng
nhưng sai số kiểm định giảm.
Đây chính xác là hiệu ứng
ta mong đợi từ chuẩn hóa.

```python
train_scratch(3)
```

## [**Cài đặt Gọn**]

Vì suy giảm trọng số có mặt khắp nơi
trong tối ưu hóa mạng nơ-ron,
framework deep learning làm cho nó đặc biệt thuận tiện,
tích hợp suy giảm trọng số vào thuật toán tối ưu hóa
để dễ dàng sử dụng kết hợp với bất kỳ hàm mất mát nào.
Hơn nữa, sự tích hợp này mang lại lợi ích tính toán,
cho phép các thủ thuật cài đặt thêm suy giảm trọng số vào thuật toán,
mà không có bất kỳ chi phí tính toán bổ sung nào.
Vì phần suy giảm trọng số của phép cập nhật
chỉ phụ thuộc vào giá trị hiện tại của từng tham số,
bộ tối ưu hóa phải chạm vào từng tham số một lần dù sao.


Dưới đây, ta chỉ định
siêu tham số suy giảm trọng số trực tiếp
thông qua `weight_decay` khi khởi tạo bộ tối ưu hóa.
Theo mặc định, PyTorch phân rã cả
trọng số và hệ số chặn đồng thời, nhưng
ta có thể cấu hình bộ tối ưu hóa để xử lý các tham số khác nhau
theo các chính sách khác nhau.
Ở đây, ta chỉ đặt `weight_decay` cho
trọng số (các tham số `net.weight`), do đó
hệ số chặn (tham số `net.bias`) sẽ không phân rã.


```python
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```


[**Đồ thị trông tương tự như khi
ta cài đặt suy giảm trọng số từ đầu**].
Tuy nhiên, phiên bản này chạy nhanh hơn
và dễ cài đặt hơn,
những lợi ích này sẽ trở nên rõ ràng hơn
khi bạn giải quyết các bài toán lớn hơn
và công việc này trở nên thường xuyên hơn.

```python
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

if tab.selected('jax'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

Cho đến nay, ta đã đề cập đến một khái niệm về
những gì tạo nên một hàm tuyến tính đơn giản.
Tuy nhiên, ngay cả với các hàm phi tuyến đơn giản, tình huống có thể phức tạp hơn nhiều. Để thấy điều này, khái niệm [không gian Hilbert nhân tái tạo (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
cho phép áp dụng các công cụ được giới thiệu
cho các hàm tuyến tính trong bối cảnh phi tuyến.
Thật không may, các thuật toán dựa trên RKHS
có xu hướng mở rộng kém với dữ liệu lớn, nhiều chiều.
Trong cuốn sách này ta thường sẽ áp dụng heuristic phổ biến
trong đó suy giảm trọng số được áp dụng
cho tất cả các lớp của mạng sâu.

## Tóm tắt

Chuẩn hóa là một phương pháp phổ biến để xử lý quá khớp. Các kỹ thuật chuẩn hóa cổ điển thêm một số hạng phạt vào hàm mất mát (khi huấn luyện) để giảm độ phức tạp của mô hình được học.
Một lựa chọn cụ thể để giữ mô hình đơn giản là sử dụng hình phạt $\ell_2$. Điều này dẫn đến suy giảm trọng số trong các bước cập nhật của thuật toán minibatch stochastic gradient descent.
Trong thực tế, chức năng suy giảm trọng số được cung cấp trong các bộ tối ưu hóa từ các framework deep learning.
Các tập tham số khác nhau có thể có các hành vi cập nhật khác nhau trong cùng một vòng lặp huấn luyện.


## Bài tập

1. Thử nghiệm với giá trị của $\lambda$ trong bài toán ước lượng trong phần này. Vẽ độ chính xác huấn luyện và kiểm định như là hàm của $\lambda$. Bạn quan sát được gì?
1. Dùng tập kiểm định để tìm giá trị tối ưu của $\lambda$. Đó có thực sự là giá trị tối ưu không? Điều này có quan trọng không?
1. Các phương trình cập nhật sẽ trông như thế nào nếu thay vì $\|\mathbf{w}\|^2$ ta dùng $\sum_i |w_i|$ như hình phạt ưa thích (chuẩn hóa $\ell_1$)?
1. Ta biết rằng $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Bạn có thể tìm phương trình tương tự cho ma trận không (xem chuẩn Frobenius trong [subsec_lin-algebra-norms](#subsec_lin-algebra-norms))?
1. Xem lại mối quan hệ giữa sai số huấn luyện và sai số tổng quát hóa. Ngoài suy giảm trọng số, tăng cường huấn luyện, và sử dụng mô hình có độ phức tạp phù hợp, các cách nào khác có thể giúp ta xử lý quá khớp?
1. Trong thống kê Bayesian ta dùng tích của tiên nghiệm và hàm hợp lý để đến được hậu nghiệm thông qua $P(w \mid x) \propto P(x \mid w) P(w)$. Bạn có thể xác định $P(w)$ với chuẩn hóa như thế nào?


[Thảo luận](https://discuss.d2l.ai/t/99)

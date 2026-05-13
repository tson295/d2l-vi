# Cài đặt Hồi quy Tuyến tính từ Đầu
<a id="sec_linear_scratch"></a>

Bây giờ ta đã sẵn sàng để xây dựng
một cài đặt đầy đủ chức năng
của hồi quy tuyến tính.
Trong phần này,
(**ta sẽ cài đặt toàn bộ phương pháp từ đầu,
bao gồm (i) mô hình; (ii) hàm mất mát;
(iii) bộ tối ưu hóa minibatch stochastic gradient descent;
và (iv) hàm huấn luyện
kết hợp tất cả các thành phần này lại với nhau.**)
Cuối cùng, ta sẽ chạy bộ sinh dữ liệu tổng hợp
từ [sec_synthetic-regression-data](#sec_synthetic-regression-data)
và áp dụng mô hình của ta
lên tập dữ liệu kết quả.
Trong khi các framework deep learning hiện đại
có thể tự động hóa hầu hết công việc này,
cài đặt mọi thứ từ đầu là cách duy nhất
để đảm bảo bạn thực sự hiểu mình đang làm gì.
Hơn nữa, khi đến lúc tùy chỉnh mô hình,
định nghĩa các lớp hay hàm mất mát của riêng mình,
việc hiểu cách hoạt động bên dưới sẽ rất hữu ích.
Trong phần này, ta chỉ dựa vào
tensor và vi phân tự động.
Về sau, ta sẽ giới thiệu cài đặt gọn hơn,
tận dụng các tính năng của framework deep learning
trong khi vẫn giữ nguyên cấu trúc như dưới đây.


```python
%matplotlib inline
from d2l import torch as d2l
import torch
```


## Định nghĩa Mô hình

[**Trước khi có thể bắt đầu tối ưu hóa các tham số mô hình**] bằng minibatch SGD,
(**ta cần có một vài tham số để bắt đầu.**)
Dưới đây ta khởi tạo trọng số bằng cách rút ra
các số ngẫu nhiên từ phân phối chuẩn với trung bình 0
và độ lệch chuẩn 0.01.
Con số kỳ diệu 0.01 thường hoạt động tốt trong thực tế,
nhưng bạn có thể chỉ định một giá trị khác
thông qua đối số `sigma`.
Ngoài ra ta đặt hệ số chặn bằng 0.
Lưu ý rằng để thiết kế hướng đối tượng
ta thêm code vào phương thức `__init__` của lớp con của `d2l.Module` (giới thiệu trong [subsec_oo-design-models](#subsec_oo-design-models)).


Tiếp theo ta phải [**định nghĩa mô hình của mình,
liên kết đầu vào và tham số với đầu ra.**]
Sử dụng cùng ký hiệu như :eqref:`eq_linreg-y-vec`
cho mô hình tuyến tính, ta đơn giản lấy tích ma trận--vector
của đặc trưng đầu vào $\mathbf{X}$
và trọng số mô hình $\mathbf{w}$,
và thêm độ lệch $b$ vào mỗi mẫu.
Tích $\mathbf{Xw}$ là một vector và $b$ là một vô hướng.
Nhờ cơ chế phát tán
(xem [subsec_broadcasting](#subsec_broadcasting)),
khi ta cộng một vector và một vô hướng,
vô hướng được cộng vào mỗi thành phần của vector.
Phương thức `forward` kết quả
được đăng ký trong lớp `LinearRegressionScratch`
thông qua `add_to_class` (giới thiệu trong [oo-design-utilities](#oo-design-utilities)).

```python
@d2l.add_to_class(LinearRegressionScratch)  
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## Định nghĩa Hàm Mất mát

Vì [**cập nhật mô hình yêu cầu lấy
gradient của hàm mất mát,**]
ta nên (**định nghĩa hàm mất mát trước.**]
Ở đây ta dùng hàm mất mát bình phương
trong :eqref:`eq_mse`.
Trong cài đặt, ta cần biến đổi giá trị thực `y`
thành shape của giá trị dự đoán `y_hat`.
Kết quả được trả về bởi phương thức sau
cũng sẽ có cùng shape với `y_hat`.
Ta cũng trả về giá trị mất mát trung bình
trong tất cả các mẫu trong minibatch.


## Định nghĩa Thuật toán Tối ưu hóa

Như đã thảo luận trong [sec_linear_regression](#sec_linear_regression),
hồi quy tuyến tính có nghiệm dạng đóng.
Tuy nhiên, mục tiêu ở đây là minh họa
cách huấn luyện các mạng nơ-ron tổng quát hơn,
và điều đó đòi hỏi ta phải chỉ cho bạn
cách dùng minibatch SGD.
Do đó ta sẽ tận dụng cơ hội này
để giới thiệu ví dụ hoạt động đầu tiên của SGD.
Ở mỗi bước, sử dụng một minibatch
được rút ngẫu nhiên từ tập dữ liệu,
ta ước lượng gradient của hàm mất mát
theo các tham số.
Tiếp theo, ta cập nhật các tham số
theo hướng có thể giảm hàm mất mát.

Code sau áp dụng phép cập nhật,
cho một tập tham số và tốc độ học `lr`.
Vì hàm mất mát của ta được tính là trung bình trên minibatch,
ta không cần điều chỉnh tốc độ học theo kích thước batch.
Trong các chương sau ta sẽ nghiên cứu
cách tốc độ học nên được điều chỉnh
cho các minibatch rất lớn khi chúng xuất hiện
trong học phân tán quy mô lớn.
Hiện tại, ta có thể bỏ qua sự phụ thuộc này.


Ta định nghĩa lớp `SGD`,
một lớp con của `d2l.HyperParameters` (giới thiệu trong [oo-design-utilities](#oo-design-utilities)),
để có API tương tự
như bộ tối ưu hóa SGD tích hợp.
Ta cập nhật các tham số trong phương thức `step`.
Phương thức `zero_grad` đặt tất cả gradient về 0,
phải được chạy trước mỗi bước lan truyền ngược.


Tiếp theo ta định nghĩa phương thức `configure_optimizers`, trả về một thực thể của lớp `SGD`.

```python
@d2l.add_to_class(LinearRegressionScratch)  
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## Huấn luyện

Bây giờ ta đã có tất cả các thành phần
(tham số, hàm mất mát, mô hình và bộ tối ưu hóa),
ta đã sẵn sàng để [**cài đặt vòng lặp huấn luyện chính.**]
Điều quan trọng là bạn phải hiểu đầy đủ code này
vì bạn sẽ sử dụng các vòng lặp huấn luyện tương tự
cho mọi mô hình deep learning khác
được đề cập trong cuốn sách này.
Trong mỗi *epoch*, ta lặp qua
toàn bộ tập dữ liệu huấn luyện,
duyệt qua mỗi mẫu một lần
(giả sử rằng số lượng mẫu
chia hết cho kích thước batch).
Trong mỗi *vòng lặp*, ta lấy một minibatch các mẫu huấn luyện,
và tính hàm mất mát của nó thông qua phương thức `training_step` của mô hình.
Sau đó ta tính gradient theo mỗi tham số.
Cuối cùng, ta sẽ gọi thuật toán tối ưu hóa
để cập nhật các tham số mô hình.
Tóm lại, ta sẽ thực thi vòng lặp sau:

* Khởi tạo tham số $(\mathbf{w}, b)$
* Lặp đến khi kết thúc
    * Tính gradient $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Cập nhật tham số $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

Nhớ lại rằng tập dữ liệu hồi quy tổng hợp
mà ta tạo ra trong [sec_synthetic-regression-data](#sec_synthetic-regression-data)
không cung cấp tập dữ liệu kiểm định.
Tuy nhiên, trong hầu hết các trường hợp,
ta sẽ muốn có một tập kiểm định
để đo chất lượng mô hình.
Ở đây ta duyệt qua dataloader kiểm định
một lần trong mỗi epoch để đo hiệu suất mô hình.
Theo thiết kế hướng đối tượng của ta,
các phương thức `prepare_batch` và `fit_epoch`
được đăng ký trong lớp `d2l.Trainer`
(giới thiệu trong [oo-design-training](#oo-design-training)).

```python
@d2l.add_to_class(d2l.Trainer)  
def prepare_batch(self, batch):
    return batch
```

```python
@d2l.add_to_class(d2l.Trainer)  
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```


Ta gần như đã sẵn sàng để huấn luyện mô hình,
nhưng trước tiên ta cần một số dữ liệu huấn luyện.
Ở đây ta dùng lớp `SyntheticRegressionData`
và truyền vào một số tham số thực tế.
Sau đó ta huấn luyện mô hình với
tốc độ học `lr=0.03`
và đặt `max_epochs=3`.
Lưu ý rằng nói chung, cả số lượng epoch
và tốc độ học đều là siêu tham số.
Nói chung, việc đặt siêu tham số khá phức tạp
và ta thường muốn dùng phân chia ba hướng,
một tập cho huấn luyện,
một tập thứ hai để chọn siêu tham số,
và tập thứ ba dành cho đánh giá cuối cùng.
Ta bỏ qua các chi tiết này hiện tại nhưng sẽ xem lại chúng
sau.

```python
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Vì ta đã tự tạo tập dữ liệu,
ta biết chính xác các tham số thực sự là gì.
Do đó, ta có thể [**đánh giá sự thành công trong việc huấn luyện
bằng cách so sánh các tham số thực sự
với những tham số ta đã học được**] thông qua vòng lặp huấn luyện.
Thật ra chúng hóa ra rất gần nhau.

```python
with torch.no_grad():
    print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```


Ta không nên coi khả năng khôi phục chính xác
các tham số thực tế là điều hiển nhiên.
Nói chung, với các mô hình sâu, không tồn tại
nghiệm duy nhất cho các tham số,
và ngay cả với các mô hình tuyến tính,
khôi phục chính xác các tham số
chỉ có thể khi không có đặc trưng nào
phụ thuộc tuyến tính vào các đặc trưng khác.
Tuy nhiên, trong machine learning,
ta thường ít quan tâm hơn
đến việc khôi phục các tham số nền tảng thực sự,
mà là các tham số
dẫn đến dự đoán chính xác cao [Vapnik.1992].
May mắn thay, ngay cả với các bài toán tối ưu hóa khó,
stochastic gradient descent thường có thể tìm thấy các nghiệm khá tốt,
một phần nhờ vào thực tế là, với các mạng sâu,
tồn tại nhiều cấu hình tham số
dẫn đến dự đoán chính xác cao.


## Tóm tắt

Trong phần này, ta đã thực hiện một bước quan trọng
hướng tới thiết kế các hệ thống deep learning
bằng cách cài đặt một mô hình mạng nơ-ron
và vòng lặp huấn luyện đầy đủ chức năng.
Trong quá trình này, ta đã xây dựng một bộ tải dữ liệu,
một mô hình, một hàm mất mát, một quy trình tối ưu hóa,
và một công cụ trực quan hóa và giám sát.
Ta đã làm điều này bằng cách kết hợp một đối tượng Python
chứa tất cả các thành phần liên quan để huấn luyện mô hình.
Mặc dù đây chưa phải là cài đặt chuyên nghiệp
nhưng nó hoàn toàn hoạt động được và code như thế này
có thể giúp bạn giải quyết các bài toán nhỏ nhanh chóng.
Trong các phần tiếp theo, ta sẽ thấy cách làm điều này
vừa *gọn hơn* (tránh code mẫu)
vừa *hiệu quả hơn* (tận dụng tối đa GPU của ta).


## Bài tập

1. Điều gì sẽ xảy ra nếu ta khởi tạo trọng số bằng không. Thuật toán có còn hoạt động không? Nếu ta
   khởi tạo các tham số với phương sai $1000$ thay vì $0.01$?
1. Giả sử bạn là [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) đang cố gắng xây dựng
   một mô hình về điện trở liên quan điện áp và dòng điện. Bạn có thể sử dụng vi phân
   tự động để học các tham số của mô hình không?
1. Bạn có thể dùng [Định luật Planck](https://en.wikipedia.org/wiki/Planck%27s_law) để xác định nhiệt độ của một vật
   bằng mật độ năng lượng quang phổ không? Để tham khảo, mật độ quang phổ $B$ của bức xạ phát ra từ vật đen là
   $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$. Ở đây
   $\lambda$ là bước sóng, $T$ là nhiệt độ, $c$ là tốc độ ánh sáng, $h$ là hằng số Planck, và $k$ là
   hằng số Boltzmann. Bạn đo năng lượng cho các bước sóng khác nhau $\lambda$ và bây giờ cần khớp
   đường cong mật độ quang phổ với định luật Planck.
1. Những vấn đề bạn có thể gặp nếu muốn tính đạo hàm bậc hai của hàm mất mát là gì? Bạn sẽ
   sửa chúng như thế nào?
1. Tại sao phương thức `reshape` lại cần thiết trong hàm `loss`?
1. Thử nghiệm với các tốc độ học khác nhau để xem giá trị hàm mất mát giảm nhanh như thế nào. Bạn có thể giảm
   sai số bằng cách tăng số epoch huấn luyện không?
1. Nếu số lượng mẫu không chia hết cho kích thước batch, điều gì xảy ra với `data_iter` ở cuối một epoch?
1. Thử cài đặt một hàm mất mát khác, chẳng hạn như mất mát giá trị tuyệt đối `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`.
    1. Kiểm tra điều gì xảy ra với dữ liệu thông thường.
    1. Kiểm tra xem có sự khác biệt trong hành vi không nếu bạn cố tình làm nhiễu một số phần tử, chẳng hạn $y_5 = 10000$, của $\mathbf{y}$.
    1. Bạn có thể nghĩ ra một giải pháp rẻ tiền để kết hợp những ưu điểm tốt nhất của mất mát bình phương và mất mát giá trị tuyệt đối không?
       Gợi ý: làm thế nào để tránh các giá trị gradient rất lớn?
1. Tại sao ta cần xáo trộn lại tập dữ liệu? Bạn có thể thiết kế một trường hợp mà tập dữ liệu được xây dựng độc ác sẽ phá vỡ thuật toán tối ưu hóa không?


[Thảo luận](https://discuss.d2l.ai/t/43)

# Ổn định Số và Khởi tạo
<a id="sec_numerical_stability"></a>


Cho đến nay, mọi mô hình chúng ta cài đặt
đều yêu cầu khởi tạo tham số
theo một phân phối được xác định trước.
Cho đến bây giờ, chúng ta coi sơ đồ khởi tạo là điều hiển nhiên,
bỏ qua chi tiết về cách các lựa chọn này được thực hiện.
Bạn thậm chí có thể có ấn tượng rằng những lựa chọn này
không đặc biệt quan trọng.
Ngược lại, sự lựa chọn sơ đồ khởi tạo
đóng vai trò quan trọng trong việc học mạng nơ-ron,
và nó có thể là yếu tố quyết định để duy trì ổn định số.
Hơn nữa, những lựa chọn này có thể gắn kết theo những cách thú vị
với việc lựa chọn hàm kích hoạt phi tuyến.
Hàm nào chúng ta chọn và cách chúng ta khởi tạo tham số
có thể quyết định thuật toán tối ưu hóa của chúng ta hội tụ nhanh như thế nào.
Những lựa chọn kém ở đây có thể khiến chúng ta gặp phải
gradient bùng nổ hoặc gradient tiêu biến trong quá trình huấn luyện.
Trong phần này, chúng ta đi sâu vào những chủ đề này một cách chi tiết hơn
và thảo luận về một số heuristic hữu ích
mà bạn sẽ thấy có ích
trong suốt sự nghiệp deep learning của mình.


```python
%matplotlib inline
from d2l import torch as d2l
import torch
```


## Gradient Tiêu biến và Gradient Bùng nổ

Xét một mạng sâu với $L$ lớp,
đầu vào $\mathbf{x}$ và đầu ra $\mathbf{o}$.
Với mỗi lớp $l$ được định nghĩa bởi một phép biến đổi $f_l$
được tham số hóa bởi trọng số $\mathbf{W}^{(l)}$,
có đầu ra lớp ẩn là $\mathbf{h}^{(l)}$ (đặt $\mathbf{h}^{(0)} = \mathbf{x}$),
mạng của chúng ta có thể được biểu diễn như sau:

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ và do đó } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$

Nếu tất cả đầu ra lớp ẩn và đầu vào là các vectơ,
chúng ta có thể viết gradient của $\mathbf{o}$ theo
bất kỳ tập tham số $\mathbf{W}^{(l)}$ nào như sau:

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$

Nói cách khác, gradient này là
tích của $L-l$ ma trận
$\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$
và vectơ gradient $\mathbf{v}^{(l)}$.
Do đó chúng ta dễ gặp phải cùng
vấn đề tràn số dưới (numerical underflow) thường xuất hiện
khi nhân quá nhiều xác suất với nhau.
Khi xử lý xác suất, một mẹo phổ biến là
chuyển sang không gian log, tức là dịch chuyển
áp lực từ phần định trị sang phần số mũ
của biểu diễn số.
Thật không may, vấn đề của chúng ta ở trên nghiêm trọng hơn:
ban đầu các ma trận $\mathbf{M}^{(l)}$ có thể có nhiều loại trị riêng khác nhau.
Chúng có thể nhỏ hoặc lớn, và
tích của chúng có thể là *rất lớn* hoặc *rất nhỏ*.

Những rủi ro do gradient không ổn định
vượt ra ngoài biểu diễn số.
Gradient có độ lớn không thể đoán trước
cũng đe dọa sự ổn định của các thuật toán tối ưu hóa của chúng ta.
Chúng ta có thể phải đối mặt với các cập nhật tham số
(i) quá lớn, phá hủy mô hình của chúng ta
(vấn đề *gradient bùng nổ*);
hoặc (ii) quá nhỏ
(vấn đề *gradient tiêu biến*),
khiến việc học không thể thực hiện được vì các tham số
hầu như không di chuyển ở mỗi lần cập nhật.


### (**Gradient Tiêu biến**)

Một nguyên nhân thường gặp gây ra vấn đề gradient tiêu biến
là sự lựa chọn hàm kích hoạt $\sigma$
được thêm vào sau các phép toán tuyến tính của mỗi lớp.
Về mặt lịch sử, hàm sigmoid
$1/(1 + \exp(-x))$ (được giới thiệu trong [sec_mlp](#sec_mlp))
rất phổ biến vì nó giống một hàm ngưỡng.
Vì các mạng nơ-ron nhân tạo ban đầu được lấy cảm hứng
từ các mạng nơ-ron sinh học,
ý tưởng về các nơ-ron kích hoạt *hoàn toàn* hoặc *không kích hoạt*
(như nơ-ron sinh học) có vẻ hấp dẫn.
Hãy xem xét kỹ hơn hàm sigmoid
để hiểu tại sao nó có thể gây ra gradient tiêu biến.


```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```


Như bạn có thể thấy, (**gradient của hàm sigmoid tiêu biến
cả khi đầu vào lớn và khi đầu vào nhỏ**).
Hơn nữa, khi lan truyền ngược qua nhiều lớp,
trừ khi chúng ta ở trong vùng Goldilocks, nơi
đầu vào của nhiều hàm sigmoid gần bằng không,
gradient của tích tổng thể có thể tiêu biến.
Khi mạng của chúng ta có nhiều lớp,
trừ khi chúng ta cẩn thận, gradient
có thể bị cắt đứt ở một lớp nào đó.
Thực vậy, vấn đề này đã từng ảnh hưởng đến việc huấn luyện mạng sâu.
Do đó, ReLU, ổn định hơn
(nhưng ít giống nơ-ron sinh học hơn),
đã nổi lên như lựa chọn mặc định cho các chuyên gia.


### [**Gradient Bùng nổ**]

Vấn đề ngược lại, khi gradient bùng nổ,
cũng có thể gây phiền toái tương tự.
Để minh họa điều này rõ hơn,
chúng ta lấy 100 ma trận ngẫu nhiên Gaussian
và nhân chúng với một ma trận ban đầu nào đó.
Với tỉ lệ mà chúng ta chọn
(lựa chọn phương sai $\sigma^2=1$),
tích ma trận bùng nổ.
Khi điều này xảy ra do khởi tạo
của một mạng sâu, chúng ta không có cơ hội
để thuật toán tối ưu hóa gradient descent hội tụ.


```python
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
```


### Phá vỡ Tính Đối xứng

Một vấn đề khác trong thiết kế mạng nơ-ron
là tính đối xứng vốn có trong cách tham số hóa của chúng.
Giả sử chúng ta có một MLP đơn giản
với một lớp ẩn và hai đơn vị.
Trong trường hợp này, chúng ta có thể hoán vị các trọng số $\mathbf{W}^{(1)}$
của lớp đầu tiên và tương tự hoán vị
các trọng số của lớp đầu ra
để thu được cùng một hàm.
Không có gì đặc biệt phân biệt
đơn vị ẩn thứ nhất và thứ hai.
Nói cách khác, chúng ta có tính đối xứng hoán vị
giữa các đơn vị ẩn của mỗi lớp.

Đây không chỉ là một phiền toái về mặt lý thuyết.
Hãy xét MLP một lớp ẩn đã đề cập
với hai đơn vị ẩn.
Để minh họa,
giả sử lớp đầu ra biến đổi hai đơn vị ẩn thành chỉ một đơn vị đầu ra.
Hãy tưởng tượng điều gì xảy ra nếu chúng ta khởi tạo
tất cả tham số của lớp ẩn
là $\mathbf{W}^{(1)} = c$ với một hằng số $c$ nào đó.
Trong trường hợp này, trong quá trình lan truyền xuôi
mỗi đơn vị ẩn nhận cùng đầu vào và tham số
tạo ra cùng kích hoạt
được đưa vào đơn vị đầu ra.
Trong quá trình lan truyền ngược,
việc lấy vi phân đơn vị đầu ra theo tham số $\mathbf{W}^{(1)}$ cho gradient mà tất cả các phần tử có cùng giá trị.
Do đó, sau khi cập nhật dựa trên gradient (ví dụ: gradient descent ngẫu nhiên theo minibatch),
tất cả các phần tử của $\mathbf{W}^{(1)}$ vẫn có cùng giá trị.
Các lần cập nhật như vậy sẽ
không bao giờ tự *phá vỡ tính đối xứng*
và chúng ta có thể không bao giờ thực hiện được
sức mạnh biểu diễn của mạng.
Lớp ẩn sẽ hoạt động
như thể nó chỉ có một đơn vị duy nhất.
Lưu ý rằng trong khi gradient descent ngẫu nhiên theo minibatch không phá vỡ tính đối xứng này,
chính quy hóa dropout (sẽ được giới thiệu sau) thì có thể!


## Khởi tạo Tham số

Một cách để giải quyết---hoặc ít nhất là giảm thiểu---các
vấn đề nêu trên là thông qua việc khởi tạo cẩn thận.
Như chúng ta sẽ thấy sau,
sự chú ý bổ sung trong quá trình tối ưu hóa
và chính quy hóa phù hợp có thể tăng cường thêm sự ổn định.


### Khởi tạo Mặc định

Trong các phần trước, ví dụ, trong [sec_linear_concise](#sec_linear_concise),
chúng ta đã sử dụng phân phối chuẩn
để khởi tạo các giá trị trọng số.
Nếu chúng ta không chỉ định phương pháp khởi tạo, framework sẽ
sử dụng phương pháp khởi tạo ngẫu nhiên mặc định, thường hoạt động tốt trong thực tế
cho các kích thước bài toán vừa phải.


### Khởi tạo Xavier
<a id="subsec_xavier"></a>

Hãy xem xét phân phối tỉ lệ của
một đầu ra $o_{i}$ cho một lớp kết nối đầy đủ nào đó
*không có phi tuyến tính*.
Với $n_\textrm{in}$ đầu vào $x_j$
và các trọng số liên quan $w_{ij}$ cho lớp này,
một đầu ra được cho bởi

$$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$

Các trọng số $w_{ij}$ đều được lấy mẫu độc lập
từ cùng một phân phối.
Hơn nữa, hãy giả sử rằng phân phối này
có giá trị trung bình bằng không và phương sai $\sigma^2$.
Lưu ý rằng điều này không có nghĩa là phân phối phải là Gaussian,
chỉ là giá trị trung bình và phương sai cần tồn tại.
Hiện tại, hãy giả sử rằng đầu vào cho lớp $x_j$
cũng có giá trị trung bình bằng không và phương sai $\gamma^2$
và chúng độc lập với $w_{ij}$ và độc lập với nhau.
Trong trường hợp này, chúng ta có thể tính giá trị trung bình của $o_i$:

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}$$

và phương sai:

$$
\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

Một cách để giữ phương sai cố định
là đặt $n_\textrm{in} \sigma^2 = 1$.
Bây giờ hãy xét lan truyền ngược.
Ở đó chúng ta phải đối mặt với một vấn đề tương tự,
mặc dù các gradient được lan truyền từ các lớp gần đầu ra hơn.
Sử dụng lý luận tương tự như cho lan truyền xuôi,
chúng ta thấy rằng phương sai của gradient có thể bùng nổ
trừ khi $n_\textrm{out} \sigma^2 = 1$,
trong đó $n_\textrm{out}$ là số đầu ra của lớp này.
Điều này đặt chúng ta vào một tình huống khó xử:
chúng ta không thể đồng thời thỏa mãn cả hai điều kiện.
Thay vào đó, chúng ta chỉ đơn giản cố gắng thỏa mãn:

$$
\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ hay tương đương }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$

Đây là lý luận nền tảng của *khởi tạo Xavier* hiện là tiêu chuẩn
và có lợi ích thực tế,
được đặt theo tên tác giả đầu tiên của những người tạo ra nó [Glorot.Bengio.2010].
Thông thường, khởi tạo Xavier
lấy mẫu trọng số từ phân phối Gaussian
với giá trị trung bình bằng không và phương sai
$\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$.
Chúng ta cũng có thể điều chỉnh điều này để
chọn phương sai khi lấy mẫu trọng số
từ phân phối đều.
Lưu ý rằng phân phối đều $U(-a, a)$ có phương sai $\frac{a^2}{3}$.
Thay $\frac{a^2}{3}$ vào điều kiện của chúng ta về $\sigma^2$
nhắc nhở chúng ta khởi tạo theo

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

Mặc dù giả định về sự vắng mặt của phi tuyến tính
trong lý luận toán học trên
có thể dễ dàng bị vi phạm trong mạng nơ-ron,
phương pháp khởi tạo Xavier
hóa ra lại hoạt động tốt trong thực tế.


### Tiếp theo

Lý luận ở trên chỉ mới chạm đến bề mặt
của các phương pháp hiện đại về khởi tạo tham số.
Một framework deep learning thường cài đặt hơn chục heuristic khác nhau.
Hơn nữa, khởi tạo tham số tiếp tục là
một lĩnh vực nghiên cứu cơ bản sôi động trong deep learning.
Trong số đó có các heuristic chuyên biệt cho
các tham số ràng buộc (chia sẻ), siêu phân giải,
mô hình chuỗi, và các tình huống khác.
Ví dụ,
Xiao.Bahri.Sohl-Dickstein.ea.2018 đã chứng minh khả năng huấn luyện
mạng nơ-ron 10,000 lớp mà không cần các mẹo kiến trúc
bằng cách sử dụng phương pháp khởi tạo được thiết kế cẩn thận.

Nếu chủ đề này thu hút bạn, chúng tôi gợi ý
tìm hiểu sâu vào các nội dung của mô-đun này,
đọc các bài báo đề xuất và phân tích từng heuristic,
và sau đó khám phá các ấn phẩm mới nhất về chủ đề này.
Có thể bạn sẽ tình cờ khám phá ra hoặc thậm chí phát minh ra
một ý tưởng thông minh và đóng góp cài đặt vào các framework deep learning.


## Tóm tắt

Gradient tiêu biến và gradient bùng nổ là các vấn đề thường gặp trong mạng sâu. Cần hết sức cẩn thận trong khởi tạo tham số để đảm bảo gradient và tham số được kiểm soát tốt.
Các heuristic khởi tạo cần thiết để đảm bảo rằng gradient ban đầu không quá lớn cũng không quá nhỏ.
Khởi tạo ngẫu nhiên là chìa khóa để đảm bảo tính đối xứng bị phá vỡ trước khi tối ưu hóa.
Khởi tạo Xavier gợi ý rằng, đối với mỗi lớp, phương sai của bất kỳ đầu ra nào không bị ảnh hưởng bởi số lượng đầu vào, và phương sai của bất kỳ gradient nào không bị ảnh hưởng bởi số lượng đầu ra.
Hàm kích hoạt ReLU giảm thiểu vấn đề gradient tiêu biến. Điều này có thể tăng tốc hội tụ.

## Bài tập

1. Bạn có thể thiết kế các trường hợp khác mà một mạng nơ-ron có thể thể hiện tính đối xứng cần phá vỡ, ngoài tính đối xứng hoán vị trong các lớp của MLP không?
1. Chúng ta có thể khởi tạo tất cả tham số trọng số trong hồi quy tuyến tính hoặc hồi quy softmax về cùng một giá trị không?
1. Tra cứu các giới hạn giải tích về trị riêng của tích hai ma trận. Điều này cho bạn biết gì về việc đảm bảo gradient được điều kiện tốt?
1. Nếu chúng ta biết rằng một số số hạng phân kỳ, liệu chúng ta có thể khắc phục điều này sau đó không? Hãy xem bài báo về layerwise adaptive rate scaling để tham khảo [You.Gitman.Ginsburg.2017].


[Discussions](https://discuss.d2l.ai/t/104)

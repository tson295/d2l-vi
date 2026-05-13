# Mạng Nơ-ron Hồi quy
<a id="sec_rnn"></a>


Trong [sec_language-model](#sec_language-model), chúng ta đã mô tả các mô hình Markov và $n$-gram cho mô hình hóa ngôn ngữ, trong đó xác suất có điều kiện của token $x_t$ tại bước thời gian $t$ chỉ phụ thuộc vào $n-1$ token trước đó.
Nếu chúng ta muốn kết hợp tác động có thể có của các token trước bước thời gian $t-(n-1)$ lên $x_t$,
chúng ta cần tăng $n$.
Tuy nhiên, số lượng tham số mô hình cũng sẽ tăng theo cấp số nhân với nó, vì chúng ta cần lưu trữ $|\mathcal{V}|^n$ số cho một tập từ vựng $\mathcal{V}$.
Do đó, thay vì mô hình hóa $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$, tốt hơn là sử dụng một mô hình biến ẩn,

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

trong đó $h_{t-1}$ là một *trạng thái ẩn* lưu trữ thông tin chuỗi đến bước thời gian $t-1$.
Nói chung,
trạng thái ẩn tại bất kỳ bước thời gian $t$ nào có thể được tính toán dựa trên cả đầu vào hiện tại $x_{t}$ và trạng thái ẩn trước đó $h_{t-1}$:

$$h_t = f(x_{t}, h_{t-1}).$$

Đối với một hàm $f$ đủ mạnh trong :eqref:`eq_ht_xt`, mô hình biến ẩn không phải là xấp xỉ. Dù sao, $h_t$ có thể chỉ đơn giản lưu trữ tất cả dữ liệu nó đã quan sát cho đến nay.
Tuy nhiên, điều này có thể làm cho cả tính toán lẫn lưu trữ trở nên tốn kém.

Nhắc lại rằng chúng ta đã thảo luận về các lớp ẩn với các đơn vị ẩn trong [chap_perceptrons](#chap_perceptrons).
Đáng chú ý là
các lớp ẩn và các trạng thái ẩn đề cập đến hai khái niệm rất khác nhau.
Các lớp ẩn là, như đã giải thích, các lớp ẩn khỏi tầm nhìn trên đường từ đầu vào đến đầu ra.
Các trạng thái ẩn về mặt kỹ thuật là *đầu vào* cho bất cứ điều gì chúng ta làm tại một bước nhất định,
và chúng chỉ có thể được tính toán bằng cách xem dữ liệu tại các bước thời gian trước đó.

*Mạng nơ-ron hồi quy* (RNN) là các mạng nơ-ron có trạng thái ẩn. Trước khi giới thiệu mô hình RNN, trước tiên chúng ta xem lại mô hình MLP được giới thiệu trong [sec_mlp](#sec_mlp).


```python
from d2l import torch as d2l
import torch
```


## Mạng Nơ-ron Không có Trạng thái Ẩn

Hãy xem xét một MLP với một lớp ẩn duy nhất.
Cho hàm kích hoạt của lớp ẩn là $\phi$.
Khi có một minibatch các ví dụ $\mathbf{X} \in \mathbb{R}^{n \times d}$ với kích thước batch $n$ và $d$ đầu vào, đầu ra của lớp ẩn $\mathbf{H} \in \mathbb{R}^{n \times h}$ được tính toán là

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{\textrm{xh}} + \mathbf{b}_\textrm{h}).$$

Trong :eqref:`rnn_h_without_state`, chúng ta có tham số trọng số $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$, tham số hệ số chặn $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$, và số đơn vị ẩn $h$, cho lớp ẩn.
Với điều đó, chúng ta áp dụng phát sóng (xem [subsec_broadcasting](#subsec_broadcasting)) trong quá trình tổng.
Tiếp theo, đầu ra của lớp ẩn $\mathbf{H}$ được sử dụng làm đầu vào của lớp đầu ra, được cho bởi

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

trong đó $\mathbf{O} \in \mathbb{R}^{n \times q}$ là biến đầu ra, $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ là tham số trọng số, và $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ là tham số hệ số chặn của lớp đầu ra. Nếu đây là bài toán phân loại, chúng ta có thể sử dụng $\mathrm{softmax}(\mathbf{O})$ để tính phân phối xác suất của các danh mục đầu ra.

Điều này hoàn toàn tương tự với bài toán hồi quy mà chúng ta đã giải trước đây trong [sec_sequence](#sec_sequence), do đó chúng ta bỏ qua chi tiết.
Chỉ cần nói rằng chúng ta có thể chọn các cặp đặc trưng-nhãn một cách ngẫu nhiên và học các tham số của mạng thông qua vi phân tự động và gradient descent ngẫu nhiên.

## Mạng Nơ-ron Hồi quy với Trạng thái Ẩn
<a id="subsec_rnn_w_hidden_states"></a>

Mọi thứ hoàn toàn khác khi chúng ta có trạng thái ẩn. Hãy xem xét cấu trúc chi tiết hơn.

Giả sử rằng chúng ta có
một minibatch đầu vào
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
tại bước thời gian $t$.
Nói cách khác,
với một minibatch $n$ ví dụ chuỗi,
mỗi hàng của $\mathbf{X}_t$ tương ứng với một ví dụ tại bước thời gian $t$ từ chuỗi.
Tiếp theo,
ký hiệu $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ là đầu ra của lớp ẩn tại bước thời gian $t$.
Không giống với MLP, ở đây chúng ta lưu đầu ra lớp ẩn $\mathbf{H}_{t-1}$ từ bước thời gian trước và giới thiệu một tham số trọng số mới $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ để mô tả cách sử dụng đầu ra lớp ẩn của bước thời gian trước trong bước thời gian hiện tại. Cụ thể, việc tính toán đầu ra lớp ẩn của bước thời gian hiện tại được xác định bởi đầu vào của bước thời gian hiện tại cùng với đầu ra lớp ẩn của bước thời gian trước:

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}  + \mathbf{b}_\textrm{h}).$$

So với :eqref:`rnn_h_without_state`, :eqref:`rnn_h_with_state` thêm một số hạng nữa $\mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ và do đó
cụ thể hóa :eqref:`eq_ht_xt`.
Từ mối quan hệ giữa các đầu ra lớp ẩn $\mathbf{H}_t$ và $\mathbf{H}_{t-1}$ của các bước thời gian liền kề,
chúng ta biết rằng các biến này đã nắm bắt và giữ lại thông tin lịch sử của chuỗi đến bước thời gian hiện tại, giống như trạng thái hoặc bộ nhớ của bước thời gian hiện tại của mạng nơ-ron. Do đó, đầu ra lớp ẩn như vậy được gọi là *trạng thái ẩn*.
Vì trạng thái ẩn sử dụng cùng định nghĩa của bước thời gian trước trong bước thời gian hiện tại, việc tính toán :eqref:`rnn_h_with_state` là *hồi quy*. Do đó, như chúng ta đã nói, các mạng nơ-ron với trạng thái ẩn
dựa trên tính toán hồi quy được đặt tên là
*mạng nơ-ron hồi quy*.
Các lớp thực hiện
tính toán của :eqref:`rnn_h_with_state`
trong RNN
được gọi là *các lớp hồi quy*.


Có nhiều cách khác nhau để xây dựng RNN.
Những cái có trạng thái ẩn được định nghĩa bởi :eqref:`rnn_h_with_state` là rất phổ biến.
Tại bước thời gian $t$,
đầu ra của lớp đầu ra tương tự như phép tính trong MLP:

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

Các tham số của RNN
bao gồm các trọng số $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$,
và hệ số chặn $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$
của lớp ẩn,
cùng với các trọng số $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$
và hệ số chặn $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$
của lớp đầu ra.
Đáng đề cập rằng
ngay cả ở các bước thời gian khác nhau,
RNN luôn sử dụng các tham số mô hình này.
Do đó, chi phí tham số hóa của một RNN
không tăng khi số bước thời gian tăng.

[fig_rnn](#fig_rnn) minh họa logic tính toán của một RNN tại ba bước thời gian liền kề.
Tại bất kỳ bước thời gian $t$ nào,
việc tính toán trạng thái ẩn có thể được coi là:
(i) ghép đầu vào $\mathbf{X}_t$ tại bước thời gian hiện tại $t$ và trạng thái ẩn $\mathbf{H}_{t-1}$ tại bước thời gian trước $t-1$;
(ii) đưa kết quả ghép vào một lớp kết nối đầy đủ với hàm kích hoạt $\phi$.
Đầu ra của lớp kết nối đầy đủ như vậy là trạng thái ẩn $\mathbf{H}_t$ của bước thời gian hiện tại $t$.
Trong trường hợp này,
các tham số mô hình là phép ghép của $\mathbf{W}_{\textrm{xh}}$ và $\mathbf{W}_{\textrm{hh}}$, và một hệ số chặn $\mathbf{b}_\textrm{h}$, tất cả từ :eqref:`rnn_h_with_state`.
Trạng thái ẩn của bước thời gian hiện tại $t$, $\mathbf{H}_t$, sẽ tham gia vào việc tính toán trạng thái ẩn $\mathbf{H}_{t+1}$ của bước thời gian tiếp theo $t+1$.
Hơn nữa, $\mathbf{H}_t$ cũng sẽ được
đưa vào lớp đầu ra kết nối đầy đủ
để tính toán đầu ra
$\mathbf{O}_t$ của bước thời gian hiện tại $t$.

![Một RNN với trạng thái ẩn.](../img/rnn.svg)
<a id="fig_rnn"></a>

Chúng ta vừa đề cập rằng phép tính $\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ cho trạng thái ẩn tương đương với
phép nhân ma trận của
phép ghép của $\mathbf{X}_t$ và $\mathbf{H}_{t-1}$
và
phép ghép của $\mathbf{W}_{\textrm{xh}}$ và $\mathbf{W}_{\textrm{hh}}$.
Mặc dù điều này có thể được chứng minh về mặt toán học,
trong phần sau chúng ta chỉ sử dụng một đoạn mã đơn giản để minh chứng.
Để bắt đầu,
chúng ta định nghĩa các ma trận `X`, `W_xh`, `H`, và `W_hh`, có các hình dạng lần lượt là (3, 1), (1, 4), (3, 4), và (4, 4).
Nhân `X` với `W_xh`, và `H` với `W_hh`, sau đó cộng hai tích này,
chúng ta thu được một ma trận có hình dạng (3, 4).


Bây giờ chúng ta ghép các ma trận `X` và `H`
theo các cột (trục 1),
và các ma trận
`W_xh` và `W_hh` theo các hàng (trục 0).
Hai phép ghép này
dẫn đến
các ma trận có hình dạng (3, 5)
và có hình dạng (5, 4), tương ứng.
Nhân hai ma trận được ghép này,
chúng ta thu được cùng ma trận đầu ra có hình dạng (3, 4)
như trên.

```python
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## Mô hình Ngôn ngữ Cấp Ký tự Dựa trên RNN

Nhắc lại rằng đối với mô hình hóa ngôn ngữ trong [sec_language-model](#sec_language-model),
chúng ta nhằm mục đích dự đoán token tiếp theo dựa trên
các token hiện tại và trước đó;
do đó chúng ta dịch chuyển chuỗi gốc đi một token
như các mục tiêu (nhãn).
Bengio.Ducharme.Vincent.ea.2003 lần đầu tiên đề xuất
sử dụng mạng nơ-ron cho mô hình hóa ngôn ngữ.
Trong phần sau, chúng ta minh họa cách RNN có thể được sử dụng để xây dựng một mô hình ngôn ngữ.
Cho kích thước minibatch là một, và chuỗi văn bản là "machine".
Để đơn giản hóa việc huấn luyện trong các phần tiếp theo,
chúng ta token hóa văn bản thành ký tự thay vì từ
và xem xét một *mô hình ngôn ngữ cấp ký tự*.
[fig_rnn_train](#fig_rnn_train) minh họa cách dự đoán ký tự tiếp theo dựa trên các ký tự hiện tại và trước đó thông qua RNN cho mô hình hóa ngôn ngữ cấp ký tự.

![Một mô hình ngôn ngữ cấp ký tự dựa trên RNN. Chuỗi đầu vào và mục tiêu lần lượt là "machin" và "achine".](../img/rnn-train.svg)
<a id="fig_rnn_train"></a>

Trong quá trình huấn luyện,
chúng ta chạy thao tác softmax trên đầu ra từ lớp đầu ra cho mỗi bước thời gian, và sau đó sử dụng mất mát entropy chéo để tính toán sai số giữa đầu ra mô hình và mục tiêu.
Do tính toán hồi quy của trạng thái ẩn trong lớp ẩn, đầu ra, $\mathbf{O}_3$, của bước thời gian 3 trong [fig_rnn_train](#fig_rnn_train) được xác định bởi chuỗi văn bản "m", "a", và "c". Vì ký tự tiếp theo của chuỗi trong dữ liệu huấn luyện là "h", mất mát của bước thời gian 3 sẽ phụ thuộc vào phân phối xác suất của ký tự tiếp theo được tạo dựa trên chuỗi đặc trưng "m", "a", "c" và mục tiêu "h" của bước thời gian này.

Trong thực tế, mỗi token được biểu diễn bởi một vector $d$ chiều, và chúng ta sử dụng kích thước batch $n>1$. Do đó, đầu vào $\mathbf X_t$ tại bước thời gian $t$ sẽ là một ma trận $n\times d$, giống hệt những gì chúng ta đã thảo luận trong [subsec_rnn_w_hidden_states](#subsec_rnn_w_hidden_states).

Trong các phần tiếp theo, chúng ta sẽ triển khai RNN
cho các mô hình ngôn ngữ cấp ký tự.


## Tóm tắt

Một mạng nơ-ron sử dụng tính toán hồi quy cho các trạng thái ẩn được gọi là mạng nơ-ron hồi quy (RNN).
Trạng thái ẩn của RNN có thể nắm bắt thông tin lịch sử của chuỗi đến bước thời gian hiện tại. Với tính toán hồi quy, số tham số mô hình RNN không tăng khi số bước thời gian tăng. Về mặt ứng dụng, RNN có thể được sử dụng để tạo ra các mô hình ngôn ngữ cấp ký tự.


## Bài tập

1. Nếu chúng ta sử dụng RNN để dự đoán ký tự tiếp theo trong một chuỗi văn bản, thứ nguyên cần thiết cho bất kỳ đầu ra nào là gì?
1. Tại sao RNN có thể biểu diễn xác suất có điều kiện của một token tại một bước thời gian nào đó dựa trên tất cả các token trước đó trong chuỗi văn bản?
1. Điều gì xảy ra với gradient nếu bạn lan truyền ngược qua một chuỗi dài?
1. Một số vấn đề liên quan đến mô hình ngôn ngữ được mô tả trong phần này là gì?


[Discussions](https://discuss.d2l.ai/t/1050)

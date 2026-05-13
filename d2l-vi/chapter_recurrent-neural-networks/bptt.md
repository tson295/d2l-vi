# Lan truyền Ngược Qua Thời gian
<a id="sec_bptt"></a>

Nếu bạn đã hoàn thành các bài tập trong [sec_rnn-scratch](#sec_rnn-scratch),
bạn sẽ thấy rằng cắt gradient là điều thiết yếu
để ngăn các gradient khổng lồ thỉnh thoảng xuất hiện
làm mất ổn định quá trình huấn luyện.
Chúng ta đã gợi ý rằng hiện tượng gradient bùng nổ
bắt nguồn từ việc lan truyền ngược qua các chuỗi dài.
Trước khi giới thiệu hàng loạt kiến trúc RNN hiện đại,
hãy xem xét kỹ hơn cách *lan truyền ngược*
hoạt động trong các mô hình chuỗi ở mức chi tiết toán học.
Hy vọng rằng phần thảo luận này sẽ làm rõ hơn
khái niệm gradient *tiêu biến* và *bùng nổ*.
Nếu bạn còn nhớ phần thảo luận của chúng ta về lan truyền xuôi và lan truyền ngược
qua đồ thị tính toán
khi giới thiệu MLP trong [sec_backprop](#sec_backprop),
thì lan truyền xuôi trong RNN
sẽ tương đối đơn giản.
Việc áp dụng lan truyền ngược trong RNN
được gọi là *lan truyền ngược qua thời gian* [Werbos.1990].
Quy trình này yêu cầu chúng ta mở rộng (hoặc trải ra)
đồ thị tính toán của một RNN
từng bước thời gian một.
RNN đã trải ra về cơ bản là
một mạng nơ-ron truyền thẳng
với tính chất đặc biệt
là cùng các tham số
được lặp lại xuyên suốt mạng đã trải ra,
xuất hiện ở mỗi bước thời gian.
Sau đó, giống như trong bất kỳ mạng nơ-ron truyền thẳng nào,
chúng ta có thể áp dụng quy tắc dây chuyền,
lan truyền ngược gradient qua mạng đã trải ra.
Gradient theo từng tham số
phải được cộng trên tất cả các vị trí
mà tham số đó xuất hiện trong mạng đã trải ra.
Việc xử lý sự ràng buộc trọng số như vậy hẳn đã quen thuộc
từ các chương của chúng ta về mạng nơ-ron tích chập.


Các phức tạp nảy sinh vì chuỗi
có thể khá dài.
Làm việc với các chuỗi văn bản
gồm hơn một nghìn token không phải là điều hiếm gặp.
Lưu ý rằng điều này gây ra vấn đề cả từ
góc độ tính toán (quá nhiều bộ nhớ)
lẫn tối ưu hóa (bất ổn số học).
Đầu vào từ bước đầu tiên đi qua
hơn 1000 phép nhân ma trận trước khi đến đầu ra,
và cần thêm 1000 phép nhân ma trận nữa
để tính gradient.
Bây giờ chúng ta phân tích những điều có thể sai
và cách xử lý chúng trong thực tế.


## Phân tích Gradient trong RNN
<a id="subsec_bptt_analysis"></a>

Chúng ta bắt đầu với một mô hình đơn giản hóa về cách RNN hoạt động.
Mô hình này bỏ qua các chi tiết về trạng thái ẩn
và cách nó được cập nhật.
Ký hiệu toán học ở đây
không phân biệt tường minh
giữa vô hướng, vectơ và ma trận.
Chúng ta chỉ đang cố gắng xây dựng một số trực giác.
Trong mô hình đơn giản hóa này,
chúng ta ký hiệu $h_t$ là trạng thái ẩn,
$x_t$ là đầu vào, và $o_t$ là đầu ra
tại bước thời gian $t$.
Nhắc lại các thảo luận của chúng ta trong
[subsec_rnn_w_hidden_states](#subsec_rnn_w_hidden_states)
rằng đầu vào và trạng thái ẩn
có thể được nối lại trước khi được nhân
với một biến trọng số trong lớp ẩn.
Do đó, chúng ta dùng $w_\textrm{h}$ và $w_\textrm{o}$ để chỉ các trọng số
của lớp ẩn và lớp đầu ra, tương ứng.
Kết quả là, các trạng thái ẩn và đầu ra
tại mỗi bước thời gian là

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_\textrm{h}),\\o_t &= g(h_t, w_\textrm{o}),\end{aligned}$$

trong đó $f$ và $g$ lần lượt là các phép biến đổi
của lớp ẩn và lớp đầu ra.
Vì vậy, chúng ta có một chuỗi các giá trị
$\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$
phụ thuộc lẫn nhau thông qua tính toán hồi quy.
Lan truyền xuôi khá đơn giản.
Tất cả những gì cần làm là lặp qua các bộ ba $(x_t, h_t, o_t)$ từng bước thời gian một.
Sai khác giữa đầu ra $o_t$ và mục tiêu mong muốn $y_t$
sau đó được đánh giá bằng một hàm mục tiêu
trên toàn bộ $T$ bước thời gian như sau

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_\textrm{h}, w_\textrm{o}) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$


Đối với lan truyền ngược, vấn đề phức tạp hơn một chút,
đặc biệt khi chúng ta tính gradient
theo các tham số $w_\textrm{h}$ của hàm mục tiêu $L$.
Cụ thể, theo quy tắc dây chuyền,

$$\begin{aligned}\frac{\partial L}{\partial w_\textrm{h}}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_\textrm{h}}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_\textrm{o})}{\partial h_t}  \frac{\partial h_t}{\partial w_\textrm{h}}.\end{aligned}$$

Thừa số thứ nhất và thứ hai của tích
trong :eqref:`eq_bptt_partial_L_wh`
dễ tính.
Thừa số thứ ba $\partial h_t/\partial w_\textrm{h}$ là nơi mọi thứ trở nên khó hơn,
vì chúng ta cần tính một cách hồi quy ảnh hưởng của tham số $w_\textrm{h}$ lên $h_t$.
Theo phép tính hồi quy
trong :eqref:`eq_bptt_ht_ot`,
$h_t$ phụ thuộc vào cả $h_{t-1}$ và $w_\textrm{h}$,
trong đó việc tính $h_{t-1}$
cũng phụ thuộc vào $w_\textrm{h}$.
Do đó, đánh giá đạo hàm toàn phần của $h_t$
theo $w_\textrm{h}$ bằng quy tắc dây chuyền cho ta

$$\frac{\partial h_t}{\partial w_\textrm{h}}= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$


Để suy ra gradient ở trên, giả sử rằng chúng ta có
ba dãy $\{a_{t}\},\{b_{t}\},\{c_{t}\}$
thỏa mãn $a_{0}=0$ và $a_{t}=b_{t}+c_{t}a_{t-1}$ với $t=1, 2,\ldots$.
Khi đó với $t\geq 1$, dễ dàng chứng minh rằng

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$

Bằng cách thay $a_t$, $b_t$, và $c_t$ theo

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_\textrm{h}},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}},\end{aligned}$$

phép tính gradient trong :eqref:`eq_bptt_partial_ht_wh_recur` thỏa mãn
$a_{t}=b_{t}+c_{t}a_{t-1}$.
Do đó, theo :eqref:`eq_bptt_at`,
chúng ta có thể loại bỏ phép tính hồi quy
trong :eqref:`eq_bptt_partial_ht_wh_recur` bằng

$$\frac{\partial h_t}{\partial w_\textrm{h}}=\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_\textrm{h})}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_\textrm{h})}{\partial w_\textrm{h}}.$$

Mặc dù chúng ta có thể dùng quy tắc dây chuyền để tính $\partial h_t/\partial w_\textrm{h}$ một cách đệ quy,
chuỗi này có thể trở nên rất dài khi $t$ lớn.
Hãy thảo luận một số chiến lược để xử lý vấn đề này.

### Tính toán Đầy đủ ###

Một ý tưởng có thể là tính tổng đầy đủ trong :eqref:`eq_bptt_partial_ht_wh_gen`.
Tuy nhiên, cách này rất chậm và gradient có thể bùng nổ,
vì những thay đổi tinh tế trong điều kiện ban đầu
có thể ảnh hưởng rất lớn đến kết quả.
Nói cách khác, chúng ta có thể thấy các hiện tượng tương tự hiệu ứng cánh bướm,
trong đó những thay đổi tối thiểu ở điều kiện ban đầu
dẫn đến những thay đổi không tương xứng ở kết quả.
Điều này nhìn chung là không mong muốn.
Rốt cuộc, chúng ta đang tìm kiếm các bộ ước lượng vững chắc có khả năng khái quát hóa tốt.
Vì vậy, chiến lược này hầu như không bao giờ được dùng trong thực tế.

### Cắt ngắn các Bước Thời gian###

Ngoài ra,
chúng ta có thể cắt ngắn tổng trong
:eqref:`eq_bptt_partial_ht_wh_gen`
sau $\tau$ bước.
Đây là điều chúng ta đã thảo luận cho đến nay.
Điều này dẫn đến một *xấp xỉ* của gradient thật,
đơn giản bằng cách kết thúc tổng tại $\partial h_{t-\tau}/\partial w_\textrm{h}$.
Trong thực tế, cách này hoạt động khá tốt.
Đây là điều thường được gọi là lan truyền ngược qua thời gian
bị cắt ngắn [Jaeger.2002].
Một hệ quả của việc này là mô hình
tập trung chủ yếu vào ảnh hưởng ngắn hạn
thay vì các hệ quả dài hạn.
Điều này thực ra là *mong muốn*, vì nó làm lệch ước lượng
về phía các mô hình đơn giản và ổn định hơn.


### Cắt ngắn Ngẫu nhiên ###

Cuối cùng, chúng ta có thể thay $\partial h_t/\partial w_\textrm{h}$
bằng một biến ngẫu nhiên đúng về mặt kỳ vọng
nhưng cắt ngắn chuỗi.
Điều này đạt được bằng cách dùng một dãy $\xi_t$
với $0 \leq \pi_t \leq 1$ được định trước,
trong đó $P(\xi_t = 0) = 1-\pi_t$ và
$P(\xi_t = \pi_t^{-1}) = \pi_t$, do đó $E[\xi_t] = 1$.
Chúng ta dùng điều này để thay gradient
$\partial h_t/\partial w_\textrm{h}$
trong :eqref:`eq_bptt_partial_ht_wh_recur`
bằng

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$


Từ định nghĩa của $\xi_t$ suy ra
rằng $E[z_t] = \partial h_t/\partial w_\textrm{h}$.
Bất cứ khi nào $\xi_t = 0$, phép tính hồi quy
kết thúc tại bước thời gian $t$ đó.
Điều này dẫn đến một tổng có trọng số của các chuỗi có độ dài khác nhau,
trong đó các chuỗi dài hiếm gặp nhưng được gán trọng số lớn một cách phù hợp.
Ý tưởng này được đề xuất bởi
Tallec.Ollivier.2017.

### So sánh các Chiến lược

![So sánh các chiến lược để tính gradient trong RNN. Từ trên xuống dưới: cắt ngắn ngẫu nhiên, cắt ngắn thông thường, và tính toán đầy đủ.](../img/truncated-bptt.svg)
<a id="fig_truncated_bptt"></a>


[fig_truncated_bptt](#fig_truncated_bptt) minh họa ba chiến lược
khi phân tích vài ký tự đầu tiên của *The Time Machine*
bằng lan truyền ngược qua thời gian cho RNN:

* Hàng đầu tiên là cắt ngắn ngẫu nhiên, chia văn bản thành các đoạn có độ dài khác nhau.
* Hàng thứ hai là cắt ngắn thông thường, chia văn bản thành các chuỗi con có cùng độ dài. Đây là điều chúng ta đã làm trong các thí nghiệm RNN.
* Hàng thứ ba là lan truyền ngược qua thời gian đầy đủ, dẫn đến một biểu thức không khả thi về mặt tính toán.


Đáng tiếc là, dù hấp dẫn về mặt lý thuyết,
cắt ngắn ngẫu nhiên không hoạt động
tốt hơn nhiều so với cắt ngắn thông thường,
rất có thể do một số yếu tố.
Thứ nhất, ảnh hưởng của một quan sát
sau một số bước lan truyền ngược
về quá khứ là khá đủ
để nắm bắt các phụ thuộc trong thực tế.
Thứ hai, phương sai tăng lên chống lại thực tế
rằng gradient chính xác hơn khi có nhiều bước hơn.
Thứ ba, chúng ta thực sự *muốn* các mô hình chỉ có
phạm vi tương tác ngắn.
Vì vậy, lan truyền ngược qua thời gian được cắt ngắn đều đặn
có một hiệu ứng chính quy hóa nhẹ có thể là điều mong muốn.

## Lan truyền Ngược Qua Thời gian Chi tiết

Sau khi thảo luận nguyên lý chung,
hãy thảo luận chi tiết về lan truyền ngược qua thời gian.
Trái với phần phân tích trong [subsec_bptt_analysis](#subsec_bptt_analysis),
trong phần sau chúng ta sẽ chỉ ra cách tính
gradient của hàm mục tiêu
theo tất cả các tham số mô hình đã được phân rã.
Để đơn giản, chúng ta xét
một RNN không có tham số hệ số chặn,
có hàm kích hoạt trong lớp ẩn
sử dụng ánh xạ đồng nhất ($\phi(x)=x$).
Với bước thời gian $t$, giả sử đầu vào của một ví dụ đơn lẻ
và mục tiêu lần lượt là $\mathbf{x}_t \in \mathbb{R}^d$ và $y_t$.
Trạng thái ẩn $\mathbf{h}_t \in \mathbb{R}^h$
và đầu ra $\mathbf{o}_t \in \mathbb{R}^q$
được tính như sau

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_\textrm{hx} \mathbf{x}_t + \mathbf{W}_\textrm{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_\textrm{qh} \mathbf{h}_{t},\end{aligned}$$

trong đó $\mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$, và
$\mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$
là các tham số trọng số.
Ký hiệu $l(\mathbf{o}_t, y_t)$
là mất mát tại bước thời gian $t$.
Do đó, hàm mục tiêu của chúng ta,
mất mát trên $T$ bước thời gian
từ đầu chuỗi là

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$


Để trực quan hóa các phụ thuộc giữa
các biến và tham số mô hình trong quá trình tính toán
của RNN,
chúng ta có thể vẽ một đồ thị tính toán cho mô hình,
như được minh họa trong [fig_rnn_bptt](#fig_rnn_bptt).
Ví dụ, việc tính các trạng thái ẩn của bước thời gian 3,
$\mathbf{h}_3$, phụ thuộc vào các tham số mô hình
$\mathbf{W}_\textrm{hx}$ và $\mathbf{W}_\textrm{hh}$,
trạng thái ẩn của bước thời gian trước đó $\mathbf{h}_2$,
và đầu vào của bước thời gian hiện tại $\mathbf{x}_3$.

![Đồ thị tính toán thể hiện các phụ thuộc cho một mô hình RNN với ba bước thời gian. Các hộp biểu diễn biến (không tô bóng) hoặc tham số (tô bóng), còn các hình tròn biểu diễn toán tử.](../img/rnn-bptt.svg)
<a id="fig_rnn_bptt"></a>

Như vừa đề cập, các tham số mô hình trong [fig_rnn_bptt](#fig_rnn_bptt)
là $\mathbf{W}_\textrm{hx}$, $\mathbf{W}_\textrm{hh}$, và $\mathbf{W}_\textrm{qh}$.
Nhìn chung, huấn luyện mô hình này đòi hỏi
tính gradient theo các tham số này
$\partial L/\partial \mathbf{W}_\textrm{hx}$, $\partial L/\partial \mathbf{W}_\textrm{hh}$, và $\partial L/\partial \mathbf{W}_\textrm{qh}$.
Theo các phụ thuộc trong [fig_rnn_bptt](#fig_rnn_bptt),
chúng ta có thể duyệt theo hướng ngược lại của các mũi tên
để lần lượt tính và lưu trữ các gradient.
Để biểu diễn linh hoạt phép nhân của
ma trận, vectơ và vô hướng có các hình dạng khác nhau
trong quy tắc dây chuyền,
chúng ta tiếp tục dùng toán tử $\textrm{prod}$
như đã mô tả trong [sec_backprop](#sec_backprop).


Trước hết, lấy đạo hàm hàm mục tiêu
theo đầu ra mô hình tại bất kỳ bước thời gian $t$ nào
là khá đơn giản:

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$

Bây giờ chúng ta có thể tính gradient của hàm mục tiêu
theo tham số $\mathbf{W}_\textrm{qh}$
trong lớp đầu ra:
$\partial L/\partial \mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$.
Dựa trên [fig_rnn_bptt](#fig_rnn_bptt),
hàm mục tiêu $L$ phụ thuộc vào $\mathbf{W}_\textrm{qh}$
thông qua $\mathbf{o}_1, \ldots, \mathbf{o}_T$.
Sử dụng quy tắc dây chuyền cho ta

$$
\frac{\partial L}{\partial \mathbf{W}_\textrm{qh}}
= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_\textrm{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

trong đó $\partial L/\partial \mathbf{o}_t$
được cho bởi :eqref:`eq_bptt_partial_L_ot`.

Tiếp theo, như thể hiện trong [fig_rnn_bptt](#fig_rnn_bptt),
tại bước thời gian cuối cùng $T$,
hàm mục tiêu
$L$ chỉ phụ thuộc vào trạng thái ẩn $\mathbf{h}_T$
thông qua $\mathbf{o}_T$.
Do đó, chúng ta có thể dễ dàng tìm gradient
$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$
bằng quy tắc dây chuyền:

$$\frac{\partial L}{\partial \mathbf{h}_T} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$

Mọi thứ trở nên phức tạp hơn với bất kỳ bước thời gian $t < T$ nào,
trong đó hàm mục tiêu $L$ phụ thuộc vào
$\mathbf{h}_t$ thông qua $\mathbf{h}_{t+1}$ và $\mathbf{o}_t$.
Theo quy tắc dây chuyền,
gradient của trạng thái ẩn
$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$
tại bất kỳ bước thời gian $t < T$ nào có thể được tính hồi quy như sau:


$$\frac{\partial L}{\partial \mathbf{h}_t} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_\textrm{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$

Để phân tích, mở rộng phép tính hồi quy
cho bất kỳ bước thời gian $1 \leq t \leq T$ nào cho ta

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_\textrm{hh}^\top\right)}^{T-i} \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$

Từ :eqref:`eq_bptt_partial_L_ht`, chúng ta có thể thấy
rằng ví dụ tuyến tính đơn giản này đã
thể hiện một số vấn đề then chốt của các mô hình chuỗi dài:
nó liên quan đến các lũy thừa có thể rất lớn của $\mathbf{W}_\textrm{hh}^\top$.
Trong đó, các trị riêng nhỏ hơn 1 sẽ tiêu biến
và các trị riêng lớn hơn 1 sẽ phân kỳ.
Điều này bất ổn về mặt số học,
biểu hiện dưới dạng gradient tiêu biến
và bùng nổ.
Một cách để xử lý điều này là cắt ngắn các bước thời gian
ở một kích thước thuận tiện về mặt tính toán
như đã thảo luận trong [subsec_bptt_analysis](#subsec_bptt_analysis).
Trong thực tế, việc cắt ngắn này cũng có thể được thực hiện
bằng cách tách gradient sau một số bước thời gian nhất định.
Về sau, chúng ta sẽ thấy cách các mô hình chuỗi tinh vi hơn
như long short-term memory có thể giảm nhẹ thêm vấn đề này.

Cuối cùng, [fig_rnn_bptt](#fig_rnn_bptt) cho thấy
rằng hàm mục tiêu $L$
phụ thuộc vào các tham số mô hình $\mathbf{W}_\textrm{hx}$ và $\mathbf{W}_\textrm{hh}$
trong lớp ẩn thông qua các trạng thái ẩn
$\mathbf{h}_1, \ldots, \mathbf{h}_T$.
Để tính gradient theo các tham số như vậy
$\partial L / \partial \mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$ và $\partial L / \partial \mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$,
chúng ta áp dụng quy tắc dây chuyền, thu được

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_\textrm{hx}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_\textrm{hh}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

trong đó $\partial L/\partial \mathbf{h}_t$,
được tính hồi quy bởi
:eqref:`eq_bptt_partial_L_hT_final_step`
và :eqref:`eq_bptt_partial_L_ht_recur`,
là đại lượng then chốt ảnh hưởng đến độ ổn định số học.


Vì lan truyền ngược qua thời gian là việc áp dụng lan truyền ngược trong RNN,
như chúng ta đã giải thích trong [sec_backprop](#sec_backprop),
huấn luyện RNN luân phiên giữa lan truyền xuôi và
lan truyền ngược qua thời gian.
Hơn nữa, lan truyền ngược qua thời gian
lần lượt tính và lưu trữ các gradient ở trên.
Cụ thể, các giá trị trung gian đã lưu
được tái sử dụng để tránh các phép tính trùng lặp,
chẳng hạn như lưu $\partial L/\partial \mathbf{h}_t$
để dùng trong việc tính cả $\partial L / \partial \mathbf{W}_\textrm{hx}$
và $\partial L / \partial \mathbf{W}_\textrm{hh}$.


## Tóm tắt

Lan truyền ngược qua thời gian chỉ đơn thuần là việc áp dụng lan truyền ngược cho các mô hình chuỗi có trạng thái ẩn.
Cần cắt ngắn, chẳng hạn như cắt ngắn thông thường hoặc ngẫu nhiên, để thuận tiện về mặt tính toán và đảm bảo độ ổn định số học.
Các lũy thừa bậc cao của ma trận có thể dẫn đến các trị riêng phân kỳ hoặc tiêu biến. Điều này biểu hiện dưới dạng gradient bùng nổ hoặc tiêu biến.
Để tính toán hiệu quả, các giá trị trung gian được lưu đệm trong quá trình lan truyền ngược qua thời gian.


## Bài tập

1. Giả sử chúng ta có một ma trận đối xứng $\mathbf{M} \in \mathbb{R}^{n \times n}$ với các trị riêng $\lambda_i$ có các vectơ riêng tương ứng là $\mathbf{v}_i$ ($i = 1, \ldots, n$). Không mất tính tổng quát, giả sử chúng được sắp xếp theo thứ tự $|\lambda_i| \geq |\lambda_{i+1}|$.
   1. Chứng minh rằng $\mathbf{M}^k$ có các trị riêng $\lambda_i^k$.
   1. Chứng minh rằng với một vectơ ngẫu nhiên $\mathbf{x} \in \mathbb{R}^n$, với xác suất cao $\mathbf{M}^k \mathbf{x}$ sẽ gần như thẳng hàng với vectơ riêng $\mathbf{v}_1$
của $\mathbf{M}$. Hãy hình thức hóa phát biểu này.
   1. Kết quả trên có ý nghĩa gì đối với gradient trong RNN?
1. Ngoài cắt gradient, bạn có thể nghĩ ra phương pháp nào khác để đối phó với gradient bùng nổ trong mạng nơ-ron hồi quy không?

[Discussions](https://discuss.d2l.ai/t/334)

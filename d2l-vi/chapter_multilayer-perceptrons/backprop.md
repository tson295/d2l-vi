# Lan truyền Xuôi, Lan truyền Ngược và Đồ thị Tính toán
<a id="sec_backprop"></a>

Cho đến nay, chúng ta đã huấn luyện mô hình
với stochastic gradient descent minibatch.
Tuy nhiên, khi cài đặt thuật toán,
chúng ta chỉ lo lắng về các phép tính liên quan
trong *lan truyền xuôi* qua mô hình.
Khi đến lúc tính gradient,
chúng ta chỉ gọi hàm lan truyền ngược được cung cấp bởi framework deep learning.

Việc tự động tính gradient
đơn giản hóa sâu sắc
việc cài đặt các thuật toán deep learning.
Trước khi có vi phân tự động,
ngay cả những thay đổi nhỏ đối với các mô hình phức tạp cũng đòi hỏi
tính lại các đạo hàm phức tạp bằng tay.
Thường xuyên đáng ngạc nhiên, các bài báo học thuật phải dành
nhiều trang để suy dẫn các quy tắc cập nhật.
Trong khi chúng ta phải tiếp tục dựa vào vi phân tự động
để có thể tập trung vào những phần thú vị,
bạn nên biết cách tính các gradient này
dưới nền nếu bạn muốn vượt qua sự hiểu biết bề ngoài
về deep learning.

Trong phần này, chúng ta đi sâu vào
chi tiết của *lan truyền ngược*
(thường được gọi là *backpropagation*).
Để truyền đạt một số hiểu biết cho cả
kỹ thuật và việc cài đặt của chúng,
chúng ta dựa vào một số toán học cơ bản và đồ thị tính toán.
Để bắt đầu, chúng ta tập trung trình bày về
MLP một lớp ẩn
với suy giảm trọng số (chuẩn hóa $\ell_2$, sẽ được mô tả trong các chương tiếp theo).

## Lan truyền Xuôi

*Lan truyền xuôi* (hay *forward pass*) đề cập đến việc tính và lưu trữ
các biến trung gian (bao gồm đầu ra)
cho một mạng nơ-ron theo thứ tự
từ lớp đầu vào đến lớp đầu ra.
Bây giờ chúng ta làm từng bước qua cơ chế
của mạng nơ-ron với một lớp ẩn.
Điều này có thể có vẻ tẻ nhạt nhưng như lời bất hủ
của bậc thầy funk James Brown,
bạn phải "trả cái giá để làm ông chủ".


Để đơn giản, hãy giả định
rằng mẫu đầu vào là $\mathbf{x}\in \mathbb{R}^d$
và lớp ẩn của chúng ta không bao gồm số hạng hệ số chặn.
Ở đây biến trung gian là:

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

trong đó $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
là tham số trọng số của lớp ẩn.
Sau khi chạy biến trung gian
$\mathbf{z}\in \mathbb{R}^h$ qua
hàm kích hoạt $\phi$,
chúng ta thu được vector kích hoạt ẩn có độ dài $h$:

$$\mathbf{h}= \phi (\mathbf{z}).$$

Đầu ra lớp ẩn $\mathbf{h}$
cũng là một biến trung gian.
Giả định rằng tham số của lớp đầu ra
chỉ có trọng số
$\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$,
chúng ta có thể thu được biến lớp đầu ra
với một vector có độ dài $q$:

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

Giả định rằng hàm mất mát là $l$
và nhãn mẫu là $y$,
chúng ta có thể tính số hạng mất mát
cho một mẫu dữ liệu đơn,

$$L = l(\mathbf{o}, y).$$

Như chúng ta sẽ thấy định nghĩa chuẩn hóa $\ell_2$
sẽ được giới thiệu sau,
với siêu tham số $\lambda$,
số hạng chuẩn hóa là

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_\textrm{F}^2 + \|\mathbf{W}^{(2)}\|_\textrm{F}^2\right),$$

trong đó chuẩn Frobenius của ma trận
đơn giản là chuẩn $\ell_2$ được áp dụng
sau khi làm phẳng ma trận thành vector.
Cuối cùng, mất mát được chuẩn hóa của mô hình
trên một mẫu dữ liệu cho trước là:

$$J = L + s.$$

Chúng ta gọi $J$ là *hàm mục tiêu*
trong thảo luận dưới đây.


## Đồ thị Tính toán của Lan truyền Xuôi

Vẽ *đồ thị tính toán* giúp chúng ta trực quan hóa
sự phụ thuộc của các toán tử
và biến trong phép tính.
[fig_forward](#fig_forward) chứa đồ thị liên quan
đến mạng đơn giản được mô tả ở trên,
trong đó các hình vuông biểu thị biến và các vòng tròn biểu thị toán tử.
Góc dưới bên trái biểu thị đầu vào
và góc trên bên phải là đầu ra.
Lưu ý rằng hướng của các mũi tên
(minh họa luồng dữ liệu)
chủ yếu hướng sang phải và lên trên.

![Đồ thị tính toán của lan truyền xuôi.](../img/forward.svg)
<a id="fig_forward"></a>

## Lan truyền Ngược

*Lan truyền ngược* đề cập đến phương pháp tính
gradient của các tham số mạng nơ-ron.
Tóm lại, phương pháp duyệt mạng theo thứ tự ngược,
từ đầu ra đến lớp đầu vào,
theo *quy tắc dây chuyền* từ giải tích.
Thuật toán lưu trữ bất kỳ biến trung gian nào
(đạo hàm riêng)
cần thiết trong khi tính gradient
theo một số tham số.
Giả định chúng ta có các hàm
$\mathsf{Y}=f(\mathsf{X})$
và $\mathsf{Z}=g(\mathsf{Y})$,
trong đó đầu vào và đầu ra
$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$
là các tensor có hình dạng tùy ý.
Bằng cách sử dụng quy tắc dây chuyền,
chúng ta có thể tính đạo hàm
của $\mathsf{Z}$ theo $\mathsf{X}$ qua

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \textrm{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

Ở đây chúng ta sử dụng toán tử $\textrm{prod}$
để nhân các đối số của nó
sau khi các phép toán cần thiết,
như chuyển vị và hoán đổi vị trí đầu vào,
đã được thực hiện.
Với vector, điều này rất đơn giản:
đó chỉ là phép nhân ma trận--ma trận.
Với tensor nhiều chiều hơn,
chúng ta sử dụng đối tác thích hợp.
Toán tử $\textrm{prod}$ ẩn tất cả các phí ký hiệu.

Nhớ lại rằng
các tham số của mạng đơn giản với một lớp ẩn,
có đồ thị tính toán trong [fig_forward](#fig_forward),
là $\mathbf{W}^{(1)}$ và $\mathbf{W}^{(2)}$.
Mục tiêu của lan truyền ngược là
tính gradient $\partial J/\partial \mathbf{W}^{(1)}$
và $\partial J/\partial \mathbf{W}^{(2)}$.
Để thực hiện điều này, chúng ta áp dụng quy tắc dây chuyền
và tính, lần lượt, gradient của
mỗi biến trung gian và tham số.
Thứ tự tính toán bị đảo ngược
so với những gì được thực hiện trong lan truyền xuôi,
vì chúng ta cần bắt đầu với kết quả của đồ thị tính toán
và làm việc theo hướng các tham số.
Bước đầu tiên là tính gradient
của hàm mục tiêu $J=L+s$
theo số hạng mất mát $L$
và số hạng chuẩn hóa $s$:

$$\frac{\partial J}{\partial L} = 1 \; \textrm{and} \; \frac{\partial J}{\partial s} = 1.$$

Tiếp theo, chúng ta tính gradient của hàm mục tiêu
theo biến của lớp đầu ra $\mathbf{o}$
theo quy tắc dây chuyền:

$$
\frac{\partial J}{\partial \mathbf{o}}
= \textrm{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Tiếp theo, chúng ta tính gradient
của số hạng chuẩn hóa
theo cả hai tham số:

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \textrm{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Bây giờ chúng ta có thể tính gradient
$\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$
của các tham số mô hình gần nhất với lớp đầu ra.
Sử dụng quy tắc dây chuyền cho:

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$

Để thu gradient theo $\mathbf{W}^{(1)}$,
chúng ta cần tiếp tục lan truyền ngược
dọc theo lớp đầu ra đến lớp ẩn.
Gradient theo đầu ra lớp ẩn
$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ được cho bởi


$$
\frac{\partial J}{\partial \mathbf{h}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

Vì hàm kích hoạt $\phi$ áp dụng theo từng phần tử,
tính gradient $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$
của biến trung gian $\mathbf{z}$
đòi hỏi chúng ta sử dụng toán tử nhân theo từng phần tử,
mà chúng ta ký hiệu là $\odot$:

$$
\frac{\partial J}{\partial \mathbf{z}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Cuối cùng, chúng ta có thể thu gradient
$\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
của các tham số mô hình gần nhất với lớp đầu vào.
Theo quy tắc dây chuyền, chúng ta có

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$


## Huấn luyện Mạng Nơ-ron

Khi huấn luyện mạng nơ-ron,
lan truyền xuôi và lan truyền ngược phụ thuộc vào nhau.
Đặc biệt, với lan truyền xuôi,
chúng ta duyệt đồ thị tính toán theo hướng phụ thuộc
và tính tất cả biến trên đường đi của nó.
Sau đó chúng được dùng cho lan truyền ngược
khi thứ tự tính toán trên đồ thị bị đảo ngược.

Lấy mạng đơn giản đã đề cập làm ví dụ minh họa.
Một mặt,
tính số hạng chuẩn hóa :eqref:`eq_forward-s`
trong quá trình lan truyền xuôi
phụ thuộc vào giá trị hiện tại của các tham số mô hình $\mathbf{W}^{(1)}$ và $\mathbf{W}^{(2)}$.
Chúng được cho bởi thuật toán tối ưu hóa theo lan truyền ngược trong lần lặp gần nhất.
Mặt khác,
tính gradient cho tham số
:eqref:`eq_backprop-J-h` trong quá trình lan truyền ngược
phụ thuộc vào giá trị hiện tại của đầu ra lớp ẩn $\mathbf{h}$,
được cho bởi lan truyền xuôi.


Do đó khi huấn luyện mạng nơ-ron, một khi các tham số mô hình được khởi tạo,
chúng ta xen kẽ lan truyền xuôi với lan truyền ngược,
cập nhật tham số mô hình bằng gradient được cho bởi lan truyền ngược.
Lưu ý rằng lan truyền ngược tái sử dụng các giá trị trung gian được lưu từ lan truyền xuôi để tránh tính toán trùng lặp.
Một trong những hệ quả là chúng ta cần giữ lại
các giá trị trung gian cho đến khi lan truyền ngược hoàn thành.
Đây cũng là một trong những lý do tại sao việc huấn luyện
đòi hỏi bộ nhớ nhiều hơn đáng kể so với dự đoán thông thường.
Hơn nữa, kích thước của các giá trị trung gian như vậy xấp xỉ
tỷ lệ với số lớp mạng và kích thước batch.
Vì vậy,
huấn luyện các mạng sâu hơn bằng kích thước batch lớn hơn
dễ dẫn đến lỗi *hết bộ nhớ* hơn.


## Tóm tắt

Lan truyền xuôi tính tuần tự và lưu trữ các biến trung gian trong đồ thị tính toán được định nghĩa bởi mạng nơ-ron. Nó tiến từ lớp đầu vào đến lớp đầu ra.
Lan truyền ngược tính tuần tự và lưu trữ các gradient của biến trung gian và tham số trong mạng nơ-ron theo thứ tự ngược.
Khi huấn luyện các mô hình deep learning, lan truyền xuôi và lan truyền ngược phụ thuộc vào nhau,
và huấn luyện đòi hỏi bộ nhớ nhiều hơn đáng kể so với dự đoán.


## Bài tập

1. Giả sử rằng đầu vào $\mathbf{X}$ cho một số hàm vô hướng $f$ là ma trận $n \times m$. Chiều của gradient của $f$ theo $\mathbf{X}$ là gì?
1. Thêm hệ số chặn vào lớp ẩn của mô hình được mô tả trong phần này (bạn không cần bao gồm hệ số chặn trong số hạng chuẩn hóa).
    1. Vẽ đồ thị tính toán tương ứng.
    1. Suy dẫn các phương trình lan truyền xuôi và lan truyền ngược.
1. Tính dung lượng bộ nhớ cho huấn luyện và dự đoán trong mô hình được mô tả trong phần này.
1. Giả sử bạn muốn tính đạo hàm bậc hai. Điều gì xảy ra với đồ thị tính toán? Bạn kỳ vọng tính toán mất bao lâu?
1. Giả sử đồ thị tính toán quá lớn cho GPU của bạn.
    1. Bạn có thể phân vùng nó qua nhiều hơn một GPU không?
    1. Các ưu điểm và nhược điểm so với huấn luyện trên minibatch nhỏ hơn là gì?

[Discussions](https://discuss.d2l.ai/t/102)

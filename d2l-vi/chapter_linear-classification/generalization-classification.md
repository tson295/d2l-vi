# Tổng quát hóa trong Phân loại

<a id="chap_classification_generalization"></a>


Cho đến nay, chúng ta đã tập trung vào cách giải quyết các bài toán phân loại đa lớp
bằng cách huấn luyện mạng nơ-ron (tuyến tính) với nhiều đầu ra và hàm softmax.
Bằng cách diễn giải đầu ra của mô hình như các dự đoán xác suất,
chúng ta đã động lực hóa và suy dẫn hàm mất mát cross-entropy,
tính log-likelihood âm mà mô hình của chúng ta (với một tập tham số cố định)
gán cho các nhãn thực.
Và cuối cùng, chúng ta đã đưa các công cụ này vào thực tế
bằng cách khớp mô hình với tập huấn luyện.
Tuy nhiên, như thường lệ, mục tiêu của chúng ta là học *các mẫu tổng quát*,
được đánh giá thực nghiệm trên dữ liệu chưa từng thấy (tập kiểm tra).
Độ chính xác cao trên tập huấn luyện không có nghĩa lý gì.
Mỗi khi mỗi đầu vào của chúng ta là duy nhất
(và thực sự điều này đúng với hầu hết các bộ dữ liệu nhiều chiều),
chúng ta có thể đạt được độ chính xác hoàn hảo trên tập huấn luyện
chỉ bằng cách ghi nhớ bộ dữ liệu trong epoch huấn luyện đầu tiên,
và sau đó tra cứu nhãn mỗi khi thấy ảnh mới.
Tuy nhiên, ghi nhớ chính xác các nhãn
liên quan đến các mẫu huấn luyện chính xác
không cho chúng ta biết cách phân loại các mẫu mới.
Nếu không có hướng dẫn thêm, chúng ta có thể phải dựa vào
đoán ngẫu nhiên mỗi khi gặp các mẫu mới.

Một số câu hỏi cấp bách đòi hỏi sự chú ý ngay lập tức:

1. Chúng ta cần bao nhiêu mẫu kiểm tra để có ước tính tốt về độ chính xác của các bộ phân loại trên tổng thể?
1. Điều gì xảy ra nếu chúng ta tiếp tục đánh giá mô hình trên cùng một tập kiểm tra nhiều lần?
1. Tại sao chúng ta nên kỳ vọng rằng việc khớp các mô hình tuyến tính với tập huấn luyện
   sẽ tốt hơn sơ đồ ghi nhớ ngây thơ của chúng ta?


Trong khi [sec_generalization_basics](#sec_generalization_basics) đã giới thiệu
các kiến thức cơ bản về quá khớp và tổng quát hóa
trong ngữ cảnh hồi quy tuyến tính,
chương này sẽ đi sâu hơn một chút,
giới thiệu một số ý tưởng nền tảng
của lý thuyết học thống kê.
Hóa ra chúng ta thường có thể đảm bảo tổng quát hóa *a priori*:
với nhiều mô hình,
và với bất kỳ giới hạn trên mong muốn nào
trên khoảng cách tổng quát hóa $\epsilon$,
chúng ta thường có thể xác định một số lượng mẫu cần thiết $n$
sao cho nếu tập huấn luyện của chúng ta chứa ít nhất $n$
mẫu, lỗi thực nghiệm của chúng ta sẽ nằm
trong $\epsilon$ của lỗi thực,
*với bất kỳ phân phối tạo dữ liệu nào*.
Thật không may, cũng hóa ra
rằng trong khi những đảm bảo này cung cấp
một tập hợp khối xây dựng trí tuệ sâu sắc,
chúng có giá trị thực tế hạn chế
cho người thực hành deep learning.
Nói ngắn gọn, những đảm bảo này gợi ý
rằng đảm bảo tổng quát hóa
của mạng nơ-ron sâu *a priori*
đòi hỏi một số lượng mẫu vô lý
(có thể hàng nghìn tỷ hoặc hơn),
ngay cả khi chúng ta thấy rằng, với các tác vụ chúng ta quan tâm,
mạng nơ-ron sâu thường tổng quát hóa
rất tốt với ít mẫu hơn nhiều (hàng nghìn).
Do đó, người thực hành deep learning thường từ bỏ
các đảm bảo *a priori* hoàn toàn,
thay vào đó sử dụng các phương pháp
đã tổng quát hóa tốt
trên các bài toán tương tự trong quá khứ,
và chứng nhận tổng quát hóa *post hoc*
thông qua đánh giá thực nghiệm.
Khi chúng ta đến [chap_perceptrons](#chap_perceptrons),
chúng ta sẽ xem xét lại tổng quát hóa
và cung cấp giới thiệu nhẹ nhàng
về tài liệu khoa học rộng lớn
đã nảy sinh trong các nỗ lực
giải thích tại sao mạng nơ-ron sâu tổng quát hóa trong thực tế.

## Tập Kiểm tra

Vì chúng ta đã bắt đầu dựa vào tập kiểm tra
như phương pháp tiêu chuẩn vàng
để đánh giá lỗi tổng quát hóa,
hãy bắt đầu bằng cách thảo luận
về tính chất của các ước tính lỗi như vậy.
Hãy tập trung vào một bộ phân loại cố định $f$,
không cần lo lắng về cách nó được thu được.
Hơn nữa giả sử rằng chúng ta sở hữu
một *tập dữ liệu mới* gồm các mẫu $\mathcal{D} = {(\mathbf{x}^{(i)},y^{(i)})}_{i=1}^n$
không được dùng để huấn luyện bộ phân loại $f$.
*Lỗi thực nghiệm* của bộ phân loại $f$ trên $\mathcal{D}$
đơn giản là tỷ lệ các mẫu
mà dự đoán $f(\mathbf{x}^{(i)})$
không đồng ý với nhãn thực $y^{(i)}$,
và được cho bởi biểu thức sau:

$$\epsilon_\mathcal{D}(f) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)}).$$

Ngược lại, *lỗi tổng thể*
là tỷ lệ *kỳ vọng*
của các mẫu trong tổng thể cơ bản
(một phân phối $P(X,Y)$ được mô tả
bởi hàm mật độ xác suất $p(\mathbf{x},y)$)
mà bộ phân loại của chúng ta không đồng ý
với nhãn thực:

$$\epsilon(f) =  E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) =
\int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Trong khi $\epsilon(f)$ là đại lượng chúng ta thực sự quan tâm,
chúng ta không thể quan sát nó trực tiếp,
giống như chúng ta không thể trực tiếp
quan sát chiều cao trung bình trong một dân số lớn
mà không đo từng người.
Chúng ta chỉ có thể ước tính đại lượng này dựa trên mẫu.
Vì tập kiểm tra $\mathcal{D}$ của chúng ta
có tính đại diện thống kê
của tổng thể cơ bản,
chúng ta có thể xem $\epsilon_\mathcal{D}(f)$ như một bộ ước lượng thống kê
của lỗi tổng thể $\epsilon(f)$.
Hơn nữa, vì đại lượng quan tâm $\epsilon(f)$ của chúng ta
là kỳ vọng (của biến ngẫu nhiên $\mathbf{1}(f(X) \neq Y)$)
và bộ ước lượng tương ứng $\epsilon_\mathcal{D}(f)$
là trung bình mẫu,
ước tính lỗi tổng thể
đơn giản là bài toán ước lượng trung bình cổ điển,
bạn có thể nhớ từ [sec_prob](#sec_prob).

Một kết quả cổ điển quan trọng từ lý thuyết xác suất
gọi là *định lý giới hạn trung tâm* đảm bảo
rằng bất cứ khi nào chúng ta sở hữu $n$ mẫu ngẫu nhiên $a_1, ..., a_n$
được rút từ bất kỳ phân phối nào với trung bình $\mu$ và độ lệch chuẩn $\sigma$,
thì khi số mẫu $n$ tiến đến vô cùng,
trung bình mẫu $\hat{\mu}$ xấp xỉ
có xu hướng về phân phối chuẩn tập trung
tại trung bình thực và với độ lệch chuẩn $\sigma/\sqrt{n}$.
Điều này đã cho chúng ta biết điều quan trọng:
khi số mẫu tăng lớn,
lỗi kiểm tra $\epsilon_\mathcal{D}(f)$ của chúng ta
nên tiến gần đến lỗi thực $\epsilon(f)$
với tốc độ $\mathcal{O}(1/\sqrt{n})$.
Do đó, để ước tính lỗi kiểm tra của chúng ta chính xác gấp đôi,
chúng ta phải thu thập tập kiểm tra lớn gấp bốn lần.
Để giảm lỗi kiểm tra của chúng ta xuống một trăm lần,
chúng ta phải thu thập tập kiểm tra lớn gấp mười nghìn lần.
Nhìn chung, tốc độ $\mathcal{O}(1/\sqrt{n})$ như vậy
thường là điều tốt nhất chúng ta có thể hy vọng trong thống kê.

Bây giờ chúng ta đã biết về tốc độ tiệm cận
mà lỗi kiểm tra $\epsilon_\mathcal{D}(f)$ hội tụ về lỗi thực $\epsilon(f)$,
chúng ta có thể phóng to vào một số chi tiết quan trọng.
Nhớ lại rằng biến ngẫu nhiên quan tâm
$\mathbf{1}(f(X) \neq Y)$
chỉ có thể lấy giá trị $0$ và $1$
và do đó là biến ngẫu nhiên Bernoulli,
được đặc trưng bởi một tham số
chỉ ra xác suất mà nó lấy giá trị $1$.
Ở đây, $1$ có nghĩa là bộ phân loại của chúng ta mắc lỗi,
do đó tham số của biến ngẫu nhiên
thực ra là tỷ lệ lỗi thực $\epsilon(f)$.
Phương sai $\sigma^2$ của Bernoulli
phụ thuộc vào tham số của nó (ở đây là $\epsilon(f)$)
theo biểu thức $\epsilon(f)(1-\epsilon(f))$.
Trong khi $\epsilon(f)$ ban đầu chưa biết,
chúng ta biết rằng nó không thể lớn hơn $1$.
Nghiên cứu nhỏ về hàm này
tiết lộ rằng phương sai của chúng ta là cao nhất
khi tỷ lệ lỗi thực gần $0.5$
và có thể thấp hơn nhiều khi nó gần
$0$ hoặc gần $1$.
Điều này cho chúng ta biết rằng độ lệch chuẩn tiệm cận
của ước tính $\epsilon_\mathcal{D}(f)$ về lỗi $\epsilon(f)$
(trên việc chọn $n$ mẫu kiểm tra)
không thể lớn hơn $\sqrt{0.25/n}$.

Nếu chúng ta bỏ qua thực tế rằng tốc độ này mô tả
hành vi khi kích thước tập kiểm tra tiến đến vô cùng
thay vì khi chúng ta có mẫu hữu hạn,
điều này cho chúng ta biết rằng nếu muốn lỗi kiểm tra $\epsilon_\mathcal{D}(f)$
xấp xỉ lỗi tổng thể $\epsilon(f)$
sao cho một độ lệch chuẩn tương ứng
với khoảng $\pm 0.01$,
thì chúng ta nên thu thập khoảng 2500 mẫu.
Nếu muốn hai độ lệch chuẩn
trong phạm vi đó và do đó tự tin 95%
rằng $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$,
thì chúng ta sẽ cần 10,000 mẫu!

Hóa ra đây chính là kích thước của tập kiểm tra
cho nhiều benchmark phổ biến trong machine learning.
Bạn có thể ngạc nhiên khi biết rằng hàng nghìn
bài báo deep learning ứng dụng được xuất bản mỗi năm
làm to chuyện về cải thiện tỷ lệ lỗi $0.01$ hoặc ít hơn.
Tất nhiên, khi tỷ lệ lỗi gần $0$ hơn nhiều,
thì cải thiện $0.01$ thực sự có thể là điều quan trọng.


Một đặc điểm khó chịu của phân tích đến nay
là nó thực sự chỉ cho chúng ta biết về tiệm cận,
tức là cách mối quan hệ giữa $\epsilon_\mathcal{D}$ và $\epsilon$
phát triển khi kích thước mẫu tiến đến vô cùng.
May mắn thay, vì biến ngẫu nhiên của chúng ta bị chặn,
chúng ta có thể thu được các giới hạn mẫu hữu hạn hợp lệ
bằng cách áp dụng bất đẳng thức do Hoeffding (1963) đề xuất:

$$P(\epsilon_\mathcal{D}(f) - \epsilon(f) \geq t) < \exp\left( - 2n t^2 \right).$$

Giải cho kích thước tập dữ liệu nhỏ nhất
cho phép chúng ta kết luận
với độ tự tin 95% rằng khoảng cách $t$
giữa ước tính $\epsilon_\mathcal{D}(f)$
và tỷ lệ lỗi thực $\epsilon(f)$
không vượt quá $0.01$,
bạn sẽ thấy rằng cần khoảng 15,000 mẫu
so với 10,000 mẫu do phân tích tiệm cận ở trên đề xuất.
Nếu bạn đi sâu hơn vào thống kê
bạn sẽ thấy xu hướng này thường đúng.
Các đảm bảo giữ ngay cả trong mẫu hữu hạn
thường hơi bảo thủ hơn.
Lưu ý rằng trong bức tranh tổng thể,
các con số này không quá khác nhau,
phản ánh tính hữu ích chung
của phân tích tiệm cận trong việc cung cấp
các con số ước lượng ngay cả khi chúng không phải là
đảm bảo chúng ta có thể đưa ra tòa.

## Tái sử dụng Tập Kiểm tra

Theo một nghĩa nào đó, bây giờ bạn đã được chuẩn bị để thành công
trong việc tiến hành nghiên cứu machine learning thực nghiệm.
Gần như tất cả các mô hình thực tế được phát triển
và kiểm định dựa trên hiệu suất tập kiểm tra
và bây giờ bạn là chuyên gia về tập kiểm tra.
Với bất kỳ bộ phân loại cố định $f$ nào,
bạn biết cách đánh giá lỗi kiểm tra $\epsilon_\mathcal{D}(f)$ của nó,
và biết chính xác điều gì có thể (và không thể)
được nói về lỗi tổng thể $\epsilon(f)$ của nó.

Vậy giả sử bạn lấy kiến thức này
và chuẩn bị huấn luyện mô hình đầu tiên $f_1$.
Biết chính xác bạn cần tự tin bao nhiêu
về hiệu suất tỷ lệ lỗi của bộ phân loại
bạn áp dụng phân tích ở trên để xác định
số lượng mẫu thích hợp
cần dành riêng cho tập kiểm tra.
Hơn nữa, giả sử bạn đã học bài học từ
[sec_generalization_basics](#sec_generalization_basics) và
đảm bảo giữ gìn sự thiêng liêng của tập kiểm tra
bằng cách tiến hành tất cả phân tích sơ bộ,
điều chỉnh siêu tham số, và thậm chí lựa chọn
giữa nhiều kiến trúc mô hình cạnh tranh
trên tập kiểm định.
Cuối cùng bạn đánh giá mô hình $f_1$ của mình
trên tập kiểm tra và báo cáo ước tính không thiên lệch
của lỗi tổng thể
với khoảng tin cậy liên quan.

Đến đây mọi thứ có vẻ đang diễn ra tốt.
Tuy nhiên, đêm đó bạn thức dậy lúc 3 giờ sáng
với một ý tưởng xuất sắc cho phương pháp mô hình hóa mới.
Ngày hôm sau, bạn code mô hình mới,
điều chỉnh siêu tham số trên tập kiểm định
và không chỉ bạn làm cho mô hình mới $f_2$ hoạt động
mà tỷ lệ lỗi của nó có vẻ thấp hơn nhiều so với $f_1$.
Tuy nhiên, niềm vui khám phá đột nhiên phai đi
khi bạn chuẩn bị cho đánh giá cuối cùng.
Bạn không có tập kiểm tra!

Mặc dù tập kiểm tra gốc $\mathcal{D}$
vẫn còn trên máy chủ của bạn,
bạn bây giờ đối mặt với hai vấn đề đáng kể.
Đầu tiên, khi bạn thu thập tập kiểm tra,
bạn đã xác định mức độ chính xác cần thiết
dưới giả định rằng bạn đang đánh giá
một bộ phân loại duy nhất $f$.
Tuy nhiên, nếu bạn bước vào lĩnh vực
đánh giá nhiều bộ phân loại $f_1, ..., f_k$
trên cùng một tập kiểm tra,
bạn phải xem xét vấn đề khám phá sai.
Trước đây, bạn có thể đã 95% chắc chắn
rằng $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$
với một bộ phân loại duy nhất $f$
và do đó xác suất của kết quả gây hiểu nhầm
chỉ là 5%.
Với $k$ bộ phân loại trong cuộc,
có thể khó đảm bảo
rằng không có một bộ phân loại nào trong số đó
có hiệu suất tập kiểm tra gây hiểu nhầm.
Với 20 bộ phân loại được xem xét,
bạn có thể không có khả năng nào
để loại trừ khả năng
rằng ít nhất một trong số chúng
nhận được điểm số gây hiểu nhầm.
Vấn đề này liên quan đến kiểm định giả thuyết bội,
mặc dù có tài liệu rộng lớn trong thống kê,
vẫn là vấn đề dai dẳng ảnh hưởng đến nghiên cứu khoa học.


Nếu điều đó chưa đủ lo lắng bạn,
có một lý do đặc biệt để không tin vào
kết quả bạn nhận được trong các đánh giá tiếp theo.
Nhớ lại rằng phân tích của chúng ta về hiệu suất tập kiểm tra
dựa trên giả định rằng bộ phân loại
được chọn khi không có tiếp xúc với tập kiểm tra
và do đó chúng ta có thể xem tập kiểm tra
như được rút ngẫu nhiên từ tổng thể cơ bản.
Ở đây, không chỉ bạn đang kiểm tra nhiều hàm,
hàm tiếp theo $f_2$ được chọn
sau khi bạn quan sát hiệu suất tập kiểm tra của $f_1$.
Một khi thông tin từ tập kiểm tra đã rò rỉ cho người mô hình hóa,
nó không bao giờ có thể là tập kiểm tra thực sự nữa theo nghĩa nghiêm ngặt.
Vấn đề này được gọi là *quá khớp thích nghi* và gần đây đã nổi lên
như một chủ đề quan tâm mãnh liệt của các nhà lý thuyết học và thống kê
[dwork2015preserving].
May mắn thay, trong khi có thể
rò rỉ tất cả thông tin ra khỏi tập giữ lại,
và các tình huống xấu nhất theo lý thuyết rất u ám,
các phân tích này có thể quá bảo thủ.
Trong thực tế, hãy cẩn thận tạo ra các tập kiểm tra thực,
tham khảo chúng càng ít càng tốt,
tính đến kiểm định giả thuyết bội
khi báo cáo khoảng tin cậy,
và tăng cảnh giác mạnh mẽ hơn
khi cổ phần cao và kích thước tập dữ liệu nhỏ.
Khi chạy một loạt thách thức benchmark,
thường là thực hành tốt để duy trì
một số tập kiểm tra để sau mỗi vòng,
tập kiểm tra cũ có thể được hạ xuống làm tập kiểm định.


## Lý thuyết Học Thống kê

Nói đơn giản, *tập kiểm tra là tất cả những gì chúng ta thực sự có*,
và tuy nhiên sự thật này có vẻ kỳ lạ không thỏa mãn.
Đầu tiên, chúng ta hiếm khi sở hữu *tập kiểm tra thực sự*---trừ khi
chúng ta là người tạo ra bộ dữ liệu,
ai đó khác có thể đã đánh giá
bộ phân loại của riêng họ trên "tập kiểm tra" danh nghĩa của chúng ta.
Và ngay cả khi chúng ta có quyền đầu tiên,
chúng ta sớm thấy mình thất vọng, muốn có thể
đánh giá các lần mô hình hóa tiếp theo
mà không có cảm giác khó chịu
rằng chúng ta không thể tin vào các con số của mình.
Hơn nữa, ngay cả tập kiểm tra thực sự cũng chỉ có thể cho chúng ta biết *post hoc*
liệu bộ phân loại có thực sự tổng quát hóa cho tổng thể hay không,
không phải liệu chúng ta có bất kỳ lý do để kỳ vọng *a priori*
rằng nó nên tổng quát hóa.

Với những lo ngại này trong tâm trí,
bạn có thể bây giờ đủ chuẩn bị
để thấy sức hút của *lý thuyết học thống kê*,
lĩnh vực con toán học của machine learning
mà những người thực hành nhằm làm rõ các
nguyên lý cơ bản giải thích
tại sao/khi nào các mô hình được huấn luyện trên dữ liệu thực nghiệm
có thể/sẽ tổng quát hóa cho dữ liệu chưa thấy.
Một trong những mục tiêu chính
của các nhà nghiên cứu học thống kê
là giới hạn khoảng cách tổng quát hóa,
liên quan các tính chất của lớp mô hình
với số lượng mẫu trong bộ dữ liệu.

Các nhà lý thuyết học nhằm giới hạn sự khác biệt
giữa *lỗi thực nghiệm* $\epsilon_\mathcal{S}(f_\mathcal{S})$
của bộ phân loại đã học $f_\mathcal{S}$,
được huấn luyện và đánh giá
trên tập huấn luyện $\mathcal{S}$,
và lỗi thực $\epsilon(f_\mathcal{S})$
của cùng bộ phân loại đó trên tổng thể cơ bản.
Điều này có thể trông giống với bài toán đánh giá
mà chúng ta vừa giải quyết nhưng có sự khác biệt lớn.
Trước đó, bộ phân loại $f$ được cố định
và chúng ta chỉ cần một bộ dữ liệu
cho mục đích đánh giá.
Và thực sự, bất kỳ bộ phân loại cố định nào đều tổng quát hóa:
lỗi của nó trên một bộ dữ liệu (chưa thấy trước đây)
là ước tính không thiên lệch của lỗi tổng thể.
Nhưng chúng ta có thể nói gì khi bộ phân loại
được huấn luyện và đánh giá trên cùng một bộ dữ liệu?
Chúng ta có thể tự tin rằng lỗi huấn luyện
sẽ gần với lỗi kiểm tra không?


Giả sử bộ phân loại đã học $f_\mathcal{S}$ của chúng ta phải được chọn
từ một tập hàm được chỉ định trước $\mathcal{F}$.
Nhớ lại từ thảo luận của chúng ta về tập kiểm tra
rằng trong khi dễ ước tính
lỗi của một bộ phân loại duy nhất,
mọi thứ trở nên phức tạp khi chúng ta bắt đầu
xem xét các tập hợp bộ phân loại.
Ngay cả khi lỗi thực nghiệm
của bất kỳ (cố định) bộ phân loại nào
sẽ gần với lỗi thực của nó
với xác suất cao,
một khi chúng ta xem xét một tập hợp bộ phân loại,
chúng ta cần lo lắng về khả năng
rằng *chỉ một* trong số chúng
sẽ nhận được lỗi được ước tính tệ.
Mối lo ngại là chúng ta có thể chọn bộ phân loại như vậy
và do đó đánh giá thấp nghiêm trọng
lỗi tổng thể.
Hơn nữa, ngay cả với các mô hình tuyến tính,
vì tham số của chúng có giá trị liên tục,
chúng ta thường chọn từ
một lớp hàm vô hạn ($|\mathcal{F}| = \infty$).

Một giải pháp tham vọng cho vấn đề
là phát triển các công cụ phân tích
để chứng minh hội tụ đồng đều, tức là
với xác suất cao,
tỷ lệ lỗi thực nghiệm của mọi bộ phân loại trong lớp $f\in\mathcal{F}$
sẽ *đồng thời* hội tụ về tỷ lệ lỗi thực của nó.
Nói cách khác, chúng ta tìm kiếm một nguyên lý lý thuyết
cho phép chúng ta nói rằng
với xác suất ít nhất $1-\delta$
(với một $\delta$ nhỏ)
không có tỷ lệ lỗi $\epsilon(f)$ của bộ phân loại nào
(trong số tất cả bộ phân loại trong lớp $\mathcal{F}$)
sẽ bị ước tính sai hơn
một lượng nhỏ $\alpha$.
Rõ ràng, chúng ta không thể đưa ra những tuyên bố như vậy
cho tất cả các lớp mô hình $\mathcal{F}$.
Nhớ lại lớp các máy ghi nhớ
luôn đạt lỗi thực nghiệm $0$
nhưng không bao giờ vượt qua đoán ngẫu nhiên
trên tổng thể cơ bản.

Theo một nghĩa nào đó, lớp các máy ghi nhớ quá linh hoạt.
Không có kết quả hội tụ đồng đều nào như vậy có thể xảy ra.
Mặt khác, một bộ phân loại cố định là vô dụng---nó
tổng quát hóa hoàn hảo, nhưng không khớp
dữ liệu huấn luyện lẫn dữ liệu kiểm tra.
Câu hỏi trung tâm của học
do đó theo truyền thống được đóng khung như sự đánh đổi
giữa các lớp mô hình linh hoạt hơn (phương sai cao hơn)
khớp dữ liệu huấn luyện tốt hơn nhưng có nguy cơ quá khớp,
so với các lớp mô hình cứng hơn (hệ số chặn cao hơn)
tổng quát hóa tốt nhưng có nguy cơ dưới khớp.
Một câu hỏi trung tâm trong lý thuyết học
là phát triển phân tích toán học thích hợp để định lượng
vị trí của mô hình dọc theo phổ này,
và cung cấp các đảm bảo liên quan.

Trong một loạt các bài báo seminal,
Vapnik và Chervonenkis đã mở rộng
lý thuyết về hội tụ
của tần số tương đối
sang các lớp hàm tổng quát hơn
[VapChe64, VapChe68, VapChe71, VapChe74b, VapChe81, VapChe91].
Một trong những đóng góp chính của dòng công trình này
là chiều Vapnik--Chervonenkis (VC),
đo (một khái niệm về)
độ phức tạp (linh hoạt) của một lớp mô hình.
Hơn nữa, một trong những kết quả chính của họ giới hạn
sự khác biệt giữa lỗi thực nghiệm
và lỗi tổng thể như một hàm
của chiều VC và số lượng mẫu:

$$P\left(R[p, f] - R_\textrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \alpha\right) \geq 1-\delta
\ \textrm{ for }\ \alpha \geq c \sqrt{(\textrm{VC} - \log \delta)/n}.$$

Ở đây $\delta > 0$ là xác suất mà giới hạn bị vi phạm,
$\alpha$ là giới hạn trên của khoảng cách tổng quát hóa,
và $n$ là kích thước tập dữ liệu.
Cuối cùng, $c > 0$ là một hằng số phụ thuộc
chỉ vào quy mô của mất mát có thể phát sinh.
Một cách sử dụng giới hạn là điền vào các giá trị mong muốn
của $\delta$ và $\alpha$
để xác định số lượng mẫu cần thu thập.
Chiều VC định lượng số lớn nhất
của điểm dữ liệu mà chúng ta có thể gán
bất kỳ nhãn (nhị phân) tùy ý nào
và cho mỗi điểm tìm một số mô hình $f$ trong lớp
đồng ý với nhãn đó.
Ví dụ, các mô hình tuyến tính trên đầu vào $d$ chiều
có chiều VC $d+1$.
Dễ dàng thấy rằng một đường thẳng có thể gán
bất kỳ nhãn nào cho ba điểm trong hai chiều,
nhưng không thể cho bốn điểm.
Thật không may, lý thuyết có xu hướng
quá bi quan với các mô hình phức tạp hơn
và thu được đảm bảo này thường đòi hỏi
nhiều mẫu hơn nhiều so với thực tế cần thiết
để đạt tỷ lệ lỗi mong muốn.
Lưu ý cũng rằng khi cố định lớp mô hình và $\delta$,
tỷ lệ lỗi của chúng ta một lần nữa giảm
với tốc độ $\mathcal{O}(1/\sqrt{n})$ thông thường.
Có vẻ không thể chúng ta làm tốt hơn về mặt $n$.
Tuy nhiên, khi chúng ta thay đổi lớp mô hình,
chiều VC có thể trình bày
một bức tranh bi quan
của khoảng cách tổng quát hóa.


## Tóm tắt

Cách đơn giản nhất để đánh giá mô hình
là tham khảo tập kiểm tra bao gồm dữ liệu chưa từng thấy.
Đánh giá tập kiểm tra cung cấp ước tính không thiên lệch của lỗi thực
và hội tụ với tốc độ $\mathcal{O}(1/\sqrt{n})$ mong muốn khi tập kiểm tra tăng.
Chúng ta có thể cung cấp khoảng tin cậy gần đúng
dựa trên phân phối tiệm cận chính xác
hoặc khoảng tin cậy mẫu hữu hạn hợp lệ
dựa trên (bảo thủ hơn) đảm bảo mẫu hữu hạn.
Thực sự đánh giá tập kiểm tra là nền tảng
của nghiên cứu machine learning hiện đại.
Tuy nhiên, tập kiểm tra hiếm khi là tập kiểm tra thực sự
(được nhiều nhà nghiên cứu sử dụng lại nhiều lần).
Một khi cùng một tập kiểm tra được dùng
để đánh giá nhiều mô hình,
việc kiểm soát khám phá sai có thể khó khăn.
Điều này có thể gây ra các vấn đề lớn trong lý thuyết.
Trong thực tế, tầm quan trọng của vấn đề
phụ thuộc vào kích thước của tập giữ lại được đề cập
và liệu chúng chỉ đang được dùng để chọn siêu tham số
hay liệu chúng đang rò rỉ thông tin trực tiếp hơn.
Tuy nhiên, đây là thực hành tốt để thu thập tập kiểm tra thực (hoặc nhiều tập)
và bảo thủ nhất có thể về tần suất chúng được sử dụng.


Hy vọng cung cấp giải pháp thỏa mãn hơn,
các nhà lý thuyết học thống kê đã phát triển các phương pháp
để đảm bảo hội tụ đồng đều trên một lớp mô hình.
Nếu thực sự lỗi thực nghiệm của mọi mô hình đồng thời
hội tụ về lỗi thực của nó,
thì chúng ta được tự do chọn mô hình hoạt động
tốt nhất, tối thiểu hóa lỗi huấn luyện,
biết rằng nó cũng sẽ hoạt động tương tự tốt
trên dữ liệu giữ lại.
Quan trọng là, bất kỳ kết quả nào như vậy phải phụ thuộc
vào một số tính chất của lớp mô hình.
Vladimir Vapnik và Alexey Chernovenkis
đã giới thiệu chiều VC,
trình bày các kết quả hội tụ đồng đều
giữ cho tất cả các mô hình trong lớp VC.
Lỗi huấn luyện của tất cả các mô hình trong lớp
được đảm bảo (đồng thời)
gần với lỗi thực của chúng,
và được đảm bảo hội tụ gần hơn nữa
với tốc độ $\mathcal{O}(1/\sqrt{n})$.
Sau khám phá cách mạng về chiều VC,
nhiều thước đo độ phức tạp thay thế đã được đề xuất,
mỗi cái tạo điều kiện cho đảm bảo tổng quát hóa tương tự.
Xem boucheron2005theory để thảo luận chi tiết
về một số cách đo độ phức tạp hàm tiên tiến.
Thật không may, trong khi các thước đo độ phức tạp này
đã trở thành công cụ hữu ích rộng rãi trong lý thuyết thống kê,
chúng hóa ra không có sức mạnh
(như áp dụng trực tiếp)
để giải thích tại sao mạng nơ-ron sâu tổng quát hóa.
Mạng nơ-ron sâu thường có hàng triệu tham số (hoặc hơn),
và có thể dễ dàng gán nhãn ngẫu nhiên cho các tập hợp điểm lớn.
Tuy nhiên, chúng tổng quát hóa tốt trên các bài toán thực tế
và, đáng ngạc nhiên, chúng thường tổng quát hóa tốt hơn,
khi chúng lớn hơn và sâu hơn,
mặc dù có chiều VC cao hơn.
Trong chương tiếp theo, chúng ta sẽ xem xét lại tổng quát hóa
trong ngữ cảnh deep learning.

## Bài tập

1. Nếu chúng ta muốn ước tính lỗi của mô hình cố định $f$
   trong phạm vi $0.0001$ với xác suất lớn hơn 99.9%,
   chúng ta cần bao nhiêu mẫu?
1. Giả sử rằng người khác sở hữu tập kiểm tra có nhãn
   $\mathcal{D}$ và chỉ cung cấp các đầu vào không có nhãn (đặc trưng).
   Bây giờ giả sử rằng bạn chỉ có thể truy cập các nhãn tập kiểm tra
   bằng cách chạy mô hình $f$ (không có ràng buộc nào được đặt ra trên lớp mô hình)
   trên mỗi đầu vào không có nhãn
   và nhận lỗi tương ứng $\epsilon_\mathcal{D}(f)$.
   Bạn cần đánh giá bao nhiêu mô hình
   trước khi rò rỉ toàn bộ tập kiểm tra
   và do đó có thể xuất hiện có lỗi $0$,
   bất kể lỗi thực của bạn là gì?
1. Chiều VC của lớp đa thức bậc năm là bao nhiêu?
1. Chiều VC của hình chữ nhật căn chỉnh theo trục trên dữ liệu hai chiều là bao nhiêu?

[Discussions](https://discuss.d2l.ai/t/6829)
